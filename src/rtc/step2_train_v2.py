from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .flood_volume import trapezoid_node_flood_volume
from .large_model_cli import _amp_enabled, _device, _step2_stats
from .models import DifferentiableHydraulicWorldModel
from .production_cli import _load_graph
from .step2_shards import load_shard_manifest, sha256_file
from .training import save_torch_checkpoint


def train_step2_large_v2_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Step2 with state/flow supervision plus exact cumulative SWMM flooding-volume labels"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--actuator-embedding-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exact-flood-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.exact_flood_loss_weight <= 0:
        raise ValueError("exact-flood-loss-weight must be positive for Formal Step2")

    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    graph = _load_graph(args.graph)
    manifest = load_shard_manifest(args.manifest)
    (
        (state_mean, state_std),
        (rain_mean, rain_std),
        (physics_mean, physics_std),
        (static_mean, static_std),
        flow_std,
    ) = _step2_stats(manifest, graph)
    with np.load(str(manifest["shards"][0]["path"]), allow_pickle=False) as first:
        config = {
            "state_dim": int(first["initial_state"].shape[-1]),
            "rainfall_dim": int(first["rainfall"].shape[-1]),
            "node_static_dim": int(graph.static_node_features.shape[-1]),
            "actuator_physics_dim": int(graph.actuator_physics.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "actuator_count": len(graph.actuator_ids),
            "actuator_embedding_dim": args.actuator_embedding_dim,
        }

    model = DifferentiableHydraulicWorldModel(**config).to(device)
    model.set_normalization(
        state_mean=torch.as_tensor(state_mean, device=device),
        state_std=torch.as_tensor(state_std, device=device),
        rain_mean=torch.as_tensor(rain_mean, device=device),
        rain_std=torch.as_tensor(rain_std, device=device),
        static_mean=torch.as_tensor(static_mean, device=device),
        static_std=torch.as_tensor(static_std, device=device),
        physics_mean=torch.as_tensor(physics_mean, device=device),
        physics_std=torch.as_tensor(physics_std, device=device),
        flow_std=torch.as_tensor(flow_std, device=device),
    )
    up = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    down = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    physics = torch.as_tensor(graph.actuator_physics, dtype=torch.float32, device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp = _amp_enabled(device, not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []

    for _epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        shard_order = list(manifest["shards"])
        rng.shuffle(shard_order)
        loss_sum = state_sum = flow_sum = flood_sum = 0.0
        seen = micro_step = 0
        for item in shard_order:
            with np.load(str(item["path"]), allow_pickle=False) as ds:
                if "exact_node_flood_volume_m3" not in ds.files:
                    raise ValueError(
                        f"Formal Step2 training requires exact cumulative flood truth in shard: {item['path']}"
                    )
                arrays = [
                    torch.from_numpy(ds[name].astype(np.float32))
                    for name in (
                        "initial_state",
                        "rainfall",
                        "settings",
                        "previous_actuator_flow",
                        "target_states",
                        "target_actuator_flows",
                        "exact_node_flood_volume_m3",
                        "elapsed_seconds",
                    )
                ]
            loader = DataLoader(
                TensorDataset(*arrays),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for (
                initial,
                rain,
                settings,
                prev,
                target_state,
                target_flow,
                exact_volume,
                elapsed,
            ) in loader:
                initial, rain, settings, prev, target_state, target_flow, exact_volume, elapsed = [
                    x.to(device, non_blocking=True)
                    for x in (
                        initial,
                        rain,
                        settings,
                        prev,
                        target_state,
                        target_flow,
                        exact_volume,
                        elapsed,
                    )
                ]
                batch = initial.shape[0]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    rollout = model.rollout(
                        initial,
                        rain,
                        settings,
                        prev,
                        up,
                        down,
                        physics.unsqueeze(0).expand(batch, -1, -1),
                        static,
                        edges,
                    )
                    state_loss = (
                        ((rollout.states - target_state) / model.transition.state_std) ** 2
                    ).mean()
                    flow_loss = (
                        ((rollout.actuator_flows - target_flow) / model.actuator.flow_std) ** 2
                    ).mean()
                    dt = elapsed[:, 1:] - elapsed[:, :-1]
                    if torch.any(dt <= 0):
                        raise ValueError("Step2 shard contains a non-increasing time grid")
                    # IMPORTANT: this is exactly the same physical-volume operator used by
                    # MPC, gradient truth and ranking evaluation. The checkpoint/current
                    # flooding rate is the left endpoint; predicted future states are the
                    # right endpoints. A future-only rectangle would train a different TFV
                    # objective from the one later optimized online.
                    pred_node_volume = trapezoid_node_flood_volume(
                        initial,
                        rollout.states,
                        flood_rate_index=2,
                        dt_seconds=dt,
                    )
                    exact = exact_volume.clamp_min(0.0)
                    node_volume_loss = torch.square(
                        torch.log1p(pred_node_volume) - torch.log1p(exact)
                    ).mean()
                    total_volume_loss = torch.square(
                        torch.log1p(pred_node_volume.sum(dim=-1))
                        - torch.log1p(exact.sum(dim=-1))
                    ).mean()
                    exact_flood_loss = node_volume_loss + total_volume_loss
                    full_loss = (
                        state_loss
                        + flow_loss
                        + args.exact_flood_loss_weight * exact_flood_loss
                    )
                    loss = full_loss / args.grad_accum
                scaler.scale(loss).backward()
                micro_step += 1
                if micro_step % args.grad_accum == 0:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)
                n = int(batch)
                loss_sum += float(full_loss.detach()) * n
                state_sum += float(state_loss.detach()) * n
                flow_sum += float(flow_loss.detach()) * n
                flood_sum += float(exact_flood_loss.detach()) * n
                seen += n
        if micro_step % args.grad_accum:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
        history.append(
            {
                "total": loss_sum / max(seen, 1),
                "state": state_sum / max(seen, 1),
                "flow": flow_sum / max(seen, 1),
                "exact_flood": flood_sum / max(seen, 1),
            }
        )

    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config=config,
        training_manifest_sha256=sha256_file(args.manifest),
        scientific_split="development",
    )
    print(
        json.dumps(
            {
                "contract": "STEP2_TRAINING_STATE_FLOW_EXACT_CUMULATIVE_FLOOD_V3_TRAPEZOID",
                "checkpoint": meta,
                "final_losses": history[-1],
                "exact_flood_loss_weight": args.exact_flood_loss_weight,
                "prediction_volume_integration": "trapezoid_current_plus_future_flooding_rate",
                "truth_source_tfv": "SWMM_NODE_STATISTICS_CUMULATIVE_EXACT_HORIZON",
                "device": str(device),
                "amp": amp,
                "micro_batch": args.batch_size,
                "grad_accum": args.grad_accum,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    train_step2_large_v2_main()
