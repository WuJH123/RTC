from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .large_model_cli import _device, _step2_stats
from .models import DifferentiableHydraulicWorldModel
from .production_cli import _load_graph
from .step2_counterfactual import (
    CounterfactualLossWeights,
    counterfactual_action_loss,
    counterfactual_groups,
    rotated_reference_pairs,
    same_prefix_diagnostic,
)
from .step2_shards import load_shard_manifest, sha256_file
from .train_state import restore_training_state, save_training_state, training_contract_sha
from .training import save_torch_checkpoint


TRAINING_CONTRACT = "STEP2_COUNTERFACTUAL_ACTION_SENSITIVE_TRAINING_V2_VECTOR_BATCHED"
CURRICULUM_CONTRACT = "FLOW_H1_TO_COUNTERFACTUAL_H360_V1"


def _stage(epoch: int, total_epochs: int, full_horizon: int) -> tuple[str, int, bool]:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    p = float(epoch + 1) / float(total_epochs)
    if p <= 0.10:
        return "flow_h1", 1, True
    if p <= 0.20:
        return "joint_h1", 1, False
    if p <= 0.30:
        return "h2", min(2, full_horizon), False
    if p <= 0.40:
        return "h6", min(6, full_horizon), False
    if p <= 0.50:
        return "h12", min(12, full_horizon), False
    if p <= 0.60:
        return "h24", min(24, full_horizon), False
    return "hfull", full_horizon, False


def _precision(device: torch.device, requested: str) -> tuple[bool, torch.dtype, bool, str]:
    mode = requested.lower()
    if mode == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            mode = "bf16"
        else:
            mode = "fp32"
    if mode == "fp32":
        return False, torch.float32, False, mode
    if mode == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested but CUDA BF16 support is unavailable")
        return True, torch.bfloat16, False, mode
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 autocast requires CUDA")
        return True, torch.float16, True, mode
    raise ValueError(f"unsupported precision mode: {requested}")


def _source_kind(group_key: str) -> str:
    raw = group_key.split("::", 1)[0].upper()
    return "D3" if "D3" in raw else "D2"


def _source_group_counts(manifest: dict[str, object]) -> dict[str, int]:
    counts = {"D2": 0, "D3": 0}
    for item in manifest["shards"]:
        with np.load(str(item["path"]), allow_pickle=False) as ds:
            if "source_kind" not in ds.files:
                raise ValueError(
                    "action-sensitive Step2 requires source_kind provenance in every shard"
                )
            for key in counterfactual_groups(ds):
                counts[_source_kind(key)] += 1
    return counts


def _source_weight(source: str, counts: dict[str, int], *, enable_balance: bool) -> float:
    if not enable_balance:
        return 1.0
    positive = {name: count for name, count in counts.items() if count > 0}
    if len(positive) < 2:
        return 1.0
    total = float(sum(positive.values()))
    return total / (len(positive) * float(positive[source]))


def _pair_numpy(
    ds,
    pairs: list[tuple[int, int]],
    horizon: int,
) -> dict[str, np.ndarray]:
    if not pairs:
        raise ValueError("cannot build an empty counterfactual pair batch")
    rows = np.asarray([idx for pair in pairs for idx in pair], dtype=int)
    payload = {
        "initial_state": ds["initial_state"][rows].astype(np.float32),
        "rainfall": ds["rainfall"][rows, :horizon].astype(np.float32),
        "settings": ds["settings"][rows, :horizon].astype(np.float32),
        "previous_actuator_flow": ds["previous_actuator_flow"][rows].astype(np.float32),
        "target_states": ds["target_states"][rows, :horizon].astype(np.float32),
        "target_actuator_flows": ds["target_actuator_flows"][rows, :horizon].astype(
            np.float32
        ),
        "elapsed_seconds": ds["elapsed_seconds"][rows, : horizon + 1].astype(np.float32),
    }
    if "exact_node_flood_volume_m3" in ds.files:
        payload["exact_node_flood_volume_m3"] = ds[
            "exact_node_flood_volume_m3"
        ][rows].astype(np.float32)
    return payload


def _to_device(
    payload: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(value).to(device, non_blocking=True)
        for name, value in payload.items()
    }


def _chunks(values: list[tuple[int, int]], pair_batch_size: int):
    for start in range(0, len(values), pair_batch_size):
        yield values[start : start + pair_batch_size]


def _history_row(
    *,
    epoch: int,
    stage: str,
    horizon: int,
    precision: str,
    totals: dict[str, float],
    pairs: int,
    micro_batches: int,
    optimizer_steps: int,
    grad_norm_mean: float,
    grad_norm_max: float,
) -> dict[str, float | int | str]:
    denominator = max(pairs, 1)
    row: dict[str, float | int | str] = {
        "epoch": epoch + 1,
        "stage": stage,
        "horizon_steps": horizon,
        "precision": precision,
        "pairs": pairs,
        "micro_batches": micro_batches,
        "optimizer_steps": optimizer_steps,
        "gradient_norm_mean": grad_norm_mean,
        "gradient_norm_max": grad_norm_max,
    }
    for name, value in totals.items():
        row[name] = value / denominator
    return row


def train_step2_large_v3_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train action-sensitive Step2 from same-prefix D2/D3 pairs with a "
            "one-step-to-H360 curriculum"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="branch rows per micro-batch; must be even (4 = two counterfactual pairs)",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--actuator-embedding-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exact-flood-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32", "fp16"),
        default="auto",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-state")
    parser.add_argument("--no-resume-training", action="store_true")
    parser.add_argument("--d2-pairs-per-group", type=int, default=8)
    parser.add_argument("--d3-pairs-per-group", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    args = parser.parse_args()

    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("counterfactual Step2 --batch-size must be a positive even number")
    if min(
        args.epochs,
        args.grad_accum,
        args.d2_pairs_per_group,
        args.d3_pairs_per_group,
    ) <= 0:
        raise ValueError("epochs/grad-accum/pair budgets must be positive")
    if args.gradient_clip <= 0 or args.learning_rate <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive")
    pair_batch_size = args.batch_size // 2

    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    manifest = load_shard_manifest(args.manifest)
    if manifest.get("counterfactual_groups_preserved") is not True:
        raise ValueError(
            "action-sensitive Step2 requires V6 group-preserving shards; recompile from "
            "the frozen D2/D3 run index"
        )
    graph = _load_graph(args.graph)
    model_step_seconds = int(manifest["model_step_seconds"])
    full_horizon = int(manifest["horizon_steps"])
    swmm_engine_version = str(manifest["swmm_engine_version"])
    (
        (state_mean, state_std),
        (rain_mean, rain_std),
        (physics_mean, physics_std),
        (static_mean, static_std),
        flow_std,
    ) = _step2_stats(manifest, graph)

    with np.load(str(manifest["shards"][0]["path"]), allow_pickle=False) as first:
        core_config = {
            "state_dim": int(first["initial_state"].shape[-1]),
            "rainfall_dim": int(first["rainfall"].shape[-1]),
            "node_static_dim": int(graph.static_node_features.shape[-1]),
            "actuator_physics_dim": int(graph.actuator_physics.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "actuator_count": len(graph.actuator_ids),
            "actuator_embedding_dim": args.actuator_embedding_dim,
            "direct_action_context": True,
        }

    model = DifferentiableHydraulicWorldModel(**core_config).to(device)
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

    precision_requested = "fp32" if args.no_amp else args.precision
    autocast_enabled, autocast_dtype, use_scaler, precision_mode = _precision(
        device, precision_requested
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    source_counts = _source_group_counts(manifest)

    weights = CounterfactualLossWeights(exact_flood=args.exact_flood_loss_weight)
    train_payload = {
        "manifest_sha256": sha256_file(args.manifest),
        "graph_sha256": sha256_file(args.graph),
        "training_contract": TRAINING_CONTRACT,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "model_step_seconds": model_step_seconds,
        "horizon_steps": full_horizon,
        "swmm_engine_version": swmm_engine_version,
        "hidden_dim": args.hidden_dim,
        "actuator_embedding_dim": args.actuator_embedding_dim,
        "direct_action_context": True,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "precision": precision_mode,
        "gradient_clip": args.gradient_clip,
        "d2_pairs_per_group": args.d2_pairs_per_group,
        "d3_pairs_per_group": args.d3_pairs_per_group,
        "loss_weights": weights.__dict__,
        "seed": args.seed,
    }
    contract_sha, code_sha = training_contract_sha(
        "step2_counterfactual_v3", train_payload
    )
    resume_path = Path(args.resume_state or (str(args.out) + ".trainstate.pt"))
    start_epoch = 0
    history: list[dict[str, object]] = []
    if not args.no_resume_training and resume_path.is_file():
        start_epoch, extra = restore_training_state(
            resume_path,
            expected_contract_sha256=contract_sha,
            expected_code_sha256=code_sha,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        if "numpy_rng_state" in extra:
            rng.bit_generator.state = extra["numpy_rng_state"]
        history = [dict(x) for x in extra.get("history", [])]
    elif args.no_resume_training and resume_path.is_file():
        resume_path.unlink()

    for epoch in range(start_epoch, args.epochs):
        stage, horizon, flow_only = _stage(epoch, args.epochs, full_horizon)
        use_d3 = horizon >= min(24, full_horizon) and not flow_only
        source_balance = (
            use_d3 and source_counts["D2"] > 0 and source_counts["D3"] > 0
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        shard_order = list(manifest["shards"])
        rng.shuffle(shard_order)

        totals = {
            "total_loss": 0.0,
            "absolute_state_loss": 0.0,
            "absolute_flow_loss": 0.0,
            "delta_state_loss": 0.0,
            "delta_flow_loss": 0.0,
            "delta_tfv_loss": 0.0,
            "ranking_loss": 0.0,
            "exact_flood_loss": 0.0,
            "physical_loss": 0.0,
            "sensitivity_ratio": 0.0,
            "sign_accuracy": 0.0,
        }
        pairs_seen = micro_step = optimizer_steps = 0
        grad_norm_sum = grad_norm_max = 0.0

        for item in shard_order:
            with np.load(str(item["path"]), allow_pickle=False) as ds:
                if "source_kind" not in ds.files:
                    raise ValueError(
                        "action-sensitive Step2 requires source_kind provenance in every shard"
                    )
                if "exact_node_flood_volume_m3" not in ds.files:
                    raise ValueError(
                        "action-sensitive Formal Step2 requires exact cumulative flood truth"
                    )

                selected: dict[str, list[tuple[int, int]]] = {"D2": [], "D3": []}
                groups = list(counterfactual_groups(ds).items())
                rng.shuffle(groups)
                for group_key, indices in groups:
                    source = _source_kind(group_key)
                    if source == "D3" and not use_d3:
                        continue
                    budget = (
                        args.d3_pairs_per_group
                        if source == "D3"
                        else args.d2_pairs_per_group
                    )
                    selected[source].extend(
                        rotated_reference_pairs(
                            ds,
                            indices,
                            epoch=epoch,
                            budget=budget,
                        )
                    )

                for source in ("D2", "D3"):
                    source_pairs = selected[source]
                    if not source_pairs:
                        continue
                    rng.shuffle(source_pairs)
                    source_factor = _source_weight(
                        source,
                        source_counts,
                        enable_balance=source_balance,
                    )
                    for pair_rows in _chunks(source_pairs, pair_batch_size):
                        pair_count = len(pair_rows)
                        np_pair = _pair_numpy(ds, pair_rows, horizon)
                        pair = _to_device(np_pair, device)
                        same_prefix_diagnostic(
                            pair["initial_state"],
                            pair["rainfall"],
                            pair["previous_actuator_flow"],
                        )
                        dt = (
                            pair["elapsed_seconds"][:, 1:]
                            - pair["elapsed_seconds"][:, :-1]
                        )
                        if torch.any(dt <= 0) or torch.any(
                            torch.abs(dt - float(model_step_seconds)) > 1e-6
                        ):
                            raise ValueError(
                                "counterfactual pair violates frozen model step"
                            )

                        with torch.autocast(
                            device_type=device.type,
                            dtype=autocast_dtype,
                            enabled=autocast_enabled,
                        ):
                            rollout = model.rollout(
                                pair["initial_state"],
                                pair["rainfall"],
                                pair["settings"],
                                pair["previous_actuator_flow"],
                                up,
                                down,
                                physics,
                                static,
                                edges,
                            )
                            metrics = counterfactual_action_loss(
                                initial_state=pair["initial_state"],
                                rollout_states=rollout.states,
                                rollout_flows=rollout.actuator_flows,
                                target_states=pair["target_states"],
                                target_flows=pair["target_actuator_flows"],
                                exact_node_flood_volume_m3=pair.get(
                                    "exact_node_flood_volume_m3"
                                ),
                                dt_seconds=dt,
                                state_std=model.transition.state_std,
                                flow_std=model.actuator.flow_std,
                                full_horizon=horizon == full_horizon,
                                weights=weights,
                                flow_only=flow_only,
                            )
                            full_loss = metrics.total * source_factor
                            loss = full_loss / args.grad_accum

                        if not torch.isfinite(full_loss):
                            raise RuntimeError(
                                f"non-finite Step2 loss at epoch={epoch + 1}, "
                                f"stage={stage}, source={source}"
                            )
                        scaler.scale(loss).backward()
                        micro_step += 1
                        pairs_seen += pair_count

                        if micro_step % args.grad_accum == 0:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.gradient_clip
                            )
                            if not torch.isfinite(grad_norm):
                                raise RuntimeError(
                                    f"non-finite Step2 gradient norm at epoch={epoch + 1}"
                                )
                            grad_value = float(grad_norm.detach())
                            grad_norm_sum += grad_value
                            grad_norm_max = max(grad_norm_max, grad_value)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1

                        totals["total_loss"] += float(full_loss.detach()) * pair_count
                        totals["absolute_state_loss"] += (
                            float(metrics.absolute_state.detach()) * pair_count
                        )
                        totals["absolute_flow_loss"] += (
                            float(metrics.absolute_flow.detach()) * pair_count
                        )
                        totals["delta_state_loss"] += (
                            float(metrics.delta_state.detach()) * pair_count
                        )
                        totals["delta_flow_loss"] += (
                            float(metrics.delta_flow.detach()) * pair_count
                        )
                        totals["delta_tfv_loss"] += (
                            float(metrics.delta_tfv.detach()) * pair_count
                        )
                        totals["ranking_loss"] += float(metrics.ranking.detach()) * pair_count
                        totals["exact_flood_loss"] += (
                            float(metrics.exact_flood.detach()) * pair_count
                        )
                        totals["physical_loss"] += (
                            float(metrics.physical.detach()) * pair_count
                        )
                        if not flow_only:
                            totals["sensitivity_ratio"] += (
                                float(metrics.sensitivity_ratio.detach()) * pair_count
                            )
                            totals["sign_accuracy"] += (
                                float(metrics.sign_correct.detach()) * pair_count
                            )

        if micro_step % args.grad_accum:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"non-finite Step2 gradient norm at epoch={epoch + 1}"
                )
            grad_value = float(grad_norm.detach())
            grad_norm_sum += grad_value
            grad_norm_max = max(grad_norm_max, grad_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if pairs_seen == 0:
            raise RuntimeError(
                "no counterfactual pairs were available; verify group-preserving D2/D3 shards"
            )
        if flow_only:
            totals["sensitivity_ratio"] = float("nan") * pairs_seen
            totals["sign_accuracy"] = float("nan") * pairs_seen
        row = _history_row(
            epoch=epoch,
            stage=stage,
            horizon=horizon,
            precision=precision_mode,
            totals=totals,
            pairs=pairs_seen,
            micro_batches=micro_step,
            optimizer_steps=optimizer_steps,
            grad_norm_mean=grad_norm_sum / max(optimizer_steps, 1),
            grad_norm_max=grad_norm_max,
        )
        history.append(row)
        save_training_state(
            resume_path,
            contract_sha256=contract_sha,
            rtc_source_tree_sha256=code_sha,
            completed_epochs=epoch + 1,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            extra_state={
                "numpy_rng_state": rng.bit_generator.state,
                "history": history,
            },
        )

    checkpoint_config = {
        **core_config,
        "model_step_seconds": model_step_seconds,
        "horizon_steps": full_horizon,
        "swmm_engine_version": swmm_engine_version,
        "time_contract": "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2",
        "training_contract_sha256": contract_sha,
        "step2_training_contract": TRAINING_CONTRACT,
        "curriculum_contract": CURRICULUM_CONTRACT,
    }
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config=checkpoint_config,
        training_manifest_sha256=sha256_file(args.manifest),
        scientific_split="development",
    )
    history_path = Path(str(args.out) + ".history.json")
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "contract": TRAINING_CONTRACT,
                "checkpoint": meta,
                "completed_epochs": args.epochs,
                "resumed_from_epoch": start_epoch,
                "final": history[-1] if history else None,
                "history": str(history_path),
                "precision": precision_mode,
                "source_group_counts": source_counts,
                "counterfactual_objectives": [
                    "delta_flow",
                    "delta_state",
                    "delta_tfv",
                    "same_checkpoint_ranking",
                ],
                "direct_action_context": True,
                "curriculum": CURRICULUM_CONTRACT,
                "branch_batch_size": args.batch_size,
                "pair_batch_size": pair_batch_size,
                "training_state": str(resume_path),
                "training_contract_sha256": contract_sha,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    train_step2_large_v3_main()
