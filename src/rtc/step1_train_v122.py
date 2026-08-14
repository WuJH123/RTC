from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .large_model_cli import _amp_enabled, _device, _filtered_index, _read_lines, _sha, _step1_normalization
from .lazy_step1 import CausalStep1TrajectoryDataset, TrajectoryBatchSampler
from .production_cli import _load_graph
from .step1_loss_v122 import Step1LossContractV122, apply_sensor_dropout_v122, step1_loss_v122
from .step1_model_v122 import STEP1_V122_CONTRACT, SparseStateEstimatorV122
from .training import save_torch_checkpoint


def train_step1_v122_main() -> None:
    parser = argparse.ArgumentParser(description="Train Project7 V122 sparse-state reconstructor")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--samples-per-trajectory", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sensor-dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum, args.samples_per_trajectory) <= 0:
        raise ValueError("V122 Step1 training dimensions must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    graph = _load_graph(args.graph)
    sensors = _read_lines(args.sensors)
    index = _filtered_index(args.run_index, split="development", fold="train")
    dataset = CausalStep1TrajectoryDataset(
        index,
        graph=graph,
        sensor_nodes=sensors,
        history_steps=13,
        model_step_seconds=300,
        scientific_split="development",
        development_fold="train",
        cache_trajectories=2,
    )
    sampler = TrajectoryBatchSampler(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        stratified=True,
        samples_per_trajectory=args.samples_per_trajectory,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    (obs_norm, ctx_norm, state_norm, static_norm) = _step1_normalization(index, graph, sensors)
    obs_mean, obs_std = obs_norm
    ctx_mean, ctx_std = ctx_norm
    state_mean, state_std = state_norm
    static_mean, static_std = static_norm

    model = SparseStateEstimatorV122(
        observed_dim=2,
        static_dim=graph.static_node_features.shape[-1],
        state_dim=6,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        context_dim=5,
        attention_heads=args.attention_heads,
        hard_sensor_consistency=True,
        history_steps=13,
        model_step_seconds=300,
        swmm_engine_version=dataset.swmm_engine_version,
        context_contract="NODE_LOCAL_CAUSAL_CONTEXT_V1",
    ).to(device)
    model.set_normalization(
        observed_mean=torch.as_tensor(obs_mean, device=device),
        observed_std=torch.as_tensor(obs_std, device=device),
        context_mean=torch.as_tensor(ctx_mean, device=device),
        context_std=torch.as_tensor(ctx_std, device=device),
        state_mean=torch.as_tensor(state_mean, device=device),
        state_std=torch.as_tensor(state_std, device=device),
        static_mean=torch.as_tensor(static_mean, device=device),
        static_std=torch.as_tensor(static_std, device=device),
    )
    loss_contract = Step1LossContractV122(sensor_dropout_probability=args.sensor_dropout)
    loss_contract.validate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = _amp_enabled(device, not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        samples = 0
        for step, (obs, mask, context, target) in enumerate(loader):
            obs, mask, context, target = [x.to(device, non_blocking=True) for x in (obs, mask, context, target)]
            obs, mask = apply_sensor_dropout_v122(obs, mask, probability=loss_contract.sensor_dropout_probability)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(obs, mask, static, edges, context)
                full_loss, metrics = step1_loss_v122(prediction, target, mask, model.state_std, contract=loss_contract)
                loss = full_loss / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            weight = int(obs.shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight
            samples += weight
        row = {key: value / max(samples, 1) for key, value in totals.items()}
        row["epoch"] = float(epoch + 1)
        history.append(row)
        print("[STEP1_V122] " + " ".join(f"{key}={value:.6g}" for key, value in row.items()), flush=True)

    config = {
        "step1_contract": STEP1_V122_CONTRACT,
        "observed_dim": 2,
        "static_dim": int(graph.static_node_features.shape[-1]),
        "state_dim": 6,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "context_dim": 5,
        "attention_heads": args.attention_heads,
        "hard_sensor_consistency": True,
        "history_steps": 13,
        "model_step_seconds": 300,
        "swmm_engine_version": dataset.swmm_engine_version,
        "context_contract": "NODE_LOCAL_CAUSAL_CONTEXT_V1",
        "sensor_dropout_probability": args.sensor_dropout,
        "loss_contract": loss_contract.__dict__,
    }
    checkpoint = save_torch_checkpoint(
        model,
        args.out,
        model_config=config,
        training_manifest_sha256=_sha(args.run_index),
        scientific_split="development",
    )
    report = {
        "contract": STEP1_V122_CONTRACT,
        "checkpoint": checkpoint,
        "training_history": history,
        "raw_stratum_counts": dataset.stratum_counts,
        "stratum_weights": sampler.stratum_weights,
        "trajectories": len(dataset.trajectories),
        "swmm_engine_version": dataset.swmm_engine_version,
        "device": str(device),
        "amp": amp,
    }
    report_path = Path(str(args.out) + ".v122.report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    train_step1_v122_main()
