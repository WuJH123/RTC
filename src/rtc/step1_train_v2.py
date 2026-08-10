from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .large_model_cli import (
    _amp_enabled,
    _device,
    _filtered_index,
    _read_lines,
    _sha,
    _step1_normalization,
)
from .lazy_step1 import CausalStep1TrajectoryDataset, TrajectoryBatchSampler
from .models import SparseStateEstimator
from .production_cli import _load_graph
from .train_state import restore_training_state, save_training_state, training_contract_sha
from .training import save_torch_checkpoint


def train_step1_large_v2_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Step1 lazily with atomic code/data/engine-bound epoch resume"
    )
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-steps", type=int, required=True)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-state")
    parser.add_argument("--no-resume-training", action="store_true")
    parser.add_argument(
        "--samples-per-trajectory",
        type=int,
        default=256,
        help="fixed Step1 epoch budget per D0/D1 trajectory under hydraulic-stratified sampling",
    )
    parser.add_argument(
        "--no-stratified-windows",
        action="store_true",
        help="disable dry/wet/flood-high balanced development/train window sampling",
    )
    args = parser.parse_args()
    if min(
        args.history_steps,
        args.model_step_seconds,
        args.epochs,
        args.batch_size,
        args.grad_accum,
        args.samples_per_trajectory,
    ) <= 0:
        raise ValueError("Step1 timing/training dimensions must be positive")

    torch.manual_seed(args.seed)
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
        history_steps=args.history_steps,
        model_step_seconds=args.model_step_seconds,
        scientific_split="development",
        development_fold="train",
        cache_trajectories=2,
    )
    (
        (obs_mean, obs_std),
        (ctx_mean, ctx_std),
        (state_mean, state_std),
        (static_mean, static_std),
    ) = _step1_normalization(index, graph, sensors)
    stratified = not args.no_stratified_windows
    sampler = TrajectoryBatchSampler(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        stratified=stratified,
        samples_per_trajectory=(args.samples_per_trajectory if stratified else None),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = SparseStateEstimator(
        observed_dim=2,
        static_dim=graph.static_node_features.shape[-1],
        state_dim=6,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        context_dim=5,
        history_steps=args.history_steps,
        model_step_seconds=args.model_step_seconds,
        swmm_engine_version=dataset.swmm_engine_version,
        context_contract="NODE_LOCAL_CAUSAL_CONTEXT_V1",
    ).to(device)
    model.set_normalization(
        observed_mean=torch.as_tensor(obs_mean, device=device),
        observed_std=torch.as_tensor(obs_std, device=device),
        static_mean=torch.as_tensor(static_mean, device=device),
        static_std=torch.as_tensor(static_std, device=device),
        context_mean=torch.as_tensor(ctx_mean, device=device),
        context_std=torch.as_tensor(ctx_std, device=device),
        state_mean=torch.as_tensor(state_mean, device=device),
        state_std=torch.as_tensor(state_std, device=device),
    )
    static = torch.as_tensor(
        graph.static_node_features, dtype=torch.float32, device=device
    )
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp = _amp_enabled(device, not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_payload = {
        "run_index_sha256": _sha(args.run_index),
        "graph_sha256": _sha(args.graph),
        "sensors_sha256": _sha(args.sensors),
        "history_steps": args.history_steps,
        "model_step_seconds": args.model_step_seconds,
        "swmm_engine_version": dataset.swmm_engine_version,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "amp": amp,
        "hydraulic_stratified_windows": stratified,
        "samples_per_trajectory": args.samples_per_trajectory if stratified else None,
        "step1_stratum_weights": sampler.stratum_weights if stratified else None,
        "step1_raw_stratum_counts": dataset.stratum_counts,
    }
    contract_sha, code_sha = training_contract_sha("step1", train_payload)
    resume_path = Path(args.resume_state or (str(args.out) + ".trainstate.pt"))
    start_epoch = 0
    history: list[float] = []
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
        sampler.epoch = int(extra.get("sampler_epoch", start_epoch))
        history = [float(x) for x in extra.get("epoch_loss", [])]
    elif args.no_resume_training and resume_path.is_file():
        resume_path.unlink()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        samples = 0
        for step, (obs, mask, context, target) in enumerate(loader):
            obs, mask, context, target = [
                x.to(device, non_blocking=True) for x in (obs, mask, context, target)
            ]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                pred = model(obs, mask, static, edges, context)
                full_loss = (((pred - target) / model.state_std) ** 2).mean()
                loss = full_loss / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(full_loss.detach()) * obs.shape[0]
            samples += obs.shape[0]
        history.append(running / max(samples, 1))
        save_training_state(
            resume_path,
            contract_sha256=contract_sha,
            rtc_source_tree_sha256=code_sha,
            completed_epochs=epoch + 1,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            extra_state={"sampler_epoch": sampler.epoch, "epoch_loss": history},
        )

    config = {
        "observed_dim": 2,
        "static_dim": int(graph.static_node_features.shape[-1]),
        "state_dim": 6,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "context_dim": 5,
        "history_steps": args.history_steps,
        "model_step_seconds": args.model_step_seconds,
        "swmm_engine_version": dataset.swmm_engine_version,
        "context_contract": "NODE_LOCAL_CAUSAL_CONTEXT_V1",
        "training_contract_sha256": contract_sha,
        "hydraulic_stratified_windows": stratified,
        "samples_per_trajectory": args.samples_per_trajectory if stratified else None,
        "step1_stratum_weights": sampler.stratum_weights if stratified else None,
    }
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config=config,
        training_manifest_sha256=_sha(args.run_index),
        scientific_split="development",
    )
    print(
        json.dumps(
            {
                "contract": "STEP1_TRAINING_V5_HYDRAULIC_STRATA_T0_ENGINE_BOUND",
                "checkpoint": meta,
                "raw_windows": len(dataset),
                "raw_stratum_counts": dataset.stratum_counts,
                "stratified_windows": stratified,
                "samples_per_trajectory_per_epoch": (
                    args.samples_per_trajectory if stratified else None
                ),
                "trajectories": len(dataset.trajectories),
                "swmm_engine_version": dataset.swmm_engine_version,
                "completed_epochs": args.epochs,
                "resumed_from_epoch": start_epoch,
                "final_normalized_mse": history[-1] if history else None,
                "training_state": str(resume_path),
                "training_contract_sha256": contract_sha,
                "device": str(device),
                "amp": amp,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    train_step1_large_v2_main()
