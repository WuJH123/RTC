from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .acceptance import (
    apply_metric_thresholds,
    event_balanced_metric,
    evaluate_step1_depth,
    mae,
    rank_correlation,
    rmse,
)
from .contracts import load_priority_nodes
from .dataset_compile import compile_branches_to_npz
from .models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from .training import save_torch_checkpoint, train_step1, train_step2


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _channel_weights(target: np.ndarray) -> np.ndarray:
    axes = tuple(range(target.ndim - 1))
    variance = np.var(target, axis=axes)
    weights = 1.0 / np.maximum(variance, 1e-8)
    return weights / np.mean(weights)


def _thresholds(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        {str(k): float(v) for k, v in payload.get("minimum", {}).items()},
        {str(k): float(v) for k, v in payload.get("maximum", {}).items()},
    )


def _write_acceptance(
    out: str | Path,
    *,
    contract: str,
    metrics: dict[str, float],
    thresholds_path: str | Path,
    model_path: str | Path,
    dataset_path: str | Path,
) -> None:
    minimum, maximum = _thresholds(thresholds_path)
    result = apply_metric_thresholds(metrics, minimum=minimum, maximum=maximum)
    payload = {
        "contract": contract,
        "passed": result.passed,
        "failed_metrics": list(result.failed_metrics),
        "metrics": result.metrics,
        "thresholds": {"minimum": minimum, "maximum": maximum},
        "model_path": str(Path(model_path)),
        "model_sha256": _sha(model_path),
        "dataset_path": str(Path(dataset_path)),
        "dataset_sha256": _sha(dataset_path),
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not result.passed:
        raise SystemExit(2)


def compile_step2_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Compile lineage-preserving Step2 dataset")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--development-fold", choices=["train", "validation", "all"], default="train")
    args = parser.parse_args()
    index = pd.read_csv(args.run_index)
    if "metadata_path" not in index.columns:
        raise ValueError("run index requires metadata_path")
    if "scientific_split" in index.columns and args.split:
        index = index[index["scientific_split"].astype(str) == args.split]
    if (
        args.split == "development"
        and args.development_fold != "all"
        and "development_fold" in index.columns
    ):
        index = index[index["development_fold"].astype(str) == args.development_fold]
    if index.empty:
        raise ValueError("no branches remain after split/fold filtering")
    compile_branches_to_npz(
        index["metadata_path"].astype(str).tolist(),
        args.out,
        provenance=index.reset_index(drop=True),
    )
    print(json.dumps({
        "branches": len(index),
        "split": args.split,
        "development_fold": args.development_fold,
        "out": args.out,
    }, indent=2))


def train_step1_main() -> None:
    parser = argparse.ArgumentParser(description="Train Step1 on development/train rainfall groups only")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    args = parser.parse_args()
    ds = np.load(args.dataset, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    split = ds["scientific_splits"].astype(str)
    fold = ds["development_folds"].astype(str)
    use = (split == "development") & (fold == "train")
    if not np.any(use):
        raise ValueError("Step1 dataset has no development/train windows")
    weights = _channel_weights(ds["target_state"][use])
    config = {
        "observed_dim": int(ds["observed_history"].shape[-1]),
        "static_dim": int(graph["static_node_features"].shape[-1]),
        "state_dim": int(ds["target_state"].shape[-1]),
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "context_dim": int(ds["context_history"].shape[-1]),
    }
    model = SparseStateEstimator(**config)
    history = train_step1(
        model,
        observed_history=ds["observed_history"][use],
        observation_mask=ds["observation_mask"][use],
        context_history=ds["context_history"][use],
        target_state=ds["target_state"][use],
        static_node_features=graph["static_node_features"],
        edge_index=graph["edge_index"],
        state_weights=weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config={**config, "state_weights": weights.tolist()},
        training_manifest_sha256=_sha(args.dataset),
    )
    print(json.dumps({
        "train_windows": int(use.sum()),
        "checkpoint": meta,
        "final_loss": history.epoch_loss[-1],
    }, indent=2))


def _load_step1(path: str | Path, device: torch.device) -> SparseStateEstimator:
    payload = torch.load(path, map_location=device)
    if payload.get("scientific_split") != "development":
        raise ValueError("Step1 checkpoint lineage is not development-only")
    cfg = dict(payload["model_config"])
    cfg.pop("state_weights", None)
    model = SparseStateEstimator(**cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def accept_step1_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step1 on held-out development/validation rainfall groups")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    ds = np.load(args.dataset, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    use = (ds["scientific_splits"].astype(str) == "development") & (
        ds["development_folds"].astype(str) == "validation"
    )
    if not np.any(use):
        raise ValueError("Step1 dataset has no development/validation windows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_step1(args.model, device)
    static = torch.as_tensor(graph["static_node_features"], dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
    indices = np.where(use)[0]
    pred: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start : start + args.batch_size]
            value = model(
                torch.as_tensor(ds["observed_history"][idx], dtype=torch.float32, device=device),
                torch.as_tensor(ds["observation_mask"][idx], dtype=torch.float32, device=device),
                static,
                edges,
                torch.as_tensor(ds["context_history"][idx], dtype=torch.float32, device=device),
            )
            pred.append(value.cpu().numpy())
    predicted = np.concatenate(pred, axis=0)
    truth = ds["target_state"][indices]
    node_ids = tuple(ds["node_ids"].astype(str).tolist())
    node_index = {node: i for i, node in enumerate(node_ids)}
    priority = load_priority_nodes(args.priority)
    missing = sorted(set(priority) - set(node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from Step1 dataset: {missing}")
    priority_idx = np.array([node_index[n] for n in priority], dtype=int)
    sensor_seen = ds["observation_mask"][indices[0], -1, :, 0] > 0
    unobserved = np.where(~sensor_seen)[0]
    metrics = evaluate_step1_depth(
        predicted_depth=predicted[..., 0],
        true_depth=truth[..., 0],
        event_ids=ds["event_ids"][indices],
        unobserved_node_indices=unobserved,
        priority_node_indices=priority_idx,
    )
    _write_acceptance(
        args.out,
        contract="STEP1_HELDOUT_RAINFALL_ACCEPTANCE_V1",
        metrics=metrics,
        thresholds_path=args.thresholds,
        model_path=args.model,
        dataset_path=args.dataset,
    )


def train_step2_main() -> None:
    parser = argparse.ArgumentParser(description="Train Step2 on development/train D2+D3 branches")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=160)
    args = parser.parse_args()
    ds = np.load(args.dataset, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    if "scientific_split" in ds.files and not np.all(ds["scientific_split"].astype(str) == "development"):
        raise ValueError("Step2 training dataset contains non-development branches")
    if "development_fold" in ds.files and not np.all(ds["development_fold"].astype(str) == "train"):
        raise ValueError("Step2 training dataset contains non-train development branches")
    config = {
        "state_dim": int(ds["initial_state"].shape[-1]),
        "rainfall_dim": int(ds["rainfall"].shape[-1]),
        "node_static_dim": int(graph["static_node_features"].shape[-1]),
        "actuator_physics_dim": int(graph["actuator_physics"].shape[-1]),
        "hidden_dim": args.hidden_dim,
    }
    model = DifferentiableHydraulicWorldModel(**config)
    state_weights = _channel_weights(ds["target_states"])
    physics = np.repeat(graph["actuator_physics"][None], ds["initial_state"].shape[0], axis=0)
    state_var_by_channel = np.var(
        ds["target_states"], axis=tuple(range(ds["target_states"].ndim - 1))
    )
    weighted_state_var = float(np.mean(state_var_by_channel * state_weights))
    flow_var = float(np.var(ds["target_actuator_flows"]))
    flow_weight = weighted_state_var / max(flow_var, 1e-8)
    history = train_step2(
        model,
        initial_state=ds["initial_state"],
        rainfall=ds["rainfall"],
        settings=ds["settings"],
        previous_actuator_flow=ds["previous_actuator_flow"],
        actuator_physics=physics,
        target_states=ds["target_states"],
        target_actuator_flows=ds["target_actuator_flows"],
        actuator_upstream=graph["actuator_upstream"],
        actuator_downstream=graph["actuator_downstream"],
        static_node_features=graph["static_node_features"],
        edge_index=graph["edge_index"],
        state_weights=state_weights,
        flow_loss_weight=flow_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config={
            **config,
            "state_weights": state_weights.tolist(),
            "flow_loss_weight": flow_weight,
        },
        training_manifest_sha256=_sha(args.dataset),
    )
    print(json.dumps({"checkpoint": meta, "final_loss": history.epoch_loss[-1]}, indent=2))


def _load_step2(path: str | Path, device: torch.device) -> DifferentiableHydraulicWorldModel:
    payload = torch.load(path, map_location=device)
    if payload.get("scientific_split") != "development":
        raise ValueError("Step2 checkpoint lineage is not development-only")
    cfg = dict(payload["model_config"])
    cfg.pop("state_weights", None)
    cfg.pop("flow_loss_weight", None)
    model = DifferentiableHydraulicWorldModel(**cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def _rollout_step2(
    model: DifferentiableHydraulicWorldModel,
    ds: np.lib.npyio.NpzFile,
    graph: np.lib.npyio.NpzFile,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    flows: list[np.ndarray] = []
    count = ds["initial_state"].shape[0]
    up = torch.as_tensor(graph["actuator_upstream"], dtype=torch.long, device=device)
    down = torch.as_tensor(graph["actuator_downstream"], dtype=torch.long, device=device)
    static = torch.as_tensor(graph["static_node_features"], dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
    base_physics = torch.as_tensor(graph["actuator_physics"], dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            b = end - start
            rollout = model.rollout(
                torch.as_tensor(ds["initial_state"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(ds["rainfall"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(ds["settings"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(ds["previous_actuator_flow"][start:end], dtype=torch.float32, device=device),
                up,
                down,
                base_physics.unsqueeze(0).expand(b, -1, -1),
                static,
                edges,
            )
            states.append(rollout.states.cpu().numpy())
            flows.append(rollout.actuator_flows.cpu().numpy())
    return np.concatenate(states), np.concatenate(flows)


def accept_step2_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step2 on held-out development/validation SWMM branches")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    ds = np.load(args.dataset, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    if "development_fold" in ds.files and not np.all(ds["development_fold"].astype(str) == "validation"):
        raise ValueError("Step2 acceptance dataset must contain development/validation only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_step2(args.model, device)
    pred_state, pred_flow = _rollout_step2(
        model, ds, graph, batch_size=args.batch_size, device=device
    )
    truth = ds["target_states"]
    true_flow = ds["target_actuator_flows"]
    event_ids = ds["event_id"].astype(str) if "event_id" in ds.files else np.arange(len(truth)).astype(str)
    times = ds["elapsed_seconds"].astype(float)
    dt = np.diff(times, axis=1)
    if np.any(dt <= 0):
        raise ValueError("Step2 time grids must be strictly increasing")
    pred_flood = np.clip(pred_state[..., 2], 0.0, None)
    true_flood = np.clip(truth[..., 2], 0.0, None)
    pred_node_volume = (pred_flood * dt[:, :, None]).sum(axis=1)
    true_node_volume = (true_flood * dt[:, :, None]).sum(axis=1)
    pred_tfv = pred_node_volume.sum(axis=1)
    true_tfv = true_node_volume.sum(axis=1)
    node_ids = tuple(ds["node_ids"].astype(str).tolist())
    node_index = {node: i for i, node in enumerate(node_ids)}
    priority = load_priority_nodes(args.priority)
    missing = sorted(set(priority) - set(node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from Step2 dataset: {missing}")
    pidx = np.array([node_index[n] for n in priority], dtype=int)
    pred_pfv = pred_node_volume[:, pidx].sum(axis=1)
    true_pfv = true_node_volume[:, pidx].sum(axis=1)
    pred_peak = pred_flood.sum(axis=2).max(axis=1)
    true_peak = true_flood.sum(axis=2).max(axis=1)
    metrics = {
        "depth_rmse_m": rmse(pred_state[..., 0], truth[..., 0]),
        "flood_rate_rmse_m3s": rmse(pred_state[..., 2], truth[..., 2]),
        "node_volume_rmse_m3": rmse(pred_state[..., 3], truth[..., 3]),
        "managed_flow_rmse_m3s": rmse(pred_flow, true_flow),
        "horizon_end_depth_rmse_m": rmse(pred_state[:, -1, :, 0], truth[:, -1, :, 0]),
        "tfv_mae_m3": mae(pred_tfv, true_tfv),
        "priority_flood_volume_mae_m3": mae(pred_pfv, true_pfv),
        "global_peak_flood_rate_mae_m3s": mae(pred_peak, true_peak),
        "tfv_rank_correlation": rank_correlation(pred_tfv, true_tfv),
        "priority_flood_rank_correlation": rank_correlation(pred_pfv, true_pfv),
        "event_balanced_depth_rmse_m": event_balanced_metric(
            pred_state[..., 0], truth[..., 0], event_ids, rmse
        ),
        "event_balanced_tfv_mae_m3": event_balanced_metric(
            pred_tfv[:, None], true_tfv[:, None], event_ids, mae
        ),
    }
    _write_acceptance(
        args.out,
        contract="STEP2_HELDOUT_TRAJECTORY_ACCEPTANCE_V1",
        metrics=metrics,
        thresholds_path=args.thresholds,
        model_path=args.model,
        dataset_path=args.dataset,
    )
