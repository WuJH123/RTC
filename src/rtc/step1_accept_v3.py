from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .code_contract import rtc_source_tree_sha256
from .contracts import load_priority_nodes
from .large_model_cli import _device, _filtered_index, _read_lines
from .lazy_step1 import CausalStep1TrajectoryDataset, TrajectoryBatchSampler
from .production_cli import _load_graph, _load_step1


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _IndexedDataset(Dataset):
    def __init__(self, base: CausalStep1TrajectoryDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        return index, *self.base[index]


def _agg() -> dict[str, float]:
    return {
        "unobs_sse": 0.0,
        "unobs_sum": 0.0,
        "unobs_sum2": 0.0,
        "unobs_n": 0.0,
        "norm_sse": 0.0,
        "norm_n": 0.0,
        "priority_sse": 0.0,
        "priority_sum": 0.0,
        "priority_sum2": 0.0,
        "priority_n": 0.0,
    }


def _nse(sse: float, total: float, total2: float, count: float) -> float:
    if count <= 0:
        return float("nan")
    sst = total2 - total * total / count
    if sst <= 1e-12:
        return float("nan")
    return 1.0 - sse / sst


def accept_step1_large_v3_main() -> None:
    parser = argparse.ArgumentParser(
        description="Accept Step1 on rainfall-group-balanced held-out causal reconstruction"
    )
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--history-steps", type=int, required=True)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device")
    args = parser.parse_args()

    device = _device(args.device)
    graph = _load_graph(args.graph)
    sensors = _read_lines(args.sensors)
    index = _filtered_index(args.run_index, split="development", fold="validation")
    base = CausalStep1TrajectoryDataset(
        index,
        graph=graph,
        sensor_nodes=sensors,
        history_steps=args.history_steps,
        model_step_seconds=args.model_step_seconds,
        scientific_split="development",
        development_fold="validation",
        cache_trajectories=2,
    )
    indexed = _IndexedDataset(base)
    sampler = TrajectoryBatchSampler(
        base, batch_size=args.batch_size, seed=0, shuffle=False
    )
    loader = DataLoader(
        indexed,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = _load_step1(args.model, device)
    runtime = dict(getattr(model, "runtime_metadata", {}))
    if int(runtime.get("history_steps", -1)) != args.history_steps:
        raise ValueError("Step1 checkpoint history_steps differs from validation contract")
    if int(runtime.get("model_step_seconds", -1)) != args.model_step_seconds:
        raise ValueError("Step1 checkpoint model_step_seconds differs from validation contract")

    sensor_idx = np.asarray([graph.node_ids.index(n) for n in sensors], dtype=int)
    unobserved = np.asarray(
        sorted(set(range(len(graph.node_ids))) - set(sensor_idx.tolist())), dtype=int
    )
    if not unobserved.size:
        raise ValueError("Step1 validation requires at least one unobserved node")
    pidx = None
    if args.priority:
        priority = load_priority_nodes(args.priority)
        missing = sorted(set(priority) - set(graph.node_ids))
        if missing:
            raise ValueError(f"priority mapping incompatible with graph: {missing}")
        pidx = np.asarray([graph.node_ids.index(n) for n in priority], dtype=int)

    static = torch.as_tensor(
        graph.static_node_features, dtype=torch.float32, device=device
    )
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    state_std = model.state_std.detach().cpu().numpy()
    groups: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for indices, obs, mask, context, target in loader:
            pred = model(
                obs.to(device),
                mask.to(device),
                static,
                edges,
                context.to(device),
            ).cpu().numpy()
            truth = target.numpy()
            for local, sample_index in enumerate(indices.numpy().astype(int)):
                group_id = base.samples[int(sample_index)].rainfall_group
                agg = groups.setdefault(group_id, _agg())
                y = truth[local, unobserved, 0].astype(float)
                e = pred[local, unobserved, 0].astype(float) - y
                agg["unobs_sse"] += float(np.square(e).sum())
                agg["unobs_sum"] += float(y.sum())
                agg["unobs_sum2"] += float(np.square(y).sum())
                agg["unobs_n"] += float(y.size)
                norm = (pred[local] - truth[local]) / state_std
                agg["norm_sse"] += float(np.square(norm).sum())
                agg["norm_n"] += float(norm.size)
                if pidx is not None:
                    py = truth[local, pidx, 0].astype(float)
                    pe = pred[local, pidx, 0].astype(float) - py
                    agg["priority_sse"] += float(np.square(pe).sum())
                    agg["priority_sum"] += float(py.sum())
                    agg["priority_sum2"] += float(np.square(py).sum())
                    agg["priority_n"] += float(py.size)
    if not groups:
        raise ValueError("Step1 validation contains no rainfall groups")

    group_metrics: list[dict[str, float | str]] = []
    for group_id, agg in sorted(groups.items()):
        row: dict[str, float | str] = {
            "rainfall_group": group_id,
            "unobserved_depth_rmse_m": float(
                np.sqrt(agg["unobs_sse"] / max(agg["unobs_n"], 1.0))
            ),
            "unobserved_depth_nse": _nse(
                agg["unobs_sse"],
                agg["unobs_sum"],
                agg["unobs_sum2"],
                agg["unobs_n"],
            ),
            "all_state_normalized_rmse": float(
                np.sqrt(agg["norm_sse"] / max(agg["norm_n"], 1.0))
            ),
        }
        if pidx is not None:
            row["priority_depth_rmse_m"] = float(
                np.sqrt(agg["priority_sse"] / max(agg["priority_n"], 1.0))
            )
            row["priority_depth_nse"] = _nse(
                agg["priority_sse"],
                agg["priority_sum"],
                agg["priority_sum2"],
                agg["priority_n"],
            )
        group_metrics.append(row)

    def mean_finite(name: str) -> float:
        values = np.asarray([float(row[name]) for row in group_metrics], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            raise ValueError(f"no finite rainfall-group values for Step1 metric {name}")
        return float(values.mean())

    metrics = {
        "unobserved_depth_rmse_m": mean_finite("unobserved_depth_rmse_m"),
        "unobserved_depth_nse": mean_finite("unobserved_depth_nse"),
        "all_state_normalized_rmse": mean_finite("all_state_normalized_rmse"),
    }
    if pidx is not None:
        metrics["priority_depth_rmse_m"] = mean_finite("priority_depth_rmse_m")
        metrics["priority_depth_nse"] = mean_finite("priority_depth_nse")
    payload = {
        "contract": "STEP1_HELDOUT_ACCEPTANCE_V3_GROUP_BALANCED_TIME_LOCKED",
        "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        "model_sha256": _sha(args.model),
        "run_index_sha256": _sha(args.run_index),
        "model_step_seconds": args.model_step_seconds,
        "history_steps": args.history_steps,
        "validation_windows": len(base),
        "rainfall_groups": len(group_metrics),
        "aggregation": "equal_weight_per_rainfall_group",
        "metrics": metrics,
        "priority_diagnostic_only": True,
        "group_metrics": group_metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    accept_step1_large_v3_main()
