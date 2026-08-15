"""Development-only Step1 long-range ablation on identical held-out validation windows.

Compare the accepted frozen legacy SparseStateEstimator with a separately trained V122
sensor-to-all-node attention checkpoint.  Both receive the same graph, sensor layout,
13-frame causal history, node-local causal context, and Development validation fold.
The report is diagnostic only: it cannot replace the frozen Step1 or rebuild the causal
Step2 state store automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rtc.large_model_cli import _device, _filtered_index, _read_lines
from rtc.lazy_step1 import CausalStep1TrajectoryDataset, TrajectoryBatchSampler
from rtc.production_cli import _load_graph
from rtc.spatial_diagnostics_v128 import DEFAULT_DISTANCE_BINS, nearest_source_hops, distance_bin_name
from rtc.step1_runtime_v122 import load_step1_v122
from rtc.step1_runtime_v127 import load_frozen_step1_v127

CONTRACT = "PROJECT7_STEP1_GLOBAL_ATTENTION_DISTANCE_ABLATION_V1_DEVELOPMENT_ONLY"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_agg() -> dict[str, float]:
    return {"sse": 0.0, "sae": 0.0, "sum": 0.0, "sum2": 0.0, "n": 0.0}


def _acc(agg: dict[str, float], truth: np.ndarray, pred: np.ndarray) -> None:
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    e = p - y
    agg["sse"] += float(np.square(e).sum())
    agg["sae"] += float(np.abs(e).sum())
    agg["sum"] += float(y.sum())
    agg["sum2"] += float(np.square(y).sum())
    agg["n"] += float(y.size)


def _metric(agg: dict[str, float]) -> dict[str, float | int]:
    n = float(agg["n"])
    if n <= 0:
        return {"samples": 0, "rmse_m": float("nan"), "mae_m": float("nan"), "nse": float("nan")}
    sst = float(agg["sum2"] - agg["sum"] * agg["sum"] / n)
    return {
        "samples": int(n),
        "rmse_m": float(np.sqrt(agg["sse"] / n)),
        "mae_m": float(agg["sae"] / n),
        "nse": float(1.0 - agg["sse"] / sst) if sst > 1e-12 else float("nan"),
    }


def _group_balanced(groups: dict[str, dict[str, dict[str, float]]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for bin_name, by_group in groups.items():
        rows = [_metric(agg) for _, agg in sorted(by_group.items())]
        finite_rmse = [float(row["rmse_m"]) for row in rows if np.isfinite(float(row["rmse_m"]))]
        finite_mae = [float(row["mae_m"]) for row in rows if np.isfinite(float(row["mae_m"]))]
        finite_nse = [float(row["nse"]) for row in rows if np.isfinite(float(row["nse"]))]
        result[bin_name] = {
            "rainfall_groups": len(rows),
            "aggregation": "equal_weight_per_rainfall_group",
            "rmse_m": float(np.mean(finite_rmse)) if finite_rmse else float("nan"),
            "mae_m": float(np.mean(finite_mae)) if finite_mae else float("nan"),
            "nse": float(np.mean(finite_nse)) if finite_nse else float("nan"),
            "group_metrics": rows,
        }
    return result


def _far_summary(metrics: dict[str, object]) -> dict[str, float | bool]:
    near = float(dict(metrics.get("1-3", {})).get("rmse_m", float("nan")))
    far_7 = float(dict(metrics.get("7-12", {})).get("rmse_m", float("nan")))
    far_13 = float(dict(metrics.get("13+", {})).get("rmse_m", float("nan")))
    return {
        "near_1_3_rmse_m": near,
        "far_7_12_rmse_m": far_7,
        "far_13_plus_rmse_m": far_13,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-index", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--legacy-model", required=True)
    p.add_argument("--attention-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device")
    args = p.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = _device(args.device)
    graph = _load_graph(args.graph)
    sensors = _read_lines(args.sensors)
    missing = sorted(set(sensors) - set(graph.node_ids))
    if missing:
        raise ValueError(f"sensor nodes absent from graph: {missing}")
    sensor_idx = np.asarray([graph.node_ids.index(node) for node in sensors], dtype=np.int64)
    distance = nearest_source_hops(graph.edge_index, len(graph.node_ids), sensor_idx)

    index = _filtered_index(args.run_index, split="development", fold="validation")
    dataset = CausalStep1TrajectoryDataset(
        index,
        graph=graph,
        sensor_nodes=sensors,
        history_steps=13,
        model_step_seconds=300,
        scientific_split="development",
        development_fold="validation",
        cache_trajectories=2,
    )
    sampler = TrajectoryBatchSampler(dataset, batch_size=args.batch_size, seed=0, shuffle=False, stratified=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)

    legacy = load_frozen_step1_v127(args.legacy_model, device)
    attention = load_step1_v122(args.attention_model, device)
    for label, model in (("legacy", legacy), ("attention", attention)):
        runtime = dict(getattr(model, "runtime_metadata", {}))
        if int(runtime.get("history_steps", -1)) != 13 or int(runtime.get("model_step_seconds", -1)) != 300:
            raise ValueError(f"{label} Step1 checkpoint violates 13-frame/300-s contract")
        if str(runtime.get("swmm_engine_version", "")) != dataset.swmm_engine_version:
            raise ValueError(f"{label} Step1 SWMM engine differs from validation dataset")

    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "legacy": {distance_bin_name(lo, hi): {} for lo, hi in DEFAULT_DISTANCE_BINS},
        "attention": {distance_bin_name(lo, hi): {} for lo, hi in DEFAULT_DISTANCE_BINS},
    }

    with torch.no_grad():
        sample_cursor = 0
        for obs, mask, context, target in loader:
            obs_d = obs.to(device)
            mask_d = mask.to(device)
            context_d = context.to(device)
            predictions = {
                "legacy": legacy(obs_d, mask_d, static, edges, context_d).cpu().numpy(),
                "attention": attention(obs_d, mask_d, static, edges, context_d).cpu().numpy(),
            }
            truth = target.numpy()
            for local in range(int(target.shape[0])):
                ref = dataset.samples[sample_cursor + local]
                group_id = str(ref.rainfall_group)
                for lo, hi in DEFAULT_DISTANCE_BINS:
                    if lo == 0:
                        continue  # configured sensor nodes are measured, not reconstructed.
                    node_mask = distance >= lo
                    if hi is not None:
                        node_mask &= distance <= hi
                    if not np.any(node_mask):
                        continue
                    bin_name = distance_bin_name(lo, hi)
                    for label in ("legacy", "attention"):
                        agg = aggregate[label][bin_name].setdefault(group_id, _new_agg())
                        _acc(agg, truth[local, node_mask, 0], predictions[label][local, node_mask, 0])
            sample_cursor += int(target.shape[0])

    metrics = {label: _group_balanced(groups) for label, groups in aggregate.items()}
    comparison: dict[str, object] = {}
    for lo, hi in DEFAULT_DISTANCE_BINS:
        if lo == 0:
            continue
        key = distance_bin_name(lo, hi)
        legacy_rmse = float(dict(metrics["legacy"].get(key, {})).get("rmse_m", float("nan")))
        attention_rmse = float(dict(metrics["attention"].get(key, {})).get("rmse_m", float("nan")))
        comparison[key] = {
            "legacy_rmse_m": legacy_rmse,
            "attention_rmse_m": attention_rmse,
            "relative_rmse_change": (
                float((attention_rmse - legacy_rmse) / legacy_rmse)
                if np.isfinite(legacy_rmse) and np.isfinite(attention_rmse) and legacy_rmse > 0
                else float("nan")
            ),
            "attention_better": bool(np.isfinite(legacy_rmse) and np.isfinite(attention_rmse) and attention_rmse < legacy_rmse),
        }

    payload = {
        "contract": CONTRACT,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "automatic_causal_state_store_rebuild": False,
        "graph_sha256": _sha(args.graph),
        "sensor_sha256": _sha(args.sensors),
        "run_index_sha256": _sha(args.run_index),
        "legacy_model_sha256": _sha(args.legacy_model),
        "attention_model_sha256": _sha(args.attention_model),
        "validation_windows": len(dataset),
        "distance_definition": "undirected shortest hops to nearest configured sensor",
        "metrics": metrics,
        "comparison": comparison,
        "legacy_far_summary": _far_summary(metrics["legacy"]),
        "attention_far_summary": _far_summary(metrics["attention"]),
        "promotion_rule": (
            "Do not replace frozen Step1 unless held-out far-distance bins improve without "
            "material near-field/overall regression; if promoted, rebuild causal state store "
            "and retrain Step2 from the beginning."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
