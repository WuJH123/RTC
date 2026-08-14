"""Train-only parity audit for true-state versus sparse-Step1 RBC first moves."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rtc.lazy_step1 import CausalStep1TrajectoryDataset
from rtc.production_cli import _load_graph, _load_step1
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step3_knowledge_seeds_v123 import sparse_state_auto_rbc_target_v123


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    out = np.empty_like(values)
    out[order] = np.arange(values.size, dtype=np.float64)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def _summary(true: np.ndarray, sparse: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    true = np.asarray(true, dtype=np.float64)
    sparse = np.asarray(sparse, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    error = sparse - true
    d_true = true - current
    d_sparse = sparse - current
    active = (np.abs(d_true) > 1.0e-7) | (np.abs(d_sparse) > 1.0e-7)
    direction = np.sign(d_true[active]) == np.sign(d_sparse[active]) if np.any(active) else np.asarray([], dtype=bool)
    active_overlap: list[float] = []
    top_overlap: list[float] = []
    for t, s in zip(d_true, d_sparse):
        ta = set(np.flatnonzero(np.abs(t) > 1.0e-7).tolist())
        sa = set(np.flatnonzero(np.abs(s) > 1.0e-7).tolist())
        union = ta | sa
        active_overlap.append(float(len(ta & sa) / len(union)) if union else 1.0)
        k = min(10, t.size)
        ti = set(np.argsort(-np.abs(t), kind="mergesort")[:k].tolist())
        si = set(np.argsort(-np.abs(s), kind="mergesort")[:k].tolist())
        top_overlap.append(float(len(ti & si) / k) if k else 1.0)
    positive = active & (d_true > 1.0e-7)
    negative = active & (d_true < -1.0e-7)
    return {
        "count": int(true.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "true_delta_mean_abs": float(np.mean(np.abs(d_true))),
        "sparse_delta_mean_abs": float(np.mean(np.abs(d_sparse))),
        "direction_accuracy": float(np.mean(direction)) if direction.size else None,
        "opening_direction_accuracy": float(np.mean(np.sign(d_sparse[positive]) == 1.0)) if np.any(positive) else None,
        "closing_direction_accuracy": float(np.mean(np.sign(d_sparse[negative]) == -1.0)) if np.any(negative) else None,
        "active_overlap": float(np.mean(active_overlap)),
        "top10_absolute_action_overlap": float(np.mean(top_overlap)),
        "true_active_fraction": float(np.mean(np.abs(d_true) > 1.0e-7)),
        "sparse_active_fraction": float(np.mean(np.abs(d_sparse) > 1.0e-7)),
        **_corr(true, sparse),
    }


def _type_ids(graph) -> np.ndarray:
    names = tuple(graph.actuator_physics_feature_names)
    cols = {name: i for i, name in enumerate(names)}
    physics = np.asarray(graph.actuator_physics)
    result = np.full(len(graph.actuator_ids), -1, dtype=np.int64)
    for i, name in enumerate(("is_pump", "is_orifice", "is_weir", "is_outlet")):
        col = cols.get(name)
        if col is not None:
            result[physics[:, col] > 0.5] = i
    if np.any(result < 0):
        raise ValueError("actuator type indicators do not cover all actuators")
    return result


def _phase(graph, state: np.ndarray) -> str:
    names = tuple(graph.static_node_feature_names)
    if "max_depth_m" not in names:
        raise ValueError("parity audit requires graph max_depth_m")
    max_depth = np.maximum(
        np.asarray(graph.static_node_features[:, names.index("max_depth_m")], dtype=np.float64),
        1.0e-6,
    )
    endpoints = np.unique(np.concatenate([graph.actuator_upstream, graph.actuator_downstream]))
    fill = float(np.max(np.maximum(state[endpoints, 0], 0.0) / max_depth[endpoints]))
    return "low" if fill < 0.25 else ("mid" if fill < 0.75 else "high")


def _subset(true, sparse, current, mask):
    return _summary(true[:, mask], sparse[:, mask], current[:, mask]) if np.any(mask) else {"count": 0}


def run(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_graph(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2"))
    fit, holdout = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    if len(fit_d2) != 112 or len([name for name in holdout if name.startswith("D2::")]) != 32:
        raise ValueError("parity audit requires the frozen 112/32 D2 split")

    train_index = pd.read_csv(args.train_index)
    train_index = train_index[
        (train_index["scientific_split"].astype(str) == "development")
        & (train_index["development_fold"].astype(str) == "train")
        & (train_index["strategy"].astype(str) == "no_control")
    ].copy()
    if train_index.empty or bool(train_index.duplicated(["event_id", "rainfall_group"], keep=False).any()):
        raise ValueError("no_control Step1 lineage is missing or duplicated")
    sensors = tuple(
        line.strip()
        for line in Path(args.sensors).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    dataset = CausalStep1TrajectoryDataset(
        train_index,
        graph=graph,
        sensor_nodes=sensors,
        history_steps=13,
        model_step_seconds=300,
        scientific_split="development",
        development_fold="train",
    )
    sample_map: dict[tuple[str, str, int], int] = {}
    for i, ref in enumerate(dataset.samples):
        key = (ref.event_id, ref.rainfall_group, int(ref.end_index * 300))
        if key in sample_map:
            raise ValueError(f"duplicate causal Step1 sample key: {key}")
        sample_map[key] = i

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    step1 = _load_step1(args.step1, device)
    true_targets: list[np.ndarray] = []
    sparse_targets: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    phases: list[str] = []
    endpoint_true: list[np.ndarray] = []
    endpoint_pred: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    endpoints = np.unique(np.concatenate([graph.actuator_upstream, graph.actuator_downstream]))
    missing: list[str] = []
    for name in fit_d2:
        entry = cache.entry(name)
        ref_index = int(entry.reference_index)
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][ref_index]).reshape(-1)[0])
        key = (str(entry.event_id), str(entry.rainfall_group), elapsed)
        sample_index = sample_map.get(key)
        if sample_index is None:
            missing.append(name)
            continue
        sample_ref = dataset.samples[sample_index]
        compact = dataset._load(sample_ref.trajectory_index)
        current_setting = compact["setting"][sample_ref.end_index].astype(np.float32)
        current_state = np.asarray(entry.arrays["initial_state"][ref_index], dtype=np.float32)
        observed, mask, context, _ = dataset[sample_index]
        with torch.no_grad():
            predicted = step1(
                observed[None].to(device),
                mask[None].to(device),
                torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
                torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
                context[None].to(device),
            )[0].detach().cpu().numpy().astype(np.float32)
        kwargs = dict(
            low_fill=0.25,
            high_fill=0.75,
            downstream_congestion_fill=0.90,
            response=0.60,
            max_delta_per_update=0.5,
        )
        true_target = sparse_state_auto_rbc_target_v123(current_state, current_setting, graph, **kwargs)
        sparse_target = sparse_state_auto_rbc_target_v123(predicted, current_setting, graph, **kwargs)
        true_targets.append(true_target)
        sparse_targets.append(sparse_target)
        currents.append(current_setting)
        phase = _phase(graph, current_state)
        phases.append(phase)
        endpoint_true.append(current_state[endpoints, 0])
        endpoint_pred.append(predicted[endpoints, 0])
        records.append({
            "group": name,
            "event_id": str(entry.event_id),
            "rainfall_group": str(entry.rainfall_group),
            "checkpoint_id": str(entry.checkpoint_id),
            "elapsed_seconds": elapsed,
            "step1_sample_index": int(sample_index),
            "phase": phase,
        })
    if missing:
        raise ValueError(f"missing causal Step1 samples: {missing[:3]} ({len(missing)} total)")

    true_targets = np.stack(true_targets)
    sparse_targets = np.stack(sparse_targets)
    currents = np.stack(currents)
    endpoint_true = np.stack(endpoint_true)
    endpoint_pred = np.stack(endpoint_pred)
    type_ids = _type_ids(graph)
    type_names = ("pump", "orifice", "weir", "outlet")
    phase_array = np.asarray(phases)
    actuator_metrics = {
        "overall": _summary(true_targets, sparse_targets, currents),
        "by_type": {
            name: _subset(true_targets, sparse_targets, currents, type_ids == i)
            for i, name in enumerate(type_names)
        },
        "by_phase": {
            phase: (
                _summary(
                    true_targets[phase_array == phase],
                    sparse_targets[phase_array == phase],
                    currents[phase_array == phase],
                )
                if np.any(phase_array == phase)
                else {"count": 0}
            )
            for phase in ("low", "mid", "high")
        },
    }
    depth_error = (endpoint_pred - endpoint_true).reshape(-1)
    depth_denom = float(np.sum((endpoint_true.reshape(-1) - np.mean(endpoint_true)) ** 2))
    endpoint_depth = {
        "node_count": int(endpoint_true.shape[1]),
        "sample_count": int(endpoint_true.size),
        "rmse_m": float(np.sqrt(np.mean(depth_error * depth_error))),
        "bias_m": float(np.mean(depth_error)),
        "nse": float(1.0 - np.sum(depth_error * depth_error) / depth_denom) if depth_denom > 1.0e-12 else None,
        "actuator_adjacent_node_ids": [str(graph.node_ids[i]) for i in endpoints],
    }
    payload: dict[str, Any] = {
        "contract": "PROJECT7_STEP3_V124_SPARSE_RBC_PARITY_V1",
        "boundary": {
            "new_swmm": False,
            "step1_retrained": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
        "lineage": {
            "graph": str(Path(args.graph).resolve()),
            "graph_sha256": _sha(args.graph),
            "cache_manifest": str(Path(args.cache_manifest).resolve()),
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "step1_checkpoint": str(Path(args.step1).resolve()),
            "step1_sha256": _sha(args.step1),
            "sensor_layout": str(Path(args.sensors).resolve()),
            "sensor_layout_sha256": _sha(args.sensors),
            "train_index": str(Path(args.train_index).resolve()),
            "train_index_sha256": _sha(args.train_index),
            "step1_source_role": "development/train/no_control_only",
        },
        "split": {
            "fit_d2_groups": 112,
            "holdout_d2_groups": 32,
            "fit_events": sorted({r["event_id"] for r in records}),
            "holdout_outcome_accessed": False,
        },
        "rbc_contract": {
            "low_fill": 0.25,
            "high_fill": 0.75,
            "downstream_congestion": 0.90,
            "response": 0.60,
            "max_delta_per_update": 0.5,
            "state_source_true": "authoritative_cache_initial_state",
            "state_source_sparse": "frozen_step1_reconstruction",
            "settings_source": "no_control_current_setting_at_checkpoint",
        },
        "dataset": {
            "matched_groups": len(records),
            "missing_groups": missing,
            "sensor_count": len(sensors),
            "history_frames": 13,
            "model_step_seconds": 300,
            "actuator_count": len(graph.actuator_ids),
            "node_count": len(graph.node_ids),
        },
        "actuator_target_parity": actuator_metrics,
        "endpoint_depth_reconstruction": endpoint_depth,
        "records": records,
        "interpretation": {
            "sparse_rbc_uses_true_state": False,
            "true_rbc_offline_only": True,
            "endpoint_state_is_primary_diagnostic": True,
        },
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STEP3_V124_SPARSE_RBC_PARITY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overall = actuator_metrics["overall"]
    md = [
        "# STEP3 V124 Sparse-RBC parity",
        "",
        "TrainFit D2 only; TRUE-RBC uses authoritative cache state offline, SPARSE-RBC uses frozen Step1 reconstruction.",
        "",
        f"Matched groups: {len(records)}/112",
        f"Endpoint depth RMSE: {endpoint_depth['rmse_m']:.6g} m",
        f"Endpoint depth NSE: {endpoint_depth['nse']}",
        f"Target MAE: {overall['mae']:.6g}",
        f"Target RMSE: {overall['rmse']:.6g}",
        f"Direction accuracy: {overall['direction_accuracy']}",
        f"Active overlap: {overall['active_overlap']:.6g}",
        f"Top-10 action overlap: {overall['top10_absolute_action_overlap']:.6g}",
        "",
        "No SWMM was run; InternalHoldout outcomes were not read.",
        "",
    ]
    (out / "STEP3_V124_SPARSE_RBC_PARITY.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit true versus sparse-state RBC parity")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--step1", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

