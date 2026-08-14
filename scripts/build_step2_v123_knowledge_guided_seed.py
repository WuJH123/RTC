"""Build causal Step1-state knowledge-guided warm starts for V12.3 diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.lazy_step1 import CausalStep1TrajectoryDataset
from rtc.production_cli import _load_step1
from rtc.step2_causal_rainfall_v123 import load_causal_forecast_store_v123
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step3_knowledge_seeds_v123 import build_knowledge_guided_seed_settings_v123


def _lines(path: str | Path) -> tuple[str, ...]:
    return tuple(x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 causal knowledge-guided seed builder")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--step1", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    graph = load_graph_v120(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    store = load_causal_forecast_store_v123(args.causal_store)
    names = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, _ = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [n for n in fit if n.startswith("D2::")]
    train_index = pd.read_csv(args.train_index)
    sensors = _lines(args.sensors)
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
        trajectory = dataset.trajectories[ref.trajectory_index]
        sample_map[(ref.event_id, ref.rainfall_group, int(ref.end_index * 300))] = i
    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_step1(args.step1, target)
    store_index = store.index()
    records = []
    first_moves = []
    unmatched = []
    for name in fit_d2:
        entry = cache.entry(name)
        elapsed = int(entry.arrays["elapsed_seconds"][entry.reference_index, 0])
        key = (str(entry.event_id), str(entry.rainfall_group), elapsed)
        sample_index = sample_map.get(key)
        if sample_index is None:
            unmatched.append(name)
            continue
        observed, mask, context, _ = dataset[sample_index]
        with torch.no_grad():
            state = model(
                observed[None].to(target),
                mask[None].to(target),
                torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=target),
                torch.as_tensor(graph.edge_index, dtype=torch.long, device=target),
                context[None].to(target),
            )[0].detach().cpu().numpy()
        forecast = store.forecast_mmhr[store_index[name]]
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        seed = build_knowledge_guided_seed_settings_v123(state, forecast, reference, graph)
        delta = seed[:2].mean(axis=0) - reference[:2].mean(axis=0)
        first_moves.append(delta.astype(np.float32))
        records.append({
            "group": name,
            "event": str(entry.event_id),
            "rainfall_group": str(entry.rainfall_group),
            "checkpoint_elapsed_seconds": elapsed,
            "step1_sample_index": int(sample_index),
            "positive_first_move_count": int(np.sum(delta > 1e-7)),
            "negative_first_move_count": int(np.sum(delta < -1e-7)),
            "zero_first_move_count": int(np.sum(np.abs(delta) <= 1e-7)),
            "max_abs_first_move": float(np.max(np.abs(delta))),
            "first_move_l1": float(np.sum(np.abs(delta))),
        })
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if first_moves:
        np.savez_compressed(out / "KNOWLEDGE_GUIDED_SEED_V123.npz", group_names=np.asarray([r["group"] for r in records]), first_move_delta=np.stack(first_moves), actuator_ids=np.asarray(graph.actuator_ids))
    payload = {
        "contract": "PROJECT7_KNOWLEDGE_GUIDED_SEED_V123_V1",
        "seed_role": "MPC warm start only; data-driven candidates remain admissible",
        "source_inputs": {"step1_reconstructed_state": True, "causal_rainfall": True, "topology": True, "future_realized_rainfall": False, "future_swmm_state": False, "future_target_flow": False},
        "step1_checkpoint": str(Path(args.step1).resolve()),
        "step1_sha256": hashlib.sha256(Path(args.step1).read_bytes()).hexdigest(),
        "sensor_layout": str(Path(args.sensors).resolve()),
        "sensor_layout_sha256": hashlib.sha256(Path(args.sensors).read_bytes()).hexdigest(),
        "causal_store": str(Path(args.causal_store).resolve()),
        "causal_store_sha256": hashlib.sha256(Path(args.causal_store).read_bytes()).hexdigest(),
        "fit_d2_groups": len(fit_d2),
        "step1_history_available_groups": len(records),
        "unmatched_groups": unmatched,
        "history_frames": 13,
        "model_step_seconds": 300,
        "records": records,
        "aggregate": {
            "positive_first_move_fraction": float(np.mean([r["positive_first_move_count"] for r in records]) / len(graph.actuator_ids)) if records else 0.0,
            "negative_first_move_fraction": float(np.mean([r["negative_first_move_count"] for r in records]) / len(graph.actuator_ids)) if records else 0.0,
            "mean_l1": float(np.mean([r["first_move_l1"] for r in records])) if records else 0.0,
            "max_abs": float(np.max([r["max_abs_first_move"] for r in records])) if records else 0.0,
        },
        "boundary": {"new_swmm": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False},
    }
    (out / "STEP2_V123_KNOWLEDGE_GUIDED_SEED.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "STEP2_V123_KNOWLEDGE_GUIDED_SEED.md").write_text("\n".join([
        "# V123 Knowledge-guided seed", "", f"Step1-reconstructed matched groups: {len(records)}/{len(fit_d2)}", f"unmatched (insufficient causal history): {len(unmatched)}", f"mean first-move L1: {payload['aggregate']['mean_l1']:.4f}", f"max absolute move: {payload['aggregate']['max_abs']:.4f}", "", "The seed uses only frozen Step1 reconstructed state, causal forecast and actuator topology; it is a warm start, not a candidate ceiling.", ""
    ]), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
