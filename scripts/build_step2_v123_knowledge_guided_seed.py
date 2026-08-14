"""Build verified sparse-RBC knowledge first moves from causal Step1 state.

This builder is development/TrainFit only.  It deliberately uses the same
`development/train/no_control` Step1 lineage as the V124 Sparse-RBC parity audit and
stores the supervisory first move relative to the physical current setting.
"""
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
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123


def _lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        x.strip()
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    )


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _train_no_control_index(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scientific_split", "development_fold", "strategy", "event_id", "rainfall_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"knowledge seed train index missing {missing}")
    frame = frame[
        (frame["scientific_split"].astype(str) == "development")
        & (frame["development_fold"].astype(str) == "train")
        & (frame["strategy"].astype(str) == "no_control")
    ].copy()
    if frame.empty:
        raise ValueError("knowledge seed requires development/train/no_control lineage")
    if bool(frame.duplicated(["event_id", "rainfall_group"], keep=False).any()):
        raise ValueError("knowledge seed no_control Step1 lineage is duplicated")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 verified sparse-RBC seed builder")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    # Kept for command-line compatibility/provenance. Sparse-RBC itself does not consume
    # a rainfall forecast; the current hydraulic state already defines the feedback move.
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--step1", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    graph = load_graph_v120(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2"))
    fit, holdout = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if len(fit_d2) != 112 or len(holdout_d2) != 32:
        raise ValueError("knowledge seed requires the frozen 112/32 D2 split")

    train_index = _train_no_control_index(args.train_index)
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
        key = (str(ref.event_id), str(ref.rainfall_group), int(ref.end_index * 300))
        if key in sample_map:
            raise ValueError(f"duplicate causal no-control Step1 sample key: {key}")
        sample_map[key] = i

    target_device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = _load_step1(args.step1, target_device)
    records: list[dict[str, object]] = []
    first_moves: list[np.ndarray] = []
    current_settings: list[np.ndarray] = []
    first_move_targets: list[np.ndarray] = []
    unmatched: list[str] = []

    for name in fit_d2:
        entry = cache.entry(name)
        ref_index = int(entry.reference_index)
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][ref_index]).reshape(-1)[0])
        key = (str(entry.event_id), str(entry.rainfall_group), elapsed)
        sample_index = sample_map.get(key)
        if sample_index is None:
            unmatched.append(name)
            continue

        sample_ref = dataset.samples[sample_index]
        compact = dataset._load(sample_ref.trajectory_index)
        current_setting = np.asarray(compact["setting"][sample_ref.end_index], dtype=np.float32)
        observed, mask, context, _ = dataset[sample_index]
        with torch.no_grad():
            state = model(
                observed[None].to(target_device),
                mask[None].to(target_device),
                torch.as_tensor(
                    graph.static_node_features,
                    dtype=torch.float32,
                    device=target_device,
                ),
                torch.as_tensor(graph.edge_index, dtype=torch.long, device=target_device),
                context[None].to(target_device),
            )[0].detach().cpu().numpy().astype(np.float32)

        reference = np.asarray(entry.arrays["settings"][ref_index], dtype=np.float32)
        anchor = build_sparse_state_auto_rbc_anchor_v123(
            state,
            current_setting,
            reference,
            graph,
            control_block_steps=2,
            max_delta_per_update=0.5,
        )
        first_target = np.asarray(anchor[0], dtype=np.float32)
        delta = first_target - current_setting
        if np.max(np.abs(delta), initial=0.0) > 0.500001:
            raise RuntimeError(f"Sparse-RBC seed violates first-move continuity for {name}")

        first_moves.append(delta.astype(np.float32))
        current_settings.append(current_setting.astype(np.float32))
        first_move_targets.append(first_target.astype(np.float32))
        records.append(
            {
                "group": name,
                "event": str(entry.event_id),
                "rainfall_group": str(entry.rainfall_group),
                "checkpoint_id": str(entry.checkpoint_id),
                "checkpoint_elapsed_seconds": elapsed,
                "step1_sample_index": int(sample_index),
                "positive_first_move_count": int(np.sum(delta > 1e-7)),
                "negative_first_move_count": int(np.sum(delta < -1e-7)),
                "zero_first_move_count": int(np.sum(np.abs(delta) <= 1e-7)),
                "max_abs_first_move": float(np.max(np.abs(delta), initial=0.0)),
                "first_move_l1": float(np.sum(np.abs(delta))),
            }
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if first_moves:
        np.savez_compressed(
            out / "KNOWLEDGE_GUIDED_SEED_V123.npz",
            group_names=np.asarray([r["group"] for r in records]),
            first_move_delta=np.stack(first_moves),
            current_setting=np.stack(current_settings),
            first_move_target=np.stack(first_move_targets),
            actuator_ids=np.asarray(graph.actuator_ids),
        )

    payload = {
        "contract": "PROJECT7_KNOWLEDGE_GUIDED_SEED_V123_V2_SPARSE_RBC",
        "seed_role": "verified sparse-state Auto-RBC anchor for D4 support design and MPC warm start",
        "source_inputs": {
            "step1_reconstructed_state": True,
            "development_train_no_control_setting": True,
            "causal_rainfall_consumed_by_anchor": False,
            "topology": True,
            "future_realized_rainfall": False,
            "future_swmm_state": False,
            "future_target_flow": False,
        },
        "lineage": {
            "step1_checkpoint": str(Path(args.step1).resolve()),
            "step1_sha256": _sha(args.step1),
            "sensor_layout": str(Path(args.sensors).resolve()),
            "sensor_layout_sha256": _sha(args.sensors),
            "train_index": str(Path(args.train_index).resolve()),
            "train_index_sha256": _sha(args.train_index),
            "cache_manifest": str(Path(args.cache_manifest).resolve()),
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "causal_store_available_not_consumed": str(Path(args.causal_store).resolve()),
            "causal_store_sha256": _sha(args.causal_store),
            "step1_source_role": "development/train/no_control_only",
        },
        "fit_d2_groups": len(fit_d2),
        "holdout_d2_groups_not_used": len(holdout_d2),
        "step1_history_available_groups": len(records),
        "unmatched_groups": unmatched,
        "history_frames": 13,
        "model_step_seconds": 300,
        "records": records,
        "aggregate": {
            "positive_first_move_fraction": (
                float(np.mean([r["positive_first_move_count"] for r in records]) / len(graph.actuator_ids))
                if records
                else 0.0
            ),
            "negative_first_move_fraction": (
                float(np.mean([r["negative_first_move_count"] for r in records]) / len(graph.actuator_ids))
                if records
                else 0.0
            ),
            "mean_l1": float(np.mean([r["first_move_l1"] for r in records])) if records else 0.0,
            "max_abs": float(np.max([r["max_abs_first_move"] for r in records])) if records else 0.0,
        },
        "boundary": {
            "new_swmm": False,
            "holdout_outcomes_accessed": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    (out / "STEP2_V123_KNOWLEDGE_GUIDED_SEED.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "STEP2_V123_KNOWLEDGE_GUIDED_SEED.md").write_text(
        "\n".join(
            [
                "# V123 verified Sparse-RBC knowledge seed",
                "",
                f"Step1-reconstructed matched groups: {len(records)}/{len(fit_d2)}",
                f"unmatched causal-history groups: {len(unmatched)}",
                f"mean first-move L1: {payload['aggregate']['mean_l1']:.4f}",
                f"max absolute move: {payload['aggregate']['max_abs']:.4f}",
                "",
                "The primary seed now uses the same development/train/no_control lineage and sparse-state RBC mechanism as the verified V124 parity audit.",
                "The causal rainfall store is retained only as frozen provenance and is not consumed by the RBC anchor.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
