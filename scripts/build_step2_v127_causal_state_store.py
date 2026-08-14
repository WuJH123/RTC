"""Build the V127 causal Step1-state store; no SWMM and no future truth are used.

The store covers every canonical D2/targeted-D3/D4 event/checkpoint needed by V127 using
only development/train no-control causal sensor histories and the frozen Step1 model.
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
from rtc.production_cli import _load_graph, _load_step1
from rtc.step2_state_store_v127 import V127_CAUSAL_STATE_CONTRACT
from rtc.step2_train_response_v60 import V60TrainCache


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _train_no_control(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scientific_split", "development_fold", "strategy", "event_id", "rainfall_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V127 train index lacks {missing}")
    frame = frame[
        (frame["scientific_split"].astype(str) == "development")
        & (frame["development_fold"].astype(str) == "train")
        & (frame["strategy"].astype(str) == "no_control")
    ].copy()
    if frame.empty:
        raise ValueError("V127 causal state store requires development/train/no_control trajectories")
    if frame.duplicated(["event_id", "rainfall_group"], keep=False).any():
        raise ValueError("V127 no-control Step1 trajectory lineage is duplicated")
    return frame


def _needed_entries(paths: list[str]) -> list[object]:
    entries: dict[tuple[str, str], object] = {}
    for path in paths:
        cache = V60TrainCache(path)
        for name in cache.names():
            entry = cache.entry(name)
            key = (str(entry.event_id), str(entry.checkpoint_id))
            previous = entries.get(key)
            if previous is not None and str(previous.rainfall_group) != str(entry.rainfall_group):
                raise ValueError(f"V127 event/checkpoint rainfall identity drift: {key}")
            entries[key] = entry
    if not entries:
        raise ValueError("V127 causal state store has no requested cache entries")
    return list(entries.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--train-index", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache")
    p.add_argument("--d4-audit-cache")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    graph = _load_graph(args.graph)
    sensors = tuple(
        line.strip() for line in Path(args.sensors).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not sensors or not set(sensors).issubset(graph.node_ids):
        raise ValueError("V127 sensor layout is empty or incompatible with graph")
    frame = _train_no_control(args.train_index)
    dataset = CausalStep1TrajectoryDataset(
        frame,
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
            raise ValueError(f"V127 duplicate Step1 causal sample: {key}")
        sample_map[key] = i

    paths = [args.cache_manifest]
    if args.d4_fit_cache:
        paths.append(args.d4_fit_cache)
    if args.d4_audit_cache:
        paths.append(args.d4_audit_cache)
    entries = _needed_entries(paths)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    step1 = _load_step1(args.step1, device)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)

    records: list[tuple[str, str, int, np.ndarray, np.ndarray, str]] = []
    for entry in sorted(entries, key=lambda e: (str(e.rainfall_group), str(e.event_id), str(e.checkpoint_id))):
        ref_index = int(entry.reference_index)
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][ref_index]).reshape(-1)[0])
        sample_index = sample_map.get((str(entry.event_id), str(entry.rainfall_group), elapsed))
        if sample_index is None:
            raise ValueError(
                f"V127 has no causal Step1 history for {entry.event_id}/{entry.checkpoint_id}/{elapsed}"
            )
        sample_ref = dataset.samples[sample_index]
        compact = dataset._load(sample_ref.trajectory_index)
        current = np.asarray(compact["setting"][sample_ref.end_index], dtype=np.float32)
        observed, mask, context, _ = dataset[sample_index]
        with torch.no_grad():
            estimate = step1(
                observed[None].to(device),
                mask[None].to(device),
                static,
                edges,
                context[None].to(device),
            )[0].detach().cpu().numpy().astype(np.float32)
        if estimate.shape != np.asarray(entry.arrays["initial_state"][ref_index]).shape:
            raise ValueError("V127 Step1 estimate/state-label schema mismatch")
        if current.shape != (len(graph.actuator_ids),):
            raise ValueError("V127 current-setting actuator count mismatch")
        records.append(
            (
                str(entry.event_id),
                str(entry.checkpoint_id),
                elapsed,
                estimate,
                current,
                _state_sha(estimate),
            )
        )

    event_ids, checkpoint_ids, elapsed, states, settings, state_sha = zip(*records, strict=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        contract=np.asarray(V127_CAUSAL_STATE_CONTRACT),
        event_ids=np.asarray(event_ids),
        checkpoint_ids=np.asarray(checkpoint_ids),
        elapsed_seconds=np.asarray(elapsed, dtype=np.int64),
        state_si=np.stack(states).astype(np.float32),
        current_setting=np.stack(settings).astype(np.float32),
        state_sha256=np.asarray(state_sha),
        step1_sha256=np.asarray(_sha(args.step1)),
        sensor_sha256=np.asarray(_sha(args.sensors)),
        graph_sha256=np.asarray(_sha(args.graph)),
    )
    report = {
        "contract": V127_CAUSAL_STATE_CONTRACT,
        "rows": len(records),
        "unique_events": len(set(event_ids)),
        "unique_checkpoints": len(set(zip(event_ids, checkpoint_ids, strict=True))),
        "state_shape": list(np.stack(states).shape),
        "actuator_count": len(graph.actuator_ids),
        "step1_sha256": _sha(args.step1),
        "sensor_sha256": _sha(args.sensors),
        "graph_sha256": _sha(args.graph),
        "future_swmm_truth_used_as_input": False,
        "new_swmm": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
