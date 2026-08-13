"""Audit whether existing Train-only assets can supply causal pre-action history.

This is a read-only evidence tool for the V9 state-sufficiency gate.  It does
not run SWMM and deliberately uses only the frozen development/train
checkpoint table, the Train no-control compact trajectories, and the V60
cache's representative initial states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FRAMES = 13
FRAME_SECONDS = 300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _train_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("scientific_split") == "development"
        and row.get("development_fold") == "train"
    ]


def _representative_cache_rows(cache_root: Path, manifest: dict[str, Any]) -> dict[str, tuple[str, np.ndarray]]:
    representatives: dict[str, tuple[str, np.ndarray]] = {}
    for shard in manifest["shards"]:
        directory = cache_root / shard["directory"]
        checkpoint_ids = np.load(directory / "checkpoint_id.npy", mmap_mode="r")
        event_ids = np.load(directory / "event_id.npy", mmap_mode="r")
        initial_states = np.load(directory / "initial_state.npy", mmap_mode="r")
        for index, checkpoint_value in enumerate(checkpoint_ids):
            checkpoint_id = str(checkpoint_value)
            if checkpoint_id not in representatives:
                representatives[checkpoint_id] = (
                    str(event_ids[index]),
                    np.array(initial_states[index], copy=True),
                )
    return representatives


def audit_history_availability(
    study_root: Path,
    cache_manifest_path: Path,
    checkpoints_path: Path,
    train_index_path: Path,
) -> dict[str, Any]:
    checkpoint_rows = _train_rows(_read_csv(checkpoints_path))
    checkpoint_by_id = {row["checkpoint_id"]: row for row in checkpoint_rows}
    no_control_rows = [
        row
        for row in _train_rows(_read_csv(train_index_path))
        if row.get("strategy") == "no_control"
    ]
    baseline_by_event = {
        row["event_id"]: Path(row["compact_path"]) for row in no_control_rows
    }
    metadata_by_event = {
        row["event_id"]: Path(row["metadata_path"]) for row in no_control_rows
    }
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_root = cache_manifest_path.parent
    representatives = _representative_cache_rows(cache_root, cache_manifest)

    failures: list[dict[str, Any]] = []
    comparisons: list[float] = []
    source_contracts: set[str] = set()
    state_channels: list[str] | None = None
    compact_keys: set[str] = set()
    for checkpoint_id, checkpoint in checkpoint_by_id.items():
        event_id = checkpoint["event_id"]
        compact_path = baseline_by_event.get(event_id)
        representative = representatives.get(checkpoint_id)
        if compact_path is None or not compact_path.exists():
            failures.append({"checkpoint_id": checkpoint_id, "reason": "missing_train_no_control_compact"})
            continue
        if representative is None:
            failures.append({"checkpoint_id": checkpoint_id, "reason": "missing_cache_representative"})
            continue
        try:
            compact = np.load(compact_path, allow_pickle=False)
            compact_keys.update(compact.files)
            if state_channels is None:
                state_channels = [str(value) for value in compact["state_channels"]]
            elapsed = np.asarray(compact["elapsed_seconds"], dtype=np.int64)
            checkpoint_seconds = int(checkpoint["checkpoint_elapsed_seconds"])
            expected = np.arange(
                checkpoint_seconds - (REQUIRED_FRAMES - 1) * FRAME_SECONDS,
                checkpoint_seconds + FRAME_SECONDS,
                FRAME_SECONDS,
                dtype=np.int64,
            )
            available = np.isin(expected, elapsed)
            current_positions = np.where(elapsed == checkpoint_seconds)[0]
            if not bool(np.all(available)) or len(current_positions) != 1:
                failures.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "reason": "exact_13_frame_prefix_unavailable",
                        "expected_elapsed_seconds": expected.tolist(),
                        "trajectory_elapsed_range": [int(elapsed[0]), int(elapsed[-1])],
                    }
                )
                continue
            current_index = int(current_positions[0])
            history = compact["state_si"][current_index - REQUIRED_FRAMES + 1 : current_index + 1]
            if history.shape[0] != REQUIRED_FRAMES:
                failures.append({"checkpoint_id": checkpoint_id, "reason": "history_frame_count_mismatch"})
                continue
            _, cache_initial = representative
            max_difference = float(np.max(np.abs(history[-1] - cache_initial)))
            comparisons.append(max_difference)
            metadata_path = metadata_by_event.get(event_id)
            if metadata_path and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                source_contracts.add(str(metadata.get("data_contract", "")))
        except Exception as exc:  # pragma: no cover - evidence should fail closed
            failures.append({"checkpoint_id": checkpoint_id, "reason": "audit_error", "error": repr(exc)})

    link_flow_fields = sorted(
        key
        for key in compact_keys
        if any(token in key.lower() for token in ("link_flow", "conduit_flow", "link_q", "conduit_q"))
    )
    actuator_flow_fields = sorted(
        key for key in compact_keys if "actuator_flow" in key.lower()
    )
    return {
        "contract": "PROJECT7_STEP2_V90_HISTORY_AVAILABILITY_AUDIT_V1",
        "scope": {
            "scientific_split": "development",
            "development_fold": "train",
            "validation_accessed": False,
            "final_accessed": False,
            "formal_run_accessed": False,
            "swmm_run": False,
        },
        "required_history": {
            "frames": REQUIRED_FRAMES,
            "frame_interval_seconds": FRAME_SECONDS,
            "window_seconds": (REQUIRED_FRAMES - 1) * FRAME_SECONDS,
            "definition": "pre-action causal frames [t-3600,...,t] inclusive",
        },
        "checkpoint_coverage": {
            "train_events": len({row["event_id"] for row in checkpoint_rows}),
            "train_checkpoints": len(checkpoint_rows),
            "no_control_train_events": len(no_control_rows),
            "recoverable_checkpoints": len(comparisons),
            "failed_checkpoints": len(failures),
            "all_required_checkpoints_recoverable": len(comparisons) == len(checkpoint_rows) and not failures,
            "current_state_max_abs_difference_m3_or_channel_units": max(comparisons) if comparisons else None,
            "current_state_median_abs_difference": float(np.median(comparisons)) if comparisons else None,
            "source_data_contracts": sorted(source_contracts),
        },
        "history_sources": {
            "primary": "Train no_control compact trajectories referenced by step1_index/train_run_index.csv",
            "checkpoint_source": str(checkpoints_path),
            "trajectory_contract": sorted(source_contracts),
            "trajectory_sampling_seconds": FRAME_SECONDS,
            "pre_action_only": True,
            "future_leakage": "NONE",
            "failure_examples": failures[:10],
        },
        "flow_availability": {
            "current_all_link_flow_available": bool(link_flow_fields),
            "link_flow_fields": link_flow_fields,
            "actuator_flow_available": bool(actuator_flow_fields),
            "actuator_flow_fields": actuator_flow_fields,
            "actuator_flow_is_not_all_link_flow": True,
            "compact_state_channels": state_channels or [],
            "diagnosis": (
                "Existing compact assets expose node state and 109-actuator flow only; "
                "no current all-link/conduit flow field is present."
            ),
        },
        "lineage": {
            "cache_manifest": str(cache_manifest_path),
            "cache_manifest_sha256": _sha256(cache_manifest_path),
            "checkpoint_table": str(checkpoints_path),
            "checkpoint_table_sha256": _sha256(checkpoints_path),
            "train_index": str(train_index_path),
            "train_index_sha256": _sha256(train_index_path),
            "graph_or_swmm": "not read; no SWMM executed",
        },
        "decision_input": {
            "history_available_for_all_train_checkpoints": len(comparisons) == len(checkpoint_rows) and not failures,
            "all_link_flow_available_online": bool(link_flow_fields),
            "recommended_next_step": (
                "Use the recovered causal 13-frame history for a diagnostic/history-enhanced cache; "
                "do not generate SWMM or claim all-link flow availability."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_history_availability(
        args.study_root,
        args.cache_manifest,
        args.checkpoints,
        args.train_index,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["checkpoint_coverage"], indent=2, sort_keys=True))
    print(json.dumps(report["flow_availability"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
