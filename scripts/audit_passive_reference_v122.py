"""Audit frozen D3 HOLD commands against the no-control target latch.

This is read-only.  It does not run SWMM and deliberately distinguishes the
commanded target sequence from realised ``current_setting`` telemetry.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rtc.step2_passive_reference_v122 import assert_passive_command_sequence_v122
from rtc.swmm_data import sha256_file


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metadata_compact(path: Path) -> tuple[dict[str, object], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compact = path.parent / str(payload["compact_file"])
    if not compact.is_file():
        raise FileNotFoundError(compact)
    return payload, compact


def audit(*, manifest: Path, run_index: Path) -> dict[str, object]:
    manifest_rows = {
        row["checkpoint_id"]: row for row in _rows(manifest)
        if row.get("data_role") == "D3_HOLD_REFERENCE"
    }
    index_rows = [
        row for row in _rows(run_index)
        if row.get("data_role") == "D3_HOLD_REFERENCE"
    ]
    if len(manifest_rows) != 144 or len(index_rows) != 144:
        raise ValueError("V122 passive audit requires exactly 144 D3 HOLD groups")

    records: list[dict[str, object]] = []
    max_target_error = 0.0
    max_command_constancy_error = 0.0
    max_d3_start_target_error = 0.0
    max_d3_start_current_error = 0.0
    command_mismatch_count = 0
    current_lag_group_count = 0
    for row in index_rows:
        checkpoint_id = row["checkpoint_id"]
        parent_row = manifest_rows[checkpoint_id]
        d3_meta_path = Path(row["metadata_path"])
        d3_meta, d3_compact = _metadata_compact(d3_meta_path)
        parent_meta_path = Path(parent_row["trajectory_metadata_path"])
        parent_meta, parent_compact = _metadata_compact(parent_meta_path)
        d3 = np.load(d3_compact, allow_pickle=False)
        parent = np.load(parent_compact, allow_pickle=False)
        elapsed = int(parent_row["checkpoint_elapsed_seconds"])
        parent_times = np.asarray(parent["elapsed_seconds"], dtype=np.int64)
        where = np.flatnonzero(parent_times == elapsed)
        if where.size != 1:
            raise ValueError(f"{checkpoint_id}: parent checkpoint time is not unique")
        checkpoint_index = int(where[0])
        active_target = np.asarray(parent["target_setting"][checkpoint_index], dtype=np.float64)
        commanded = np.asarray(d3["commanded_setting"], dtype=np.float64)
        target = np.asarray(d3["target_setting"], dtype=np.float64)
        current = np.asarray(d3["current_setting"], dtype=np.float64)
        assert_passive_command_sequence_v122(commanded, active_target)
        target_error = float(np.max(np.abs(commanded - active_target)))
        const_error = float(np.max(np.abs(commanded - commanded[0])))
        start_target_error = float(np.max(np.abs(target[0] - active_target)))
        start_current_error = float(np.max(np.abs(current[0] - active_target)))
        max_target_error = max(max_target_error, target_error)
        max_command_constancy_error = max(max_command_constancy_error, const_error)
        max_d3_start_target_error = max(max_d3_start_target_error, start_target_error)
        max_d3_start_current_error = max(max_d3_start_current_error, start_current_error)
        command_mismatch_count += int(target_error > 1.0e-9 or const_error > 1.0e-9)
        current_lag_group_count += int(start_current_error > 1.0e-9)
        records.append({
            "checkpoint_id": checkpoint_id,
            "event_id": row["event_id"],
            "checkpoint_elapsed_seconds": elapsed,
            "d3_metadata_path": str(d3_meta_path.resolve()),
            "parent_metadata_path": str(parent_meta_path.resolve()),
            "d3_metadata_sha256": sha256_file(d3_meta_path),
            "parent_metadata_sha256": sha256_file(parent_meta_path),
            "command_constancy_error_max": const_error,
            "command_target_error_max": target_error,
            "d3_start_target_error_max": start_target_error,
            "d3_start_current_lag_max": start_current_error,
            "d3_data_role": d3_meta.get("data_role"),
            "parent_data_role": parent_meta.get("data_role"),
        })

    return {
        "contract": "PROJECT7_V122_PASSIVE_REFERENCE_SEMANTICS_AUDIT_V1",
        "swmm_run": False,
        "training_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "run_index": str(run_index.resolve()),
        "run_index_sha256": sha256_file(run_index),
        "hold_group_count": len(records),
        "commanded_setting_is_active_target": command_mismatch_count == 0,
        "command_mismatch_group_count": command_mismatch_count,
        "max_command_target_error": max_target_error,
        "max_command_constancy_error": max_command_constancy_error,
        "max_d3_start_target_error": max_d3_start_target_error,
        "max_d3_start_current_lag": max_d3_start_current_error,
        "groups_with_realised_current_lag": current_lag_group_count,
        "passive_reference_verified": command_mismatch_count == 0,
        "current_setting_is_telemetry_not_command": True,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    payload = audit(
        manifest=Path(args.manifest).resolve(),
        run_index=Path(args.run_index).resolve(),
    )
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "STEP2_V122_PASSIVE_REFERENCE_AUDIT.json"
    md_path = out / "STEP2_V122_PASSIVE_REFERENCE_AUDIT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "# V122 passive reference audit\n\n"
        f"- HOLD groups: {payload['hold_group_count']}\n"
        f"- commanded sequence equals active target: {payload['passive_reference_verified']}\n"
        f"- max command-target error: {payload['max_command_target_error']:.3e}\n"
        f"- groups with realised-current lag: {payload['groups_with_realised_current_lag']}\n"
        "\n`current_setting` is retained as physical telemetry; it is not used as the passive command anchor.\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), **{
        k: payload[k] for k in (
            "hold_group_count", "passive_reference_verified",
            "max_command_target_error", "max_command_constancy_error",
            "groups_with_realised_current_lag",
        )
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
