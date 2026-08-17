"""Audit the current Direct-TFV authoritative Development closed loop.

The audit separates execution correctness from scientific benefit.  Execution must prove the exact
scored first move was written/read back on the 600-s grid without post-score projection, every MPC
decision screened all 109 facilities, and every accepted action remained inside TrainFit support.
If a baseline node-statistics file is supplied, authoritative TFV difference is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


DIRECT_TFV_CLOSED_LOOP_AUDIT_CONTRACT = "PROJECT7_DIRECT_TFV_AUTHORITATIVE_CLOSED_LOOP_AUDIT_V1"


def _tfv(path: str | Path) -> float:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return float(sum(float(row["delta_flooding_volume_m3"]) for row in csv.DictReader(handle)))


def _load(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Direct-TFV closed-loop metadata must be an object")
    decision_name = meta.get("decision_file")
    if not decision_name:
        raise ValueError("Direct-TFV closed-loop metadata lacks decision_file")
    rows = [
        json.loads(line)
        for line in (metadata_path.parent / str(decision_name)).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Direct-TFV decision log contains a non-object row")
    return meta, rows


def audit_direct_tfv_closed_loop(
    *,
    metadata_path: str | Path,
    baseline_node_statistics: str | Path | None = None,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path).resolve()
    meta, rows = _load(metadata_path)
    if meta.get("strategy") != "proposed_direct_tfv_selection_aware_trust_region":
        raise ValueError("audit requires the current Direct-TFV proposed strategy")
    if int(meta.get("control_update_seconds", -1)) != 600:
        raise ValueError("Direct-TFV closed-loop control grid is not 600 s")
    if int(meta.get("decisions", -1)) != len(rows):
        raise ValueError("Direct-TFV metadata decision count differs from decision log")

    grid_violations = support_violations = engineering_violations = readback_failures = 0
    mpc_actions = hold_actions = fatal_fallbacks = 0
    runtimes: list[float] = []
    predicted: list[float] = []
    screened_counts: list[int] = []
    active_counts: list[int] = []
    changed_counts: list[int] = []
    support_ratios: list[float] = []
    fatal_prefixes = ("FALLBACK_",)
    first_elapsed = None
    for row in rows:
        elapsed = int(row.get("elapsed_seconds", -1))
        if first_elapsed is None:
            first_elapsed = elapsed
        if first_elapsed is None or elapsed < first_elapsed or (elapsed - first_elapsed) % 600:
            grid_violations += 1
        source = str(row.get("source", ""))
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict):
            engineering_violations += 1
            continue
        if source.startswith(fatal_prefixes):
            fatal_fallbacks += 1
        if diagnostics.get("score_equals_execute") is not True:
            engineering_violations += 1
        target_delta = float(diagnostics.get("command_delta_from_previous_target_max", diagnostics.get("target_change_max", 0.0)))
        if not np.isfinite(target_delta) or target_delta > 0.5 + 1.0e-7:
            engineering_violations += 1
        mismatch = float(diagnostics.get("previous_write_target_readback_mismatch_max", 0.0))
        failed = int(diagnostics.get("target_readback_failed_actuators", 0))
        if mismatch > 1.0e-6 or failed:
            readback_failures += 1
        runtime = diagnostics.get("decision_runtime_seconds")
        if runtime is not None and np.isfinite(float(runtime)):
            runtimes.append(float(runtime))
        if source == "MPC_DIRECT_TFV_TRUST_REGION":
            mpc_actions += 1
            screened = int(diagnostics.get("screened_facility_count", -1))
            active = int(diagnostics.get("active_facility_count", -1))
            changed = int(diagnostics.get("first_move_changed_facility_count", -1))
            ratio = float(diagnostics.get("maximum_support_ratio", np.inf))
            screened_counts.append(screened)
            active_counts.append(active)
            changed_counts.append(changed)
            support_ratios.append(ratio)
            if screened != 109:
                engineering_violations += 1
            if active <= 0 or changed < 0 or changed > active:
                engineering_violations += 1
            if not np.isfinite(ratio) or ratio > 1.0001:
                support_violations += 1
            value = float(diagnostics.get("predicted_delta_tfv_m3", np.nan))
            if not np.isfinite(value) or value >= 0.0:
                engineering_violations += 1
            else:
                predicted.append(value)
        elif source == "HOLD_DIRECT_TFV_NO_CONFIDENT_BENEFIT":
            hold_actions += 1

    statistics_name = meta.get("node_statistics_file")
    proposed_tfv = None
    if statistics_name and (metadata_path.parent / str(statistics_name)).is_file():
        proposed_tfv = _tfv(metadata_path.parent / str(statistics_name))
    baseline_tfv = None if baseline_node_statistics is None else _tfv(baseline_node_statistics)
    delta_vs_baseline = (
        None if proposed_tfv is None or baseline_tfv is None else float(proposed_tfv - baseline_tfv)
    )
    execution_passed = bool(
        rows
        and grid_violations == 0
        and support_violations == 0
        and engineering_violations == 0
        and readback_failures == 0
        and fatal_fallbacks == 0
        and mpc_actions > 0
        and (not runtimes or max(runtimes) < 600.0)
    )
    scientific_benefit_passed = None if delta_vs_baseline is None else bool(delta_vs_baseline < 0.0)
    return {
        "contract": DIRECT_TFV_CLOSED_LOOP_AUDIT_CONTRACT,
        "development_only": True,
        "metadata_path": str(metadata_path),
        "decisions": len(rows),
        "mpc_action_decisions": int(mpc_actions),
        "hold_decisions": int(hold_actions),
        "fatal_fallback_count": int(fatal_fallbacks),
        "grid_violation_count": int(grid_violations),
        "support_violation_count": int(support_violations),
        "engineering_violation_count": int(engineering_violations),
        "target_readback_failure_count": int(readback_failures),
        "screened_facility_count_values": sorted(set(screened_counts)),
        "active_facility_count_min": min(active_counts) if active_counts else 0,
        "active_facility_count_max": max(active_counts) if active_counts else 0,
        "first_move_changed_facility_count_min": min(changed_counts) if changed_counts else 0,
        "first_move_changed_facility_count_max": max(changed_counts) if changed_counts else 0,
        "maximum_support_ratio": max(support_ratios) if support_ratios else 0.0,
        "decision_runtime_seconds_max": max(runtimes) if runtimes else None,
        "selected_predicted_delta_tfv_m3": predicted,
        "authoritative_swmm": {
            "proposed_tfv_m3": proposed_tfv,
            "baseline_tfv_m3": baseline_tfv,
            "delta_tfv_vs_baseline_m3": delta_vs_baseline,
        },
        "execution_passed": execution_passed,
        "scientific_benefit_passed": scientific_benefit_passed,
        "passed": bool(execution_passed and scientific_benefit_passed is not False),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--baseline-node-statistics")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    payload = audit_direct_tfv_closed_loop(
        metadata_path=args.metadata,
        baseline_node_statistics=args.baseline_node_statistics,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
