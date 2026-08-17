"""Audit optimizer-consistent Direct-TFV Development execution and admission semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_direct_tfv_closed_loop_current import audit_direct_tfv_closed_loop

DIRECT_TFV_CALIBRATED_CLOSED_LOOP_AUDIT_CONTRACT = (
    "PROJECT7_DIRECT_TFV_OPTIMIZER_CONSISTENT_CLOSED_LOOP_AUDIT_V2"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--baseline-node-statistics")
    p.add_argument("--baseline-metadata")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = audit_direct_tfv_closed_loop(
        metadata_path=args.metadata,
        baseline_node_statistics=args.baseline_node_statistics,
        baseline_metadata=args.baseline_metadata,
    )
    meta_path = Path(args.metadata).resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (meta_path.parent / str(meta["decision_file"])).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    action_count = hold_count = admission_violations = sequence_support_violations = 0
    upper_bounds: list[float] = []
    raw_predictions: list[float] = []
    sequence_ratios: list[float] = []
    for row in rows:
        source = str(row.get("source", ""))
        diagnostics = row.get("diagnostics") or {}
        selected_source = str(diagnostics.get("direct_tfv_selected_source", ""))
        raw = diagnostics.get("raw_optimized_predicted_delta_tfv_m3")
        upper = diagnostics.get("admission_upper_bound_m3")
        sequence_ratio = diagnostics.get("joint_sequence_support_max_ratio")
        if raw is not None and np.isfinite(float(raw)):
            raw_predictions.append(float(raw))
        if upper is not None and np.isfinite(float(upper)):
            upper_bounds.append(float(upper))
        if sequence_ratio is None or not np.isfinite(float(sequence_ratio)):
            sequence_support_violations += 1
        else:
            ratio = float(sequence_ratio)
            sequence_ratios.append(ratio)
            if ratio > 1.0001:
                sequence_support_violations += 1
        if diagnostics.get("joint_sequence_support_used") is not True:
            sequence_support_violations += 1
        if source == "MPC_DIRECT_TFV_RECEDING":
            action_count += 1
            if diagnostics.get("admission_passed") is not True:
                admission_violations += 1
            if upper is None or not np.isfinite(float(upper)) or float(upper) >= 0.0:
                admission_violations += 1
            if int(diagnostics.get("first_move_changed_facility_count", 0)) <= 0:
                admission_violations += 1
        elif source.startswith("HOLD_DIRECT_TFV_"):
            hold_count += 1
            if selected_source == "DIRECT_TFV_RECEDING_LBFGSB":
                admission_violations += 1
    if action_count + hold_count != len(rows):
        admission_violations += 1
    payload = {
        **base,
        "contract": DIRECT_TFV_CALIBRATED_CLOSED_LOOP_AUDIT_CONTRACT,
        "calibrated_admission_required": True,
        "joint_sequence_support_required": True,
        "calibrated_action_count": int(action_count),
        "calibrated_hold_count": int(hold_count),
        "admission_violation_count": int(admission_violations),
        "joint_sequence_support_violation_count": int(sequence_support_violations),
        "raw_optimized_predicted_delta_tfv_m3": raw_predictions,
        "admission_upper_bound_m3": upper_bounds,
        "joint_sequence_support_max_ratio": sequence_ratios,
    }
    payload["execution_passed"] = bool(
        base["execution_passed"]
        and admission_violations == 0
        and sequence_support_violations == 0
    )
    payload["passed"] = bool(
        payload["execution_passed"] and base["scientific_benefit_passed"] is not False
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
