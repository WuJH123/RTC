"""Calibrate V125 direct candidate-vs-anchor TFV/PFV risk from D4-FIT only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rtc.step3_calibration_v125 import (
    anchor_override_audit_v125,
    calibrate_anchor_override_margin_v125,
    calibration_json_v125,
    pfv_deterioration_audit_v125,
)

REQUIRED = {
    "split_role",
    "rainfall_group",
    "plan_row_id",
    "truth_tfv_advantage_m3",
    "predicted_tfv_advantage_m3",
    "truth_pfv_advantage_m3",
    "predicted_pfv_advantage_m3",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evidence-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--pfv-soft-margin-m3", type=float, required=True)
    args = p.parse_args()

    frame = pd.read_csv(args.evidence_csv)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"V125 evidence CSV missing columns: {missing}")
    fit = frame[frame["split_role"].astype(str) == "fit"].copy()
    audit = frame[frame["split_role"].astype(str) == "audit"].copy()
    if fit.empty or audit.empty:
        raise ValueError("V125 requires pre-frozen non-empty D4 fit and audit roles")
    if set(fit["rainfall_group"].astype(str)) & set(audit["rainfall_group"].astype(str)):
        raise ValueError("V125 D4 fit/audit rainfall groups overlap")

    calibration = calibrate_anchor_override_margin_v125(
        truth_tfv_advantage_m3=fit["truth_tfv_advantage_m3"],
        predicted_tfv_advantage_m3=fit["predicted_tfv_advantage_m3"],
        truth_pfv_advantage_m3=fit["truth_pfv_advantage_m3"],
        predicted_pfv_advantage_m3=fit["predicted_pfv_advantage_m3"],
        rainfall_groups=fit["rainfall_group"].astype(str),
        row_ids=fit["plan_row_id"].astype(str),
        quantile=args.quantile,
    )
    payload = json.loads(calibration_json_v125(calibration))
    payload["audit"] = {
        "tfv": anchor_override_audit_v125(
            truth_advantage_m3=audit["truth_tfv_advantage_m3"],
            predicted_advantage_m3=audit["predicted_tfv_advantage_m3"],
            margin_m3=calibration.tfv_margin_m3,
        ),
        "pfv": pfv_deterioration_audit_v125(
            truth_advantage_m3=audit["truth_pfv_advantage_m3"],
            predicted_advantage_m3=audit["predicted_pfv_advantage_m3"],
            error_margin_m3=calibration.pfv_error_margin_m3,
            soft_margin_m3=float(args.pfv_soft_margin_m3),
        ),
    }
    payload["boundary"] = {
        "calibration_uses_fit_only": True,
        "audit_used_for_calibration": False,
        "direct_anchor_relative_targets": True,
        "pfv_is_soft_not_hard": True,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
