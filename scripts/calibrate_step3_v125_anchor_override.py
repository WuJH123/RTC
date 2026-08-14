"""Calibrate the V125 candidate-vs-anchor TFV admission margin from D4-FIT only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rtc.step3_calibration_v125 import (
    anchor_override_audit_v125,
    calibrate_anchor_override_margin_v125,
    calibration_json_v125,
)

REQUIRED = {
    "split_role",
    "rainfall_group",
    "plan_row_id",
    "truth_candidate_tfv_m3",
    "truth_anchor_tfv_m3",
    "predicted_candidate_delta_tfv_m3",
    "predicted_anchor_delta_tfv_m3",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evidence-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quantile", type=float, default=0.95)
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
        truth_candidate_tfv_m3=fit["truth_candidate_tfv_m3"],
        truth_anchor_tfv_m3=fit["truth_anchor_tfv_m3"],
        predicted_candidate_delta_tfv_m3=fit["predicted_candidate_delta_tfv_m3"],
        predicted_anchor_delta_tfv_m3=fit["predicted_anchor_delta_tfv_m3"],
        rainfall_groups=fit["rainfall_group"].astype(str),
        row_ids=fit["plan_row_id"].astype(str),
        quantile=args.quantile,
    )
    truth_adv = audit["truth_candidate_tfv_m3"].to_numpy() - audit["truth_anchor_tfv_m3"].to_numpy()
    pred_adv = (
        audit["predicted_candidate_delta_tfv_m3"].to_numpy()
        - audit["predicted_anchor_delta_tfv_m3"].to_numpy()
    )
    payload = json.loads(calibration_json_v125(calibration))
    payload["audit"] = anchor_override_audit_v125(
        truth_advantage_m3=truth_adv,
        predicted_advantage_m3=pred_adv,
        margin_m3=calibration.margin_m3,
    )
    payload["boundary"] = {
        "calibration_uses_fit_only": True,
        "audit_used_for_calibration": False,
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
