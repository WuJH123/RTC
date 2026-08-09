from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Paper-strength design targets, not software execution gates. The scientific invariants are
# group-disjoint development train/validation and untouched Final. Calibration/safety-audit
# cohorts are optional unless a later analysis explicitly uses them.
RECOMMENDED_RAINFALL_GROUPS = 160
RECOMMENDED_ROLE_GROUPS = {
    "development": 96,
    "calibration": 24,
    "safety_audit": 16,
    "final": 24,
}
RECOMMENDED_DEV_VALIDATION_GROUPS = 19
_ALLOWED_SPLITS = {"development", "calibration", "safety_audit", "final"}


def validate_formal_rainfall_design(frame: pd.DataFrame) -> dict[str, object]:
    """Validate the rainfall-group split without imposing arbitrary sample-count gates.

    Required for correctness:
    - one authoritative row per event_id;
    - no rainfall group crosses scientific splits;
    - development contains group-disjoint train and validation folds;
    - Final exists and remains separate from development;
    - referenced event INPs exist.

    Larger cohort sizes are reported against recommended paper-strength targets but do not
    prevent the pipeline from running, piloting, resuming or training on the data available.
    """

    required = {
        "event_id",
        "rainfall_group",
        "inp_path",
        "scientific_split",
        "development_fold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"rainfall event registry missing columns: {missing}")
    data = frame.copy()
    for column in ("event_id", "rainfall_group", "scientific_split", "development_fold"):
        data[column] = data[column].fillna("").astype(str)
    if data.empty:
        raise ValueError("rainfall event registry is empty")
    if data["event_id"].duplicated().any():
        raise ValueError("event_id must be unique: one authoritative event INP per event row")
    if (data["rainfall_group"] == "").any():
        raise ValueError("rainfall_group cannot be empty")
    invalid_roles = sorted(set(data["scientific_split"]) - _ALLOWED_SPLITS)
    if invalid_roles:
        raise ValueError(f"unsupported scientific_split values: {invalid_roles}")
    cross = data.groupby("rainfall_group")["scientific_split"].nunique()
    if (cross != 1).any():
        raise ValueError("rainfall-group leakage exists across scientific splits")

    group_roles = data[["rainfall_group", "scientific_split"]].drop_duplicates()
    total = int(group_roles["rainfall_group"].nunique())
    role_counts = {
        str(k): int(v)
        for k, v in group_roles.groupby("scientific_split")["rainfall_group"].count().to_dict().items()
    }
    if role_counts.get("development", 0) < 2:
        raise ValueError("development requires at least two rainfall groups so train/validation can be disjoint")
    if role_counts.get("final", 0) < 1:
        raise ValueError("at least one untouched Final rainfall group is required")

    dev = data[data["scientific_split"] == "development"]
    if set(dev["development_fold"]) != {"train", "validation"}:
        raise ValueError("development groups must contain both train and validation folds")
    if (dev.groupby("rainfall_group")["development_fold"].nunique() != 1).any():
        raise ValueError("rainfall group crosses development train/validation")
    dev_val = int(
        dev.loc[dev["development_fold"] == "validation", "rainfall_group"].nunique()
    )
    if dev_val < 1:
        raise ValueError("development validation requires at least one independent rainfall group")
    if (
        data.loc[data["scientific_split"] != "development", "development_fold"] != ""
    ).any():
        raise ValueError("non-development rows must not carry development_fold")

    descriptor_columns = [
        c
        for c in (
            "total_depth_mm",
            "duration_minutes",
            "peak_intensity_mmhr",
            "antecedent_rainfall_mm",
        )
        if c in data.columns
    ]
    descriptor_summary: dict[str, object] = {}
    for column in descriptor_columns:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"rainfall descriptor {column} must be finite and non-negative")
        descriptor_summary[column] = {
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(np.max(values)),
        }

    missing_inp = [str(p) for p in data["inp_path"] if not Path(str(p)).is_file()]
    if missing_inp:
        raise ValueError(f"event registry references missing INPs: {missing_inp[:10]}")

    recommendations: list[str] = []
    if total < RECOMMENDED_RAINFALL_GROUPS:
        recommendations.append(
            f"paper-strength target is >= {RECOMMENDED_RAINFALL_GROUPS} independent rainfall groups; current={total}"
        )
    for role, target in RECOMMENDED_ROLE_GROUPS.items():
        count = int(role_counts.get(role, 0))
        if count < target:
            recommendations.append(f"recommended {role} groups >= {target}; current={count}")
    if dev_val < RECOMMENDED_DEV_VALIDATION_GROUPS:
        recommendations.append(
            f"recommended development validation groups >= {RECOMMENDED_DEV_VALIDATION_GROUPS}; current={dev_val}"
        )

    return {
        "contract": "RAINFALL_GROUP_SPLIT_VALIDATION_V2",
        "rainfall_groups": total,
        "role_group_counts": role_counts,
        "development_validation_groups": dev_val,
        "event_rows": int(len(data)),
        "descriptor_summary": descriptor_summary,
        "required_invariants_passed": True,
        "recommended_total_groups": RECOMMENDED_RAINFALL_GROUPS,
        "recommended_role_groups": RECOMMENDED_ROLE_GROUPS,
        "recommended_development_validation_groups": RECOMMENDED_DEV_VALIDATION_GROUPS,
        "paper_strength_recommendations_met": not recommendations,
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rainfall-group-disjoint development/Final design and report cohort-size recommendations"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evidence = validate_formal_rainfall_design(pd.read_csv(args.events))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
