from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORMAL_MIN_RAINFALL_GROUPS = 160
FORMAL_MIN_ROLE_GROUPS = {
    "development": 96,
    "calibration": 24,
    "safety_audit": 16,
    "final": 24,
}
FORMAL_MIN_DEV_VALIDATION_GROUPS = 19


def validate_formal_rainfall_design(frame: pd.DataFrame) -> dict[str, object]:
    """Validate a fresh, rainfall-group-disjoint Formal event registry.

    160 is a project design minimum, not a universal hydrological constant. It is chosen for
    this very large 109-actuator study so all four scientific roles remain independently
    populated, including 24 untouched Final groups and ~19 development-validation groups.
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
    if data["event_id"].duplicated().any():
        raise ValueError("event_id must be unique: one authoritative event INP per event row")
    if (data["rainfall_group"] == "").any():
        raise ValueError("rainfall_group cannot be empty")
    cross = data.groupby("rainfall_group")["scientific_split"].nunique()
    if (cross != 1).any():
        raise ValueError("rainfall-group leakage exists across scientific splits")

    group_roles = data[["rainfall_group", "scientific_split"]].drop_duplicates()
    total = int(group_roles["rainfall_group"].nunique())
    if total < FORMAL_MIN_RAINFALL_GROUPS:
        raise ValueError(
            f"Formal design requires >= {FORMAL_MIN_RAINFALL_GROUPS} independent rainfall groups; got {total}"
        )
    role_counts = group_roles.groupby("scientific_split")["rainfall_group"].count().to_dict()
    for role, minimum in FORMAL_MIN_ROLE_GROUPS.items():
        count = int(role_counts.get(role, 0))
        if count < minimum:
            raise ValueError(f"{role} requires >= {minimum} rainfall groups; got {count}")

    dev = data[data["scientific_split"] == "development"]
    if set(dev["development_fold"]) != {"train", "validation"}:
        raise ValueError("development groups must be split into train and validation")
    if (dev.groupby("rainfall_group")["development_fold"].nunique() != 1).any():
        raise ValueError("rainfall group crosses development train/validation")
    dev_val = int(
        dev.loc[dev["development_fold"] == "validation", "rainfall_group"].nunique()
    )
    if dev_val < FORMAL_MIN_DEV_VALIDATION_GROUPS:
        raise ValueError(
            f"development validation requires >= {FORMAL_MIN_DEV_VALIDATION_GROUPS} rainfall groups; got {dev_val}"
        )
    if (
        data.loc[data["scientific_split"] != "development", "development_fold"] != ""
    ).any():
        raise ValueError("non-development rows must not carry development_fold")

    # If event descriptors are provided, make them part of the evidence and reject invalid
    # values. The generator/source of rainfall can vary, so the validator does not invent IDF
    # bounds or synthetic distributions that are not supported by project data.
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

    return {
        "contract": "FORMAL_FRESH_RAINFALL_COHORTS_V1",
        "minimum_total_groups": FORMAL_MIN_RAINFALL_GROUPS,
        "rainfall_groups": total,
        "role_group_counts": {str(k): int(v) for k, v in role_counts.items()},
        "development_validation_groups": dev_val,
        "event_rows": int(len(data)),
        "descriptor_summary": descriptor_summary,
        "fresh_hydraulic_data_instruction": "Generate all SWMM baselines/D0/D1/D2/D3/closed-loop outputs into a new output root under the current code contract; do not import historical RTC outputs/models.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the >=160-group fresh Formal rainfall/event registry before expensive SWMM generation"
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
