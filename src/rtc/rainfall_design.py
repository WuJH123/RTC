from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_RAINFALL_GROUPS = 30
REQUIRED_ROLE_GROUPS = {"development": 24, "final": 6}
REQUIRED_DEV_TRAIN_GROUPS = 18
REQUIRED_DEV_VALIDATION_GROUPS = 6
FROZEN_DURATIONS = {60, 120, 180, 240, 300, 360}
FROZEN_RETURN_PERIODS = {5, 10, 20, 50, 100}
_ALLOWED_SPLITS = {"development", "final"}


def _duration_counts(frame: pd.DataFrame) -> dict[int, int]:
    return {
        int(k): int(v)
        for k, v in frame.groupby(frame["duration_minutes"].astype(int))["event_id"].count().to_dict().items()
    }


def validate_formal_rainfall_design(frame: pd.DataFrame) -> dict[str, object]:
    """Validate the preregistered Project7 v0.6.9 18/6/6 forcing-only split.

    Active correctness requirements:
    - exactly 30 independent rainfall groups;
    - top-level roles are only development and untouched Final;
    - development = 18 Train + 6 held-out Validation;
    - Final = 6;
    - no rainfall group crosses roles/folds;
    - Train has three events at every frozen duration;
    - Validation and Final each have one event at every frozen duration;
    - Validation and Final each span all five return periods;
    - referenced INPs exist.

    Calibration/safety-audit and the former 160-group recommendation are obsolete for this
    execution contract and are intentionally not accepted as active roles.
    """

    required = {
        "event_id",
        "rainfall_group",
        "inp_path",
        "scientific_split",
        "development_fold",
        "return_period_year",
        "duration_minutes",
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
    if len(data) != REQUIRED_RAINFALL_GROUPS:
        raise ValueError(
            f"active Project7 registry must contain exactly {REQUIRED_RAINFALL_GROUPS} event rows"
        )
    if data["rainfall_group"].nunique() != REQUIRED_RAINFALL_GROUPS:
        raise ValueError("active Project7 registry must contain 30 independent rainfall groups")

    invalid_roles = sorted(set(data["scientific_split"]) - _ALLOWED_SPLITS)
    if invalid_roles:
        raise ValueError(
            f"obsolete/unsupported scientific_split values in active registry: {invalid_roles}"
        )
    cross = data.groupby("rainfall_group")["scientific_split"].nunique()
    if (cross != 1).any():
        raise ValueError("rainfall-group leakage exists across scientific splits")

    group_roles = data[["rainfall_group", "scientific_split"]].drop_duplicates()
    role_counts = {
        str(k): int(v)
        for k, v in group_roles.groupby("scientific_split")["rainfall_group"].count().to_dict().items()
    }
    if role_counts != REQUIRED_ROLE_GROUPS:
        raise ValueError(
            f"active split must be development=24/final=6, got {role_counts}"
        )

    dev = data[data["scientific_split"] == "development"].copy()
    final = data[data["scientific_split"] == "final"].copy()
    if set(dev["development_fold"]) != {"train", "validation"}:
        raise ValueError("development groups must contain both train and validation folds")
    if (dev.groupby("rainfall_group")["development_fold"].nunique() != 1).any():
        raise ValueError("rainfall group crosses development train/validation")
    train = dev[dev["development_fold"] == "train"].copy()
    validation = dev[dev["development_fold"] == "validation"].copy()
    if len(train) != REQUIRED_DEV_TRAIN_GROUPS or len(validation) != REQUIRED_DEV_VALIDATION_GROUPS:
        raise ValueError(
            "development split must be exactly 18 Train + 6 Validation rainfall groups"
        )
    if (final["development_fold"] != "").any():
        raise ValueError("Final rows must not carry development_fold")

    durations = set(pd.to_numeric(data["duration_minutes"], errors="raise").astype(int))
    return_periods = set(pd.to_numeric(data["return_period_year"], errors="raise").astype(int))
    if durations != FROZEN_DURATIONS:
        raise ValueError(f"duration levels differ from frozen set: {sorted(durations)}")
    if return_periods != FROZEN_RETURN_PERIODS:
        raise ValueError(f"return-period levels differ from frozen set: {sorted(return_periods)}")

    if _duration_counts(train) != {duration: 3 for duration in sorted(FROZEN_DURATIONS)}:
        raise ValueError("Train must contain exactly three events at every frozen duration")
    if _duration_counts(validation) != {duration: 1 for duration in sorted(FROZEN_DURATIONS)}:
        raise ValueError("Validation must contain exactly one event at every frozen duration")
    if _duration_counts(final) != {duration: 1 for duration in sorted(FROZEN_DURATIONS)}:
        raise ValueError("Final must contain exactly one event at every frozen duration")
    if set(validation["return_period_year"].astype(int)) != FROZEN_RETURN_PERIODS:
        raise ValueError("Validation must span all five frozen return periods")
    if set(final["return_period_year"].astype(int)) != FROZEN_RETURN_PERIODS:
        raise ValueError("Final must span all five frozen return periods")

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
        "contract": "PROJECT7_V069_RAINFALL_SPLIT_VALIDATION_18_6_6_V1",
        "rainfall_groups": REQUIRED_RAINFALL_GROUPS,
        "role_group_counts": role_counts,
        "development_train_groups": int(len(train)),
        "development_validation_groups": int(len(validation)),
        "final_groups": int(len(final)),
        "train_duration_counts": _duration_counts(train),
        "validation_duration_counts": _duration_counts(validation),
        "final_duration_counts": _duration_counts(final),
        "validation_return_periods": sorted(set(validation["return_period_year"].astype(int))),
        "final_return_periods": sorted(set(final["return_period_year"].astype(int))),
        "descriptor_summary": descriptor_summary,
        "forcing_only_split_preregistered": True,
        "hydraulic_outcomes_used_for_split": False,
        "calibration_role_active": False,
        "safety_audit_role_active": False,
        "required_invariants_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Project7 v0.6.9 18 Train / 6 Validation / 6 Final registry"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evidence = validate_formal_rainfall_design(pd.read_csv(args.events, keep_default_na=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
