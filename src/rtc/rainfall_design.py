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
_GENERIC_SPLITS = {"development", "calibration", "safety_audit", "final"}
_ACTIVE_SPLITS = {"development", "final"}


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ("event_id", "rainfall_group", "scientific_split", "development_fold"):
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str)
    return data


def _descriptor_summary(data: pd.DataFrame) -> dict[str, object]:
    columns = [
        c
        for c in (
            "total_depth_mm",
            "duration_minutes",
            "peak_intensity_mmhr",
            "antecedent_rainfall_mm",
        )
        if c in data.columns
    ]
    result: dict[str, object] = {}
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"rainfall descriptor {column} must be finite and non-negative")
        result[column] = {
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(np.max(values)),
        }
    return result


def validate_formal_rainfall_design(frame: pd.DataFrame) -> dict[str, object]:
    """Generic leakage/readiness validator retained for reusable workspace utilities.

    This function is intentionally cohort-size agnostic and accepts historical synthetic
    calibration/safety roles used by generic regression fixtures. It does **not** authorize
    those roles for Project7 v0.6.9 Formal execution. The active workflow and CLI additionally
    call :func:`validate_project7_v069_rainfall_design`, which rejects them and locks 18/6/6.
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
    data = _normalize(frame)
    if data.empty:
        raise ValueError("rainfall event registry is empty")
    if data["event_id"].duplicated().any():
        raise ValueError("event_id must be unique: one authoritative event INP per event row")
    if (data["rainfall_group"] == "").any():
        raise ValueError("rainfall_group cannot be empty")

    invalid_roles = sorted(set(data["scientific_split"]) - _GENERIC_SPLITS)
    if invalid_roles:
        raise ValueError(f"unsupported scientific_split values: {invalid_roles}")
    cross = data.groupby("rainfall_group")["scientific_split"].nunique()
    if (cross != 1).any():
        raise ValueError("rainfall-group leakage exists across scientific splits")

    group_roles = data[["rainfall_group", "scientific_split"]].drop_duplicates()
    role_counts = {
        str(k): int(v)
        for k, v in group_roles.groupby("scientific_split")["rainfall_group"].count().to_dict().items()
    }
    if role_counts.get("development", 0) < 2:
        raise ValueError("development requires at least two rainfall groups for train/validation")
    if role_counts.get("final", 0) < 1:
        raise ValueError("at least one untouched Final rainfall group is required")

    dev = data[data["scientific_split"] == "development"].copy()
    if set(dev["development_fold"]) != {"train", "validation"}:
        raise ValueError("development groups must contain both train and validation folds")
    if (dev.groupby("rainfall_group")["development_fold"].nunique() != 1).any():
        raise ValueError("rainfall group crosses development train/validation")
    non_dev = data[data["scientific_split"] != "development"]
    if (non_dev["development_fold"] != "").any():
        raise ValueError("non-development rows must not carry development_fold")

    missing_inp = [str(p) for p in data["inp_path"] if not Path(str(p)).is_file()]
    if missing_inp:
        raise ValueError(f"event registry references missing INPs: {missing_inp[:10]}")

    return {
        "contract": "GENERIC_RAINFALL_GROUP_SPLIT_INVARIANTS_V3_COMPATIBILITY_ONLY",
        "rainfall_groups": int(data["rainfall_group"].nunique()),
        "role_group_counts": role_counts,
        "development_train_groups": int(
            dev.loc[dev["development_fold"] == "train", "rainfall_group"].nunique()
        ),
        "development_validation_groups": int(
            dev.loc[dev["development_fold"] == "validation", "rainfall_group"].nunique()
        ),
        "final_groups": int(role_counts.get("final", 0)),
        "descriptor_summary": _descriptor_summary(data),
        "project7_v069_formal_authorization": False,
        "required_invariants_passed": True,
    }


def _duration_counts(frame: pd.DataFrame) -> dict[int, int]:
    return {
        int(k): int(v)
        for k, v in frame.groupby(frame["duration_minutes"].astype(int))["event_id"].count().to_dict().items()
    }


def validate_project7_v069_rainfall_design(frame: pd.DataFrame) -> dict[str, object]:
    """Strictly validate the preregistered Project7 v0.6.9 18/6/6 forcing-only split."""

    generic = validate_formal_rainfall_design(frame)
    data = _normalize(frame)
    present_roles = set(data["scientific_split"])
    if present_roles != _ACTIVE_SPLITS:
        raise ValueError(
            "obsolete/unsupported roles in active Project7 v0.6.9 registry: "
            f"{sorted(present_roles - _ACTIVE_SPLITS)}"
        )
    required = {"return_period_year", "duration_minutes"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Project7 v0.6.9 split lacks forcing columns: {missing}")
    if len(data) != REQUIRED_RAINFALL_GROUPS:
        raise ValueError(
            f"active Project7 registry must contain exactly {REQUIRED_RAINFALL_GROUPS} event rows"
        )
    if data["rainfall_group"].nunique() != REQUIRED_RAINFALL_GROUPS:
        raise ValueError("active Project7 registry must contain 30 independent rainfall groups")
    if generic["role_group_counts"] != REQUIRED_ROLE_GROUPS:
        raise ValueError(
            f"active split must be development=24/final=6, got {generic['role_group_counts']}"
        )

    dev = data[data["scientific_split"] == "development"].copy()
    train = dev[dev["development_fold"] == "train"].copy()
    validation = dev[dev["development_fold"] == "validation"].copy()
    final = data[data["scientific_split"] == "final"].copy()
    if len(train) != REQUIRED_DEV_TRAIN_GROUPS or len(validation) != REQUIRED_DEV_VALIDATION_GROUPS:
        raise ValueError("development split must be exactly 18 Train + 6 Validation rainfall groups")

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

    return {
        **generic,
        "contract": "PROJECT7_V069_RAINFALL_SPLIT_VALIDATION_18_6_6_V1",
        "rainfall_groups": REQUIRED_RAINFALL_GROUPS,
        "role_group_counts": REQUIRED_ROLE_GROUPS,
        "development_train_groups": int(len(train)),
        "development_validation_groups": int(len(validation)),
        "final_groups": int(len(final)),
        "train_duration_counts": _duration_counts(train),
        "validation_duration_counts": _duration_counts(validation),
        "final_duration_counts": _duration_counts(final),
        "validation_return_periods": sorted(set(validation["return_period_year"].astype(int))),
        "final_return_periods": sorted(set(final["return_period_year"].astype(int))),
        "forcing_only_split_preregistered": True,
        "hydraulic_outcomes_used_for_split": False,
        "calibration_role_active": False,
        "safety_audit_role_active": False,
        "project7_v069_formal_authorization": True,
        "required_invariants_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Project7 v0.6.9 18 Train / 6 Validation / 6 Final registry"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evidence = validate_project7_v069_rainfall_design(
        pd.read_csv(args.events, keep_default_na=False)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
