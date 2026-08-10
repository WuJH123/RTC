from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rtc.rainfall_design import validate_formal_rainfall_design
from rtc.splits import assign_rainfall_group_splits


FINAL = {
    "T5_D360",
    "T10_D60",
    "T20_D180",
    "T20_D240",
    "T50_D120",
    "T100_D300",
}
VALIDATION = {
    "T5_D180",
    "T10_D300",
    "T20_D60",
    "T50_D240",
    "T50_D360",
    "T100_D120",
}


def _active_registry(tmp_path: Path) -> pd.DataFrame:
    inp = tmp_path / "event.inp"
    inp.write_text("[OPTIONS]\nFLOW_UNITS CMS\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    for return_period in (5, 10, 20, 50, 100):
        for duration in (60, 120, 180, 240, 300, 360):
            event_id = f"T{return_period}_D{duration}"
            if event_id in FINAL:
                split, fold = "final", ""
            elif event_id in VALIDATION:
                split, fold = "development", "validation"
            else:
                split, fold = "development", "train"
            rows.append(
                {
                    "event_id": event_id,
                    "rainfall_group": event_id,
                    "inp_path": str(inp),
                    "return_period_year": return_period,
                    "duration_minutes": duration,
                    "total_depth_mm": float(return_period + duration / 10),
                    "peak_intensity_mmhr": float(return_period + duration / 5),
                    "antecedent_rainfall_mm": 0.0,
                    "scientific_split": split,
                    "development_fold": fold,
                }
            )
    return pd.DataFrame(rows)


def test_active_30_event_split_is_exactly_18_6_6(tmp_path: Path) -> None:
    evidence = validate_formal_rainfall_design(_active_registry(tmp_path))
    assert evidence["rainfall_groups"] == 30
    assert evidence["role_group_counts"] == {"development": 24, "final": 6}
    assert evidence["development_train_groups"] == 18
    assert evidence["development_validation_groups"] == 6
    assert evidence["final_groups"] == 6
    assert evidence["train_duration_counts"] == {
        60: 3,
        120: 3,
        180: 3,
        240: 3,
        300: 3,
        360: 3,
    }
    assert evidence["validation_duration_counts"] == {
        60: 1,
        120: 1,
        180: 1,
        240: 1,
        300: 1,
        360: 1,
    }
    assert evidence["final_duration_counts"] == evidence["validation_duration_counts"]
    assert evidence["validation_return_periods"] == [5, 10, 20, 50, 100]
    assert evidence["final_return_periods"] == [5, 10, 20, 50, 100]
    assert evidence["required_invariants_passed"] is True


def test_obsolete_calibration_or_safety_roles_are_rejected(tmp_path: Path) -> None:
    frame = _active_registry(tmp_path)
    frame.loc[0, "scientific_split"] = "calibration"
    frame.loc[0, "development_fold"] = ""
    with pytest.raises(ValueError, match="obsolete/unsupported"):
        validate_formal_rainfall_design(frame)


def test_generic_split_helper_no_longer_creates_calibration_or_safety(tmp_path: Path) -> None:
    inp = tmp_path / "event.inp"
    inp.write_text("[OPTIONS]\nFLOW_UNITS CMS\n", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "event_id": [f"e{i:02d}" for i in range(30)],
            "rainfall_group": [f"g{i:02d}" for i in range(30)],
            "inp_path": [str(inp)] * 30,
        }
    )
    out = assign_rainfall_group_splits(frame, seed=42)
    assert set(out["scientific_split"]) == {"development", "final"}
    assert (out["scientific_split"] == "development").sum() == 24
    assert (out["scientific_split"] == "final").sum() == 6
    dev = out[out["scientific_split"] == "development"]
    assert (dev["development_fold"] == "train").sum() == 18
    assert (dev["development_fold"] == "validation").sum() == 6
