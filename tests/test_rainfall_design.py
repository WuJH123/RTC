from __future__ import annotations

from pathlib import Path

import pandas as pd

from rtc.rainfall_design import validate_formal_rainfall_design
from rtc.splits import assign_rainfall_group_splits


def _registry(tmp_path: Path, n: int = 160) -> pd.DataFrame:
    inp = tmp_path / "event.inp"
    inp.write_text("[OPTIONS]\nFLOW_UNITS CMS\n", encoding="utf-8")
    base = pd.DataFrame(
        {
            "event_id": [f"e{i:03d}" for i in range(n)],
            "rainfall_group": [f"g{i:03d}" for i in range(n)],
            "inp_path": [str(inp)] * n,
            "total_depth_mm": [20.0 + i * 0.5 for i in range(n)],
            "duration_minutes": [60 + (i % 12) * 30 for i in range(n)],
            "peak_intensity_mmhr": [15.0 + (i % 20) for i in range(n)],
        }
    )
    return assign_rainfall_group_splits(base, seed=42)


def test_160_group_default_split_meets_recommended_paper_design(tmp_path: Path) -> None:
    frame = _registry(tmp_path, 160)
    evidence = validate_formal_rainfall_design(frame)
    assert evidence["rainfall_groups"] == 160
    assert evidence["role_group_counts"] == {
        "calibration": 24,
        "development": 96,
        "final": 24,
        "safety_audit": 16,
    }
    assert evidence["development_validation_groups"] == 19
    assert evidence["required_invariants_passed"] is True
    assert evidence["paper_strength_recommendations_met"] is True


def test_smaller_valid_design_runs_but_reports_paper_strength_recommendations(tmp_path: Path) -> None:
    frame = _registry(tmp_path, 120)
    evidence = validate_formal_rainfall_design(frame)
    assert evidence["required_invariants_passed"] is True
    assert evidence["rainfall_groups"] == 120
    assert evidence["paper_strength_recommendations_met"] is False
    assert evidence["recommendations"]
