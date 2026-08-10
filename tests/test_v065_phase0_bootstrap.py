from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.checkpoint_design import _eligible_checkpoint_mask
from rtc.phase0_design import design_phase0_events
from rtc.timing_freeze import freeze_phase0_timing


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(12):
        rows.append(
            {
                "event_id": f"E{i:02d}",
                "rainfall_group": f"R{i:02d}",
                "inp_path": f"event_{i:02d}.inp",
                "scientific_split": "development" if i < 10 else "final",
                "development_fold": "train" if i < 8 else ("validation" if i < 10 else ""),
                "total_depth_mm": 10.0 + 2.0 * i,
                "duration_minutes": 60.0 + 15.0 * i,
                "peak_intensity_mmhr": 20.0 + 5.0 * (i % 4),
                "antecedent_rainfall_mm": float(i % 3),
            }
        )
    return pd.DataFrame(rows)


def test_phase0_selector_is_development_train_only_and_deterministic() -> None:
    first, summary1 = design_phase0_events(_events(), rainfall_groups=6, seed=42)
    second, summary2 = design_phase0_events(_events(), rainfall_groups=6, seed=42)
    assert first["event_id"].tolist() == second["event_id"].tolist()
    assert first["rainfall_group"].nunique() == 6
    assert (first["scientific_split"] == "development").all()
    assert (first["development_fold"] == "train").all()
    assert summary1["hydraulic_outcomes_used"] is False
    assert summary1["selection_mode"] == "DETERMINISTIC_FARTHEST_POINT_FORCING_COVERAGE"
    assert summary1["selected_rainfall_groups"] == summary2["selected_rainfall_groups"]


def test_phase0_selector_can_fallback_without_descriptors() -> None:
    frame = _events().drop(
        columns=[
            "total_depth_mm",
            "duration_minutes",
            "peak_intensity_mmhr",
            "antecedent_rainfall_mm",
        ]
    )
    selected, summary = design_phase0_events(frame, rainfall_groups=4, seed=7)
    assert selected["rainfall_group"].nunique() == 4
    assert summary["forcing_descriptor_columns"] == []
    assert summary["selection_mode"] == "SEEDED_GROUP_ONLY_FALLBACK_NO_FORCING_DESCRIPTORS"


def test_checkpoint_selection_reserves_the_requested_future_tail() -> None:
    elapsed = np.array([0, 1800, 3600, 5400, 7200, 9000], dtype=int)

    eligible = _eligible_checkpoint_mask(
        elapsed,
        minimum_elapsed_minutes=30,
        minimum_tail_minutes=60,
    )

    assert elapsed[eligible].tolist() == [1800, 3600, 5400]


def _phase0_summary(path: Path, *, censored: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "contract": "PHASE0_D2_STEP_RESPONSE_TIMESCALE_V5_HIGH_FREQUENCY",
                "horizon_censored": censored,
                "candidate_production_timing": {
                    "model_observation_seconds": 300,
                    "control_update_seconds": 600,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_timing_freeze_binds_project7_fixed_360min_grid(tmp_path: Path) -> None:
    summary = _phase0_summary(tmp_path / "timescale.json", censored=False)
    payload = freeze_phase0_timing(
        phase0_summary_path=summary,
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_minutes=360,
        control_start_minutes=60,
        max_setting_delta_per_update=0.5,
    )
    assert payload["contract"] == "RTC_PHASE0_TIMING_FREEZE_V2_PROJECT7_360MIN"
    assert payload["timing"]["history_span_seconds"] == 3600
    assert payload["timing"]["horizon_steps"] == 72
    assert payload["timing"]["d3_control_blocks"] == 36
    assert payload["controller"]["max_setting_delta_per_update"] == 0.5


def test_timing_freeze_keeps_censored_finding_as_diagnostic(tmp_path: Path) -> None:
    summary = _phase0_summary(tmp_path / "timescale.json", censored=True)
    payload = freeze_phase0_timing(
        phase0_summary_path=summary,
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_minutes=360,
        control_start_minutes=60,
        max_setting_delta_per_update=0.5,
    )
    assert payload["phase0_horizon_censored"] is True
    assert payload["phase0_censor_role"] == "diagnostic_not_horizon_selection_gate"
    assert payload["horizon_selection_basis"] == "USER_FROZEN_IDEALIZED_METHODOLOGY_TESTBED_360MIN"
