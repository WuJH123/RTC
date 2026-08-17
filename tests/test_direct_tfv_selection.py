from __future__ import annotations

import numpy as np

from rtc.step2_tfv_selection import (
    DirectTFVSelectionDesign,
    calibrate_selected_action_margin,
    evaluate_selection_margin,
    finite_sample_upper_quantile,
)


def test_finite_sample_upper_quantile_uses_noninterpolated_upper_rank() -> None:
    value, rank = finite_sample_upper_quantile([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.20)
    assert rank == 5
    assert value == 5.0


def test_calibration_uses_only_raw_action_selected_groups() -> None:
    calibration = calibrate_selected_action_margin(
        best_candidate_prediction_m3=[-100.0, 10.0, -200.0, -300.0, -400.0, -500.0],
        best_candidate_truth_m3=[50.0, -100.0, -50.0, 100.0, -450.0, -600.0],
        design=DirectTFVSelectionDesign(alpha=0.20),
    )
    # The positive-prediction group is already HOLD under the raw policy and must not dilute the
    # selected-action residual calibration.
    assert calibration.calibration_groups == 6
    assert calibration.calibration_action_groups == 5
    assert len(calibration.residuals_m3) == 5
    assert calibration.margin_m3 >= 0.0


def test_guard_can_replace_optimistic_harmful_action_with_hold() -> None:
    raw = evaluate_selection_margin(
        best_candidate_prediction_m3=[-100.0, -50.0],
        best_candidate_truth_m3=[200.0, -300.0],
        oracle_truth_m3=[0.0, -300.0],
        oracle_is_hold=[True, False],
        margin_m3=0.0,
    )
    guarded = evaluate_selection_margin(
        best_candidate_prediction_m3=[-100.0, -50.0],
        best_candidate_truth_m3=[200.0, -300.0],
        oracle_truth_m3=[0.0, -300.0],
        oracle_is_hold=[True, False],
        margin_m3=120.0,
    )
    assert raw["false_action_when_hold_oracle_fraction"] == 1.0
    assert raw["selected_harmful_fraction"] == 0.5
    assert guarded["false_action_when_hold_oracle_fraction"] == 0.0
    assert guarded["selected_harmful_fraction"] == 0.0
    assert guarded["hold_selected_fraction"] == 1.0


def test_zero_margin_matches_raw_hold_rule() -> None:
    metrics = evaluate_selection_margin(
        best_candidate_prediction_m3=[-1.0, 0.0, 1.0],
        best_candidate_truth_m3=[-10.0, -20.0, -30.0],
        oracle_truth_m3=[-10.0, -20.0, -30.0],
        oracle_is_hold=[False, False, False],
        margin_m3=0.0,
    )
    assert np.isclose(metrics["action_selected_fraction"], 1.0 / 3.0)
    assert np.isclose(metrics["hold_selected_fraction"], 2.0 / 3.0)
