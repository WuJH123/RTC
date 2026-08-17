from __future__ import annotations

import numpy as np
import torch

from rtc.step2_tfv_selection_v2 import (
    calibrate_minimum_predicted_improvement,
    evaluate_selection_threshold,
)
from rtc.step2_tfv_value_training_v3 import _hold_action_loss, _oracle_choice_loss


def test_hold_action_loss_prefers_correct_sign_and_penalizes_false_benefit() -> None:
    truth = torch.tensor([-1000.0, 1200.0, 2500.0])
    correct = torch.tensor([-900.0, 900.0, 2000.0])
    wrong = torch.tensor([900.0, -900.0, -2000.0])
    scale = torch.tensor(1000.0)
    good = _hold_action_loss(
        correct,
        truth,
        scale_m3=scale,
        harmful_false_benefit_weight=2.0,
        practical_zero_m3=1.0,
    )
    bad = _hold_action_loss(
        wrong,
        truth,
        scale_m3=scale,
        harmful_false_benefit_weight=2.0,
        practical_zero_m3=1.0,
    )
    assert good < bad


def test_oracle_choice_loss_includes_hold_as_real_option() -> None:
    truth = torch.tensor([500.0, 1000.0])  # HOLD=0 is authoritative oracle.
    safe_prediction = torch.tensor([400.0, 800.0])
    unsafe_prediction = torch.tensor([-400.0, -800.0])
    scale = torch.tensor(1000.0)
    assert _oracle_choice_loss(safe_prediction, truth, scale_m3=scale) < _oracle_choice_loss(
        unsafe_prediction, truth, scale_m3=scale
    )


def test_empirical_threshold_keeps_zero_harm_beneficial_actions_instead_of_hold_all() -> None:
    prediction = [-100.0, -90.0, -80.0, -20.0]
    truth = [-500.0, -300.0, 200.0, 100.0]
    oracle = [-500.0, -300.0, 0.0, 0.0]
    oracle_hold = [False, False, True, True]
    calibration = calibrate_minimum_predicted_improvement(
        best_candidate_prediction_m3=prediction,
        best_candidate_truth_m3=truth,
        oracle_truth_m3=oracle,
        oracle_is_hold=oracle_hold,
    )
    metrics = evaluate_selection_threshold(
        best_candidate_prediction_m3=prediction,
        best_candidate_truth_m3=truth,
        oracle_truth_m3=oracle,
        oracle_is_hold=oracle_hold,
        minimum_predicted_improvement_m3=calibration.minimum_predicted_improvement_m3,
    )
    assert metrics["selected_harmful_count"] == 0
    assert metrics["action_selected_count"] > 0
    assert metrics["selected_true_delta_tfv_m3"] < 0.0
    assert np.isfinite(calibration.minimum_predicted_improvement_m3)
