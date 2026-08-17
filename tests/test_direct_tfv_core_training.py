from __future__ import annotations

import torch

from rtc.step2_tfv_value_training_v4 import (
    DIRECT_TFV_TRAINING_CONTRACT,
    _control_sign_loss,
    _facility_balanced_regression,
)


def test_core_training_contract_is_v4() -> None:
    assert DIRECT_TFV_TRAINING_CONTRACT == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V4"


def test_facility_balanced_regression_does_not_let_repeated_facility_dominate() -> None:
    scale = torch.tensor(1.0)
    prediction = torch.tensor([2.0, 2.0, 2.0, 0.0])
    truth = torch.zeros(4)
    facility = torch.tensor([0, 0, 0, 1])
    balanced = _facility_balanced_regression(
        prediction,
        truth,
        scale_m3=scale,
        facility_ids=facility,
    )
    facility0 = torch.nn.functional.smooth_l1_loss(
        prediction[:3], truth[:3], reduction="mean"
    )
    facility1 = torch.nn.functional.smooth_l1_loss(
        prediction[3:], truth[3:], reduction="mean"
    )
    torch.testing.assert_close(balanced, 0.5 * (facility0 + facility1))


def test_control_sign_loss_is_symmetric_and_prefers_correct_hold_relative_sign() -> None:
    truth = torch.tensor([-1000.0, 1200.0])
    correct = torch.tensor([-900.0, 900.0])
    wrong = -correct
    scale = torch.tensor(1000.0)
    good = _control_sign_loss(
        correct,
        truth,
        scale_m3=scale,
        practical_zero_m3=1.0,
    )
    bad = _control_sign_loss(
        wrong,
        truth,
        scale_m3=scale,
        practical_zero_m3=1.0,
    )
    assert good < bad
