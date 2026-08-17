from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rtc.direct_tfv_admission import (
    DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
    _minimum_conformal_sample_size,
    _one_sided_conformal_upper,
    admission_margin_m3,
)
from rtc.step3_tfv_value_mpc_v3 import DirectTFVMPCResultV3
from rtc.step3_tfv_value_mpc_v4 import DirectTFVRecedingMPCV4
from rtc.step3_tfv_value_mpc_v5 import DirectTFVRecedingMPCV5


def _calibration() -> dict:
    return {
        "contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
        "development_only": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "optimizer_replay_count": 6,
        "density_floor_changed_facilities": 20,
        "global_margin_m3": 1000.0,
        "dense_margin_m3": 5000.0,
    }


def _result(*, score: float, changed: int = 3, active: int = 23) -> DirectTFVMPCResultV3:
    settings = torch.full((72, 109), 0.5)
    if changed:
        settings[:2, :changed] = 0.6
    return DirectTFVMPCResultV3(
        settings=settings,
        predicted_delta_tfv_m3=score,
        selected_source="DIRECT_TFV_RECEDING_LBFGSB",
        optimizer_success=True,
        optimizer_steps=3,
        optimizer_starts=2,
        gradient_norm=1.0,
        scipy_message="ok",
        elapsed_seconds=0.2,
        screened_facility_count=109,
        predicted_beneficial_facility_count=40,
        active_facility_count=active,
        active_facility_ids=tuple(f"A{i}" for i in range(active)),
        active_facility_screening_scores_m3=tuple(-100.0 for _ in range(active)),
        first_move_changed_facility_count=changed,
        maximum_support_ratio=1.0,
        training_joint_changed_facility_q90=20.0,
    )


def _bare() -> DirectTFVRecedingMPCV5:
    mpc = object.__new__(DirectTFVRecedingMPCV5)
    mpc.admission_calibration = _calibration()
    mpc.design = SimpleNamespace(prediction_horizon_steps=72)
    return mpc


def test_one_sided_conformal_requires_supported_finite_sample_coverage() -> None:
    assert _minimum_conformal_sample_size(0.90) == 10
    with pytest.raises(ValueError, match="needs at least"):
        _one_sided_conformal_upper([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.90)
    values = [float(value) for value in range(1, 11)]
    assert _one_sided_conformal_upper(values, 0.90) == 10.0


def test_admission_margin_uses_dense_geometry_for_large_active_set() -> None:
    assert admission_margin_m3(_calibration(), 5) == (1000.0, "global")
    assert admission_margin_m3(_calibration(), 20) == (5000.0, "dense")


def test_v5_rejects_optimizer_minimum_when_upper_bound_is_not_beneficial(monkeypatch) -> None:
    monkeypatch.setattr(DirectTFVRecedingMPCV4, "optimize", lambda self, **kwargs: _result(score=-4000.0))
    mpc = _bare()
    active_target = torch.full((109,), 0.5)
    result = mpc.optimize(
        current_state=torch.zeros((1, 1, 1)),
        rainfall=torch.zeros((1, 1, 1, 1)),
        previous_actuator_flow=torch.zeros((1, 109)),
        current_settings=active_target,
        active_target=active_target,
    )
    assert result.selected_source == "HOLD_CALIBRATED_TFV_UPPER_BOUND_NONNEGATIVE"
    assert result.raw_optimized_predicted_delta_tfv_m3 == pytest.approx(-4000.0)
    assert result.admission_upper_bound_m3 == pytest.approx(1000.0)
    assert result.admission_passed is False
    torch.testing.assert_close(result.settings, active_target[None].expand(72, -1))


def test_v5_keeps_strong_action_when_calibrated_upper_bound_is_negative(monkeypatch) -> None:
    monkeypatch.setattr(DirectTFVRecedingMPCV4, "optimize", lambda self, **kwargs: _result(score=-20000.0))
    mpc = _bare()
    active_target = torch.full((109,), 0.5)
    result = mpc.optimize(
        current_state=torch.zeros((1, 1, 1)),
        rainfall=torch.zeros((1, 1, 1, 1)),
        previous_actuator_flow=torch.zeros((1, 109)),
        current_settings=active_target,
        active_target=active_target,
    )
    assert result.selected_source == "DIRECT_TFV_RECEDING_LBFGSB"
    assert result.admission_upper_bound_m3 == pytest.approx(-15000.0)
    assert result.admission_passed is True


def test_v5_rejects_future_only_noop_first_move(monkeypatch) -> None:
    monkeypatch.setattr(
        DirectTFVRecedingMPCV4,
        "optimize",
        lambda self, **kwargs: _result(score=-20000.0, changed=0, active=1),
    )
    mpc = _bare()
    active_target = torch.full((109,), 0.5)
    result = mpc.optimize(
        current_state=torch.zeros((1, 1, 1)),
        rainfall=torch.zeros((1, 1, 1, 1)),
        previous_actuator_flow=torch.zeros((1, 109)),
        current_settings=active_target,
        active_target=active_target,
    )
    assert result.selected_source == "HOLD_NO_EXECUTABLE_FIRST_MOVE"
    assert result.admission_passed is False
