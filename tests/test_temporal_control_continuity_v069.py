from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.closed_loop import CausalObservation, ControllerAction
from rtc.project7_contract import (
    PRODUCTION_CONTROLLER_CONTRACT,
    validate_project7_runtime_config,
)
from rtc.runtime import choose_first_move, command_continuity
from rtc.runtime_controller_guard import ContinuityGuardController
from rtc.tfv_mpc import _project_block_settings_
from rtc.timing_freeze import freeze_phase0_timing


def _observation(current: float, target: float | None = None) -> CausalObservation:
    value = current if target is None else target
    return CausalObservation(
        elapsed_seconds=3600,
        current_time=datetime(2020, 1, 1, 1, 0, 0),
        sensor_ids=(),
        sensor_depth_m=np.asarray([], dtype=float),
        sensor_head_m=np.asarray([], dtype=float),
        actuator_ids=("A1",),
        actuator_target_setting=np.asarray([value], dtype=float),
        actuator_current_setting=np.asarray([current], dtype=float),
        actuator_flow_m3s=np.asarray([0.0], dtype=float),
        rainfall_node_ids=(),
        observed_rainfall_mmhr=np.asarray([], dtype=float),
    )


def _project7_config() -> dict[str, object]:
    return {
        "contract": PRODUCTION_CONTROLLER_CONTRACT,
        "model_step_seconds": 300,
        "control_update_seconds": 600,
        "record_stride_seconds": 300,
        "control_start_minutes": 60,
        "exact_global_peak": False,
        "methodology_testbed": {
            "claim_scope": "IDEALIZED_METHODOLOGY_TESTBED_NOT_FIELD_DIGITAL_TWIN",
            "effective_warmup_minutes": 120,
            "dwf_background_loading": True,
            "baseline_true_state_advantage": ["internal_rtc", "auto_rbc", "efd"],
        },
        "controller": {
            "history_steps": 13,
            "horizon_steps": 72,
            "max_setting_delta_per_update": 0.5,
            "enforce_cross_decision_target_continuity": True,
            "enforce_sequential_horizon_continuity": True,
        },
    }


def test_first_move_cannot_abruptly_reverse_previous_target() -> None:
    decision = choose_first_move(
        optimized_sequence=np.asarray([[0.0]], dtype=float),
        surrogate_admissible=True,
        fallback_first_move=np.asarray([0.5]),
        current_settings=np.asarray([0.5]),
        previous_requested_settings=np.asarray([1.0]),
        max_delta_per_update=0.5,
    )
    assert decision.requested.tolist() == pytest.approx([0.5])


def test_progressive_reversal_is_allowed_over_multiple_decisions() -> None:
    first = choose_first_move(
        optimized_sequence=np.asarray([[0.0]], dtype=float),
        surrogate_admissible=True,
        fallback_first_move=np.asarray([0.5]),
        current_settings=np.asarray([0.5]),
        previous_requested_settings=np.asarray([1.0]),
        max_delta_per_update=0.5,
    )
    second = choose_first_move(
        optimized_sequence=np.asarray([[0.0]], dtype=float),
        surrogate_admissible=True,
        fallback_first_move=np.asarray([0.0]),
        current_settings=np.asarray([0.5]),
        previous_requested_settings=first.requested,
        max_delta_per_update=0.5,
    )
    assert first.requested.tolist() == pytest.approx([0.5])
    assert second.requested.tolist() == pytest.approx([0.0])


def test_command_continuity_reports_previous_target_violation() -> None:
    result = command_continuity(
        np.asarray([0.0]),
        np.asarray([0.5]),
        previous_requested_settings=np.asarray([1.0]),
        max_delta_per_update=0.5,
    )
    assert result.passed is False
    assert result.failed_current_indices == ()
    assert result.failed_previous_indices == (0,)


def test_full_horizon_is_projected_as_one_sequential_path() -> None:
    blocks = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]], dtype=torch.float32)
    _project_block_settings_(
        blocks,
        current_settings=torch.tensor([0.0]),
        max_delta_per_update=0.5,
    )
    assert blocks[0, :, 0].tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0])
    delta = torch.diff(torch.cat([torch.tensor([0.0]), blocks[0, :, 0]]))
    assert float(torch.abs(delta).max()) <= 0.5 + 1e-8


class _ConstantController:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        return ControllerAction(
            settings={aid: self.value for aid in obs.actuator_ids},
            source="CONSTANT",
        )


def test_diagnostic_extreme_ramps_instead_of_jumping() -> None:
    guard = ContinuityGuardController(
        _ConstantController(1.0), max_delta_per_update=0.5, allow_projection=True
    )
    first = guard(_observation(0.0))
    second = guard(_observation(0.5, target=0.5))
    assert first.settings["A1"] == pytest.approx(0.5)
    assert second.settings["A1"] == pytest.approx(1.0)
    assert first.diagnostics is not None
    assert first.diagnostics["continuity_projection_applied"] is True


def test_proposed_outer_guard_refuses_illegal_jump() -> None:
    guard = ContinuityGuardController(
        _ConstantController(1.0), max_delta_per_update=0.5, allow_projection=False
    )
    with pytest.raises(RuntimeError, match="outside its declared temporal-continuity contract"):
        guard(_observation(0.0))


def test_project7_runtime_contract_freezes_60_120_360() -> None:
    evidence = validate_project7_runtime_config(_project7_config())
    assert evidence["first_proposed_control_minutes"] == 60
    assert evidence["effective_warmup_minutes"] == 120
    assert evidence["prediction_horizon_minutes"] == 360
    timing = evidence["timing"]
    assert isinstance(timing, dict)
    assert timing["history_span_seconds"] == 3600
    assert timing["horizon_seconds"] == 21600
    assert timing["d3_control_blocks"] == 36


def test_project7_contract_rejects_old_short_horizon() -> None:
    config = _project7_config()
    controller = dict(config["controller"])
    controller["horizon_steps"] = 24
    config["controller"] = controller
    with pytest.raises(ValueError, match="frozen runtime timing mismatch"):
        validate_project7_runtime_config(config)


def test_timing_freeze_preserves_censor_but_keeps_preregistered_h360(tmp_path: Path) -> None:
    summary = tmp_path / "phase0.json"
    summary.write_text(
        json.dumps(
            {
                "contract": "PHASE0_D2_STEP_RESPONSE_TIMESCALE_V5_HIGH_FREQUENCY",
                "horizon_censored": True,
                "candidate_production_timing": {
                    "model_observation_seconds": 300,
                    "control_update_seconds": 600,
                },
            }
        ),
        encoding="utf-8",
    )
    result = freeze_phase0_timing(
        phase0_summary_path=summary,
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_minutes=360,
        control_start_minutes=60,
        max_setting_delta_per_update=0.5,
    )
    assert result["phase0_horizon_censored"] is True
    assert result["phase0_censor_role"] == "diagnostic_not_horizon_selection_gate"
    assert result["controller"]["horizon_steps"] == 72
