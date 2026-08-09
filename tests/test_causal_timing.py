from __future__ import annotations

import pytest

from rtc.causal_timing import CausalTimingContract, timing_from_controller_config


def test_5min_observation_10min_control_has_exact_60min_history_at_first_mpc() -> None:
    timing = CausalTimingContract(
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_steps=24,
        control_start_minutes=60,
        record_stride_seconds=300,
    )
    timing.validate()
    assert timing.history_span_seconds == 3600
    assert timing.control_start_seconds == 3600
    assert timing.control_block_steps == 2
    assert timing.horizon_seconds == 7200
    assert [i * 300 for i in range(timing.history_steps)] == list(range(0, 3601, 300))


def test_first_control_cannot_precede_complete_history() -> None:
    timing = CausalTimingContract(
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_steps=12,
        control_start_minutes=50,
        record_stride_seconds=300,
    )
    with pytest.raises(ValueError, match="before a full causal Step1 history"):
        timing.validate()


def test_first_control_must_stay_on_control_clock() -> None:
    timing = CausalTimingContract(
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_steps=12,
        control_start_minutes=65,
        record_stride_seconds=300,
    )
    with pytest.raises(ValueError, match="control-update grid"):
        timing.validate()


def test_config_requires_explicit_history_and_horizon() -> None:
    with pytest.raises(ValueError, match="explicit timing fields"):
        timing_from_controller_config(
            {
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "control_start_minutes": 60,
                "controller": {},
            }
        )
