from __future__ import annotations

import pytest

from rtc.checkpoint_design import _assert_replayable_no_control_prefix


def test_d2_checkpoint_requires_controls_disabled_no_write_prefix() -> None:
    _assert_replayable_no_control_prefix(
        {
            "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
            "python_actuator_writes": False,
            "native_controls_enabled": False,
        },
        "no_control.json",
    )


@pytest.mark.parametrize(
    "meta,match",
    [
        (
            {"data_contract": "CLOSED_LOOP_COMPACT_V2", "controller_present": True},
            "Do not use D1 controlled states",
        ),
        (
            {
                "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
                "python_actuator_writes": False,
                "native_controls_enabled": True,
            },
            "Internal-RTC controls enabled",
        ),
        (
            {
                "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
                "python_actuator_writes": True,
                "native_controls_enabled": False,
            },
            "Python actuator writes",
        ),
    ],
)
def test_d2_checkpoint_rejects_unreplayable_prefix(meta, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _assert_replayable_no_control_prefix(meta, "bad.json")
