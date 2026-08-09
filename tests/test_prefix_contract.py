from __future__ import annotations

import json
from pathlib import Path

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


def test_cached_no_control_baseline_is_replayable(tmp_path: Path) -> None:
    inp = tmp_path / "no_control.inp"
    inp.write_text("[JUNCTIONS]\nN1 0 1 0 0 0\n[CONTROLS]\n", encoding="utf-8")
    decisions = tmp_path / "run.decisions.jsonl"
    decisions.write_text("", encoding="utf-8")
    meta = {
        "data_contract": "CLOSED_LOOP_COMPACT_V2",
        "controller_present": False,
        "inp_path": str(inp),
        "decision_file": decisions.name,
    }
    metadata_path = tmp_path / "run.json"
    metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    _assert_replayable_no_control_prefix(meta, metadata_path)


@pytest.mark.parametrize(
    "meta,match",
    [
        (
            {"data_contract": "CLOSED_LOOP_COMPACT_V2", "controller_present": True},
            "Python.*decisions",
        ),
        (
            {
                "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
                "python_actuator_writes": False,
                "native_controls_enabled": True,
            },
            "native controls enabled",
        ),
        (
            {
                "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
                "python_actuator_writes": True,
                "native_controls_enabled": False,
            },
            "Python writes",
        ),
    ],
)
def test_d2_checkpoint_rejects_unreplayable_prefix(meta, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _assert_replayable_no_control_prefix(meta, "bad.json")
