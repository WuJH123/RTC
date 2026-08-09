from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rtc.d3_runner_guard import validate_d3_execution_contract


def _manifest(path: Path) -> Path:
    sequence = [{"A1": 0.5}] * 12
    pd.DataFrame(
        [
            {
                "settings_sequence_json": json.dumps(sequence),
                "trajectory_metadata_path": "baseline.json",
                "model_horizon_steps": 24,
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "control_block_steps": 2,
                "control_blocks": 12,
                "d3_time_contract": "D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1",
                "d3_feasibility_contract": "D3_SEQUENTIAL_SETTING_RATE_FEASIBILITY_V1",
                "sequence_rate_feasible": True,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_d3_runner_guard_accepts_matching_model_and_control_clocks(tmp_path: Path) -> None:
    evidence = validate_d3_execution_contract(
        _manifest(tmp_path / "d3.csv"), control_block_seconds=600, stride_seconds=300
    )
    assert evidence["model_horizon_steps"] == 24
    assert evidence["control_blocks"] == 12
    assert evidence["control_block_steps"] == 2


def test_d3_runner_guard_rejects_runtime_control_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime control block differs"):
        validate_d3_execution_contract(
            _manifest(tmp_path / "d3.csv"), control_block_seconds=900, stride_seconds=300
        )


def test_d3_runner_guard_rejects_runtime_model_step_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime stride"):
        validate_d3_execution_contract(
            _manifest(tmp_path / "d3.csv"), control_block_seconds=600, stride_seconds=60
        )
