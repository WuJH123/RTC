from __future__ import annotations

import json

import pandas as pd
import pytest

from rtc.causal_timing import CausalTimingContract
from rtc.d3_design_cli import design_d3_manifest
from rtc.inp import Actuator, ActuatorCatalog


def _timing(*, horizon_steps: int = 24) -> CausalTimingContract:
    return CausalTimingContract(
        model_step_seconds=300,
        control_update_seconds=600,
        history_steps=13,
        horizon_steps=horizon_steps,
        control_start_minutes=60,
        record_stride_seconds=300,
    )


def _catalog() -> ActuatorCatalog:
    return ActuatorCatalog(
        (
            Actuator("P1", "pump", "N1", "N2"),
            Actuator("O1", "orifice", "N2", "N3"),
        )
    )


def _checkpoint() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checkpoint_id": "E1__t0060",
                "event_id": "E1",
                "rainfall_group": "R1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_minutes": 60,
                "inp_path": "event.inp",
                "setting:P1": 0.0,
                "setting:O1": 0.5,
            }
        ]
    )


def test_d3_uses_control_blocks_not_model_horizon_steps() -> None:
    timing = _timing(horizon_steps=24)
    assert timing.control_block_steps == 2
    assert timing.d3_control_blocks == 12

    manifest = design_d3_manifest(
        checkpoints=_checkpoint(),
        catalog=_catalog(),
        timing=timing,
        sequences_per_checkpoint=2,
        perturbation_std=0.1,
        change_probability=0.5,
        seed=7,
    )
    assert set(manifest["model_horizon_steps"].astype(int)) == {24}
    assert set(manifest["control_block_steps"].astype(int)) == {2}
    assert set(manifest["control_blocks"].astype(int)) == {12}
    assert set(manifest["model_step_seconds"].astype(int)) == {300}
    assert set(manifest["control_update_seconds"].astype(int)) == {600}
    assert set(manifest["d3_time_contract"].astype(str)) == {
        "D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1"
    }
    assert all(
        len(json.loads(raw)) == 12 for raw in manifest["settings_sequence_json"].astype(str)
    )


def test_d3_rejects_partial_control_block_horizon() -> None:
    timing = _timing(horizon_steps=25)
    with pytest.raises(ValueError, match="integer number of control blocks"):
        _ = timing.d3_control_blocks
