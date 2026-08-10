from __future__ import annotations

import json

import pandas as pd

from rtc.data_design import canonical_action_sha
from rtc.d2_runner import _parse_snapshot_horizons
from rtc.phase0_pulse import design_pulse_recovery


def test_snapshot_horizons_must_fit_executed_horizon() -> None:
    assert _parse_snapshot_horizons(
        "210,240,300,360", horizon_minutes=360, stride_seconds=60
    ) == (210, 240, 300, 360)


def test_pulse_design_releases_to_complete_base_action_after_one_block() -> None:
    base = {"P1": 0.5, "O1": 0.2}
    candidate = {"P1": 0.65, "O1": 0.2}
    base_sha = canonical_action_sha(base)
    frame = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "rainfall_group": "G1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "E1:t0060",
                "checkpoint_minutes": 60,
                "inp_path": "event.inp",
                "trajectory_metadata_path": "d0.json",
                "actuator_id": "P1",
                "base_setting": 0.5,
                "requested_setting": 0.5,
                "base_action_sha256": base_sha,
                "candidate_action_sha256": base_sha,
                "candidate_settings_json": json.dumps(base, sort_keys=True),
            },
            {
                "event_id": "E1",
                "rainfall_group": "G1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "E1:t0060",
                "checkpoint_minutes": 60,
                "inp_path": "event.inp",
                "trajectory_metadata_path": "d0.json",
                "actuator_id": "P1",
                "base_setting": 0.5,
                "requested_setting": 0.65,
                "base_action_sha256": base_sha,
                "candidate_action_sha256": canonical_action_sha(candidate),
                "candidate_settings_json": json.dumps(candidate, sort_keys=True),
            },
        ]
    )
    designed = design_pulse_recovery(
        frame,
        model_step_seconds=60,
        control_block_seconds=600,
        horizon_minutes=60,
        pulses_per_checkpoint=1,
    )
    assert len(designed) == 2
    hold = designed[designed["data_role"] == "PHASE0_PULSE_HOLD_REFERENCE"].iloc[0]
    pulse = designed[designed["data_role"] == "PHASE0_PULSE_RELEASE_RECOVERY"].iloc[0]
    hold_sequence = json.loads(str(hold["settings_sequence_json"]))
    pulse_sequence = json.loads(str(pulse["settings_sequence_json"]))
    assert len(hold_sequence) == len(pulse_sequence) == 6
    assert pulse_sequence[0] == candidate
    assert pulse_sequence[1:] == [base] * 5
    assert all(set(step) == {"P1", "O1"} for step in pulse_sequence)
