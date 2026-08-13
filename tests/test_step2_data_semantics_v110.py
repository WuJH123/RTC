from __future__ import annotations

import numpy as np
import torch

from rtc.step2_control_response_v60 import PreparedStaticV60
from rtc.step2_data_semantics_v110 import (
    CandidateMechanismRecordV110,
    action_sequence_descriptors_v110,
    reference_phase_descriptors_v110,
    summarize_mechanism_records_v110,
)


def test_action_sequence_descriptors_capture_exposure_timing_and_reversal() -> None:
    reference = np.zeros((6, 3), dtype=np.float32)
    candidate = np.zeros((2, 6, 3), dtype=np.float32)
    candidate[0, 1:4, 0] = 0.5
    candidate[1, 0:2, 1] = 0.4
    candidate[1, 2:5, 1] = -0.4
    candidate[1, 3:, 2] = 0.2

    out = action_sequence_descriptors_v110(reference, candidate, model_step_seconds=300)

    assert out["changed_actuator_count"].tolist() == [1, 2]
    assert out["first_change_min"].tolist() == [10.0, 5.0]
    assert out["last_change_min"].tolist() == [20.0, 30.0]
    assert out["active_duration_min"].tolist() == [15.0, 30.0]
    assert out["peak_simultaneous_changed_actuators"].tolist() == [1, 2]
    assert out["action_reversal_count"].tolist() == [0, 1]
    assert float(out["action_l1_exposure"][1]) > float(out["action_l1_exposure"][0])


def _prepared() -> PreparedStaticV60:
    return PreparedStaticV60(
        node_static=torch.zeros(3, 2),
        actuator_physics=torch.zeros(1, 2),
        actuator_upstream=torch.tensor([0]),
        actuator_downstream=torch.tensor([1]),
        invert_elevation_m=torch.zeros(3),
        max_depth_m=torch.tensor([2.0, 4.0, 1.0]),
        surcharge_depth_m=torch.zeros(3),
        storage_capacity_m3=torch.tensor([0.0, 100.0, 50.0]),
        storage_mask=torch.tensor([False, True, True]),
        actuator_feature_names=("x", "y"),
    )


def test_reference_phase_descriptors_are_current_state_only() -> None:
    state = np.zeros((3, 6), dtype=np.float32)
    state[:, 0] = [1.0, 3.8, 0.2]
    state[:, 2] = [0.0, 0.01, 0.0]
    state[:, 3] = [0.0, 80.0, 50.0]
    state[:, 4] = [2.0, 3.0, 4.0]
    state[:, 5] = [1.0, 1.0, 5.0]

    result = reference_phase_descriptors_v110(state, _prepared())

    assert 0.0 < result["reference_depth_fill_mean"] < 1.0
    assert result["reference_near_surcharge_fraction"] == 1.0 / 3.0
    assert result["reference_flood_active_fraction"] == 1.0 / 3.0
    assert result["reference_storage_fill_mean"] == 0.9
    assert result["reference_storage_near_capacity_fraction"] == 0.5
    assert result["reference_net_inflow_m3s_mean"] == 2.0 / 3.0


def test_mechanism_summary_marks_future_truth_as_forbidden_online_input() -> None:
    record = CandidateMechanismRecordV110(
        source_kind="D2",
        group_name="D2::g",
        event_id="e",
        rainfall_group="r",
        checkpoint_id="c",
        candidate_index=1,
        changed_actuator_count=1,
        action_l1_exposure=1.0,
        action_l2_exposure=1.0,
        max_abs_setting_delta=0.5,
        first_change_min=5.0,
        last_change_min=10.0,
        active_duration_min=10.0,
        action_transition_count=1,
        action_reversal_count=0,
        peak_simultaneous_changed_actuators=1,
        reference_depth_fill_mean=0.2,
        reference_depth_fill_p90=0.5,
        reference_near_surcharge_fraction=0.0,
        reference_flood_active_fraction=0.0,
        reference_storage_fill_mean=0.2,
        reference_storage_near_capacity_fraction=0.0,
        reference_net_inflow_m3s_mean=0.0,
        hydraulic_effect_energy=2.0,
        onset_min=20.0,
        peak_response_min=40.0,
        remote_effect_fraction_gt8=0.3,
        delta_tfv_m3=-100.0,
    )
    summary = summarize_mechanism_records_v110([record])
    forbidden = summary["training_use"]["forbidden_online_inputs"]
    assert any("future SWMM" in item for item in forbidden)
    assert summary["source_counts"] == {"D2": 1}
