from __future__ import annotations

import torch

from rtc.step2_control_response_v90 import (
    project_candidate_flows_v90,
    project_candidate_states_v90,
)
from rtc.step2_hydraulic_eval_v90 import decide_state_sufficiency_v90
from rtc.step2_v90_contract import LEVEL_A, LEVEL_B, LEVEL_C


def _reference_states() -> torch.Tensor:
    # [B,C,J,N,S] = depth, head, flooding, volume, inflow, outflow
    reference = torch.zeros(1, 1, 1, 2, 6)
    reference[..., 0, 0] = 0.02
    reference[..., 0, 1] = 10.02
    reference[..., 0, 2] = 0.01
    reference[..., 0, 3] = 0.50
    reference[..., 0, 4] = 0.02
    reference[..., 0, 5] = 0.03
    reference[..., 1, 1] = 20.0
    return reference


def test_v90_negative_flooding_effect_survives_physical_projection():
    reference = _reference_states()
    raw_delta = torch.zeros_like(reference)
    raw_delta[..., 0, 2] = -0.25
    projected = project_candidate_states_v90(
        reference,
        raw_delta,
        invert_elevation_m=torch.tensor([10.0, 20.0]),
    )
    # Absolute candidate flooding is physically projected to zero, but the signed
    # primary effect is NOT recomputed from the projected candidate.
    assert projected[..., 0, 2].item() == 0.0
    assert torch.isclose(raw_delta[..., 0, 2], torch.tensor(-0.25)).item()


def test_v90_negative_depth_effect_survives_physical_projection():
    reference = _reference_states()
    raw_delta = torch.zeros_like(reference)
    raw_delta[..., 0, 0] = -0.20
    raw_delta[..., 0, 1] = -0.20
    projected = project_candidate_states_v90(
        reference,
        raw_delta,
        invert_elevation_m=torch.tensor([10.0, 20.0]),
    )
    assert projected[..., 0, 0].item() == 0.0
    assert projected[..., 0, 1].item() == 10.0
    assert torch.isclose(raw_delta[..., 0, 0], torch.tensor(-0.20)).item()
    assert torch.isclose(raw_delta[..., 0, 1], torch.tensor(-0.20)).item()


def test_v90_signed_flow_semantics_are_not_projection_semantics():
    reference = torch.tensor([[[[0.02, 0.30]]]])
    raw_delta = torch.tensor([[[[-0.25, -0.10]]]])
    projected = project_candidate_flows_v90(reference, raw_delta)
    assert torch.allclose(projected, torch.tensor([[[[0.0, 0.20]]]]))
    assert torch.allclose(raw_delta, torch.tensor([[[[-0.25, -0.10]]]]))


def test_v90_zero_action_raw_effect_is_exact_zero_after_projection():
    reference = _reference_states()
    raw_delta = torch.zeros_like(reference)
    projected = project_candidate_states_v90(
        reference,
        raw_delta,
        invert_elevation_m=torch.tensor([10.0, 20.0]),
    )
    assert torch.equal(raw_delta, torch.zeros_like(raw_delta))
    assert torch.equal(projected, reference)


def _ladder(skills):
    keys = (
        "delta_depth_m_skill_vs_zero",
        "delta_flood_m3s_skill_vs_zero",
        "delta_storage_m3_skill_vs_zero",
        "delta_managed_flow_m3s_skill_vs_zero",
    )
    return {"overall": dict(zip(keys, skills, strict=True))}


def test_v90_ladder_selects_predicted_reference_when_b_closes_oracle_gap():
    result = decide_state_sufficiency_v90({
        LEVEL_A: _ladder([0.00, 0.00, 0.00, 0.00]),
        LEVEL_B: _ladder([0.18, 0.16, 0.20, 0.14]),
        LEVEL_C: _ladder([0.20, 0.18, 0.22, 0.16]),
    })
    assert result["decision"] == "PREDICTED_REFERENCE_TRAJECTORY_SUFFICIENT"


def test_v90_ladder_never_calls_oracle_failure_reference_bottleneck():
    result = decide_state_sufficiency_v90({
        LEVEL_A: _ladder([0.00, 0.00, 0.00, 0.00]),
        LEVEL_B: _ladder([0.02, 0.01, 0.03, 0.02]),
        LEVEL_C: _ladder([0.30, 0.25, 0.28, 0.24]),
    })
    assert result["decision"] == "REFERENCE_HYDRAULIC_ACCURACY_PRIMARY_BOTTLENECK"


def test_v90_markov_insufficiency_requires_oracle_at_or_below_zero():
    result = decide_state_sufficiency_v90({
        LEVEL_A: _ladder([-0.10, -0.10, -0.10, -0.10]),
        LEVEL_B: _ladder([-0.08, -0.05, -0.06, -0.07]),
        LEVEL_C: _ladder([-0.01, 0.0, -0.02, -0.03]),
    })
    assert result["decision"] == "MARKOV_INSUFFICIENCY_SUPPORTED"
