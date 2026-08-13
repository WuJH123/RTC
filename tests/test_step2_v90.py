from __future__ import annotations

import torch

from rtc.step2_control_response_v90 import (
    project_candidate_flows_v90,
    project_candidate_states_v90,
)


def _reference_states() -> torch.Tensor:
    # [B,C,J,N,S] = depth, head, flooding, volume, inflow, outflow
    reference = torch.zeros(1, 1, 1, 2, 6)
    reference[..., 0, 0] = 0.02
    reference[..., 0, 1] = 10.02
    reference[..., 0, 2] = 0.01
    reference[..., 0, 3] = 0.50
    reference[..., 0, 4] = 0.02
    reference[..., 0, 5] = 0.03
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
    assert raw_delta[..., 0, 2].item() == -0.25


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
    assert raw_delta[..., 0, 0].item() == -0.20
    assert raw_delta[..., 0, 1].item() == -0.20


def test_v90_signed_flow_semantics_are_not_projection_semantics():
    reference = torch.tensor([[[[0.02, 0.30]]]])
    raw_delta = torch.tensor([[[[-0.25, -0.10]]]])
    projected = project_candidate_flows_v90(reference, raw_delta)
    assert torch.equal(projected, torch.tensor([[[[0.0, 0.20]]]]))
    assert torch.equal(raw_delta, torch.tensor([[[[-0.25, -0.10]]]]))


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
