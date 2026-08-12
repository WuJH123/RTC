from __future__ import annotations

import torch
from torch import nn

from rtc.step2_control_response_v45 import (
    DifferentiableCounterfactualResponseModelV45,
    direct_effect_parameter_names_v45,
)
from test_step2_state_topology_interaction_v43 import _fixture


def _v45_fixture(*, actuators: int = 6, horizon: int = 8, candidates: int = 2):
    base, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(
        actuators=actuators, horizon=horizon, candidates=candidates
    )
    model = DifferentiableCounterfactualResponseModelV45(
        state_dim=base.state_dim,
        rainfall_dim=1,
        node_static_dim=prepared.static_node_features.shape[1],
        actuator_physics_dim=prepared.actuator_physics.shape[1],
        hidden_dim=base.hidden_dim,
        actuator_count=base.actuator_count,
        actuator_embedding_dim=base.actuator_identity.embedding_dim,
        temporal_embedding_dim=base.temporal_identity.embedding_dim,
        state_mean=base.state_mean,
        state_std=base.state_std,
        flow_std=base.flow_std,
        d2_state_scale=base.d2_state_scale,
        d3_state_scale=base.d3_state_scale,
        d2_flow_scale=base.d2_flow_scale,
        d3_flow_scale=base.d3_flow_scale,
        d2_tfv_scale=float(base.d2_tfv_scale),
        d3_tfv_scale=float(base.d3_tfv_scale),
        max_horizon_steps=base.max_horizon_steps,
        effect_rank=base.effect_rank,
    )
    return model, prepared, initial, rainfall, reference, candidate, previous, elapsed


def _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed, source="D3"):
    return model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind=source
    )


def test_v45_candidate_equals_reference_exact_zero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.equal(output.delta_states_physical, torch.zeros_like(output.delta_states_physical))
    assert torch.equal(output.delta_flows_physical, torch.zeros_like(output.delta_flows_physical))
    assert torch.equal(output.direct_delta_tfv_m3, torch.zeros_like(output.direct_delta_tfv_m3))
    assert torch.equal(output.trajectory_delta_tfv_m3, torch.zeros_like(output.trajectory_delta_tfv_m3))


def test_v45_single_and_multi_action_are_nonzero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1)
    candidate[:, 0, :, 0] += 0.2
    single = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.any(single.direct_delta_tfv_m3 != 0.0)
    candidate[:, 0, :, 1] += 0.2
    multi = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.any(multi.direct_delta_tfv_m3 != 0.0)


def test_v45_direct_tfv_has_horizon_node_normalization():
    """The direct TFV head must not accumulate one full scale per H×N cell."""
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(
        candidates=1, horizon=72
    )
    class _FirstChannel(nn.Module):
        def forward(self, value):
            return 100.0 * value[..., :1]

    # Make the diagnostic deterministic and expose the aggregation contract,
    # rather than relying on a random head's initial output magnitude.
    model.global_tfv_head_v45 = _FirstChannel()
    model.d3_tfv_scale.fill_(1.0)
    candidate[:, 0, :, :2] += 0.2
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.isfinite(output.direct_delta_tfv_m3).all()
    # Two bounded candidate/reference evaluations may differ by at most
    # sixteen normalized TFV scales; this catches accidental H×N integration.
    assert float(output.direct_delta_tfv_m3.abs().max().detach()) <= 16.0 + 1e-4


def test_v45_joint_context_changes_when_second_actuator_changes():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1)
    candidate[:, 0, :, 0] += 0.2
    first = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    candidate[:, 0, :, 1] += 0.2
    second = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert not torch.equal(first.direct_delta_tfv_m3, second.direct_delta_tfv_m3)


def test_v45_output_is_not_sum_of_independent_single_outputs():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1)
    candidate_a = candidate.clone(); candidate_a[:, 0, :, 0] += 0.2
    candidate_b = candidate.clone(); candidate_b[:, 0, :, 1] += 0.2
    candidate_ab = candidate.clone(); candidate_ab[:, 0, :, :2] += 0.2
    delta_a = _forward(model, prepared, initial, rainfall, reference, candidate_a, previous, elapsed).direct_delta_tfv_m3
    delta_b = _forward(model, prepared, initial, rainfall, reference, candidate_b, previous, elapsed).direct_delta_tfv_m3
    delta_ab = _forward(model, prepared, initial, rainfall, reference, candidate_ab, previous, elapsed).direct_delta_tfv_m3
    assert not torch.allclose(delta_ab, delta_a + delta_b, atol=1e-7, rtol=1e-6)


def test_v45_actuator_identity_and_shared_effect_encoder():
    model, *_ = _v45_fixture()
    assert model.actuator_count == 6
    assert hasattr(model, "effect_encoder_v45")
    assert not hasattr(model, "candidate_effect_encoder_v45")
    assert not hasattr(model, "reference_effect_encoder_v45")
    assert direct_effect_parameter_names_v45(model)


def test_v45_109_actuators_supported():
    model, *_ = _v45_fixture(actuators=109, candidates=1)
    assert model.actuator_count == 109


def test_v45_future_action_cannot_affect_past():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1, horizon=8)
    candidate[:, 0, 1, 0] += 0.2
    early = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    future = candidate.clone(); future[:, 0, 6:, 1] += 0.3
    late = _forward(model, prepared, initial, rainfall, reference, future, previous, elapsed)
    torch.testing.assert_close(early.delta_states_physical[:, :, :6], late.delta_states_physical[:, :, :6], atol=1e-7, rtol=1e-6)


def test_v45_72_step_horizon_forward_backward_finite():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1, horizon=72)
    candidate[:, 0, :, :2] += 0.2
    candidate.requires_grad_(True)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0]
    assert torch.isfinite(objective)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_v45_nonnegative_flooding_and_head_depth_consistency():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v45_fixture(candidates=1)
    candidate[:, 0, :, :2] += 0.2
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.all(output.candidate_states_physical[..., 2] >= 0.0)
    expected_head = output.candidate_states_physical[..., 0] + prepared.invert_elevation_m.reshape(1, 1, 1, -1)
    torch.testing.assert_close(output.candidate_states_physical[..., 1], expected_head)
