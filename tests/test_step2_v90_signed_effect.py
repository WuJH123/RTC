from __future__ import annotations

import torch

from test_step2_v80 import _graph, _inputs, _model
from rtc.step2_control_response_v80 import DirectHydraulicEffectSurrogateV80
from rtc.step2_control_response_v80 import prepare_static_v80
from rtc.step2_control_response_v90 import (
    HydraulicEffectOutputV90,
    TrajectoryConditionedHydraulicEffectSurrogateV90,
    project_candidate_states_v90,
)


def _output(raw_state: torch.Tensor, raw_flow: torch.Tensor) -> HydraulicEffectOutputV90:
    reference_state = torch.zeros_like(raw_state)
    reference_flow = torch.zeros_like(raw_flow)
    candidate_state = project_candidate_states_v90(
        reference_state,
        raw_state,
        invert_elevation_m=torch.zeros(raw_state.shape[-2]),
    )
    return HydraulicEffectOutputV90(
        raw_delta_states_physical=raw_state,
        raw_delta_flows_physical=raw_flow,
        reference_states_physical=reference_state,
        reference_flows_physical=reference_flow,
        candidate_states_projected_physical=candidate_state,
        candidate_flows_projected_physical=reference_flow + raw_flow,
    )


def test_negative_flood_effect_is_not_clipped_in_primary_signed_output():
    raw = torch.zeros(1, 1, 1, 1, 6)
    raw[..., 2] = -0.25
    output = _output(raw, torch.zeros(1, 1, 1, 1))
    assert output.raw_delta_states_physical[..., 2].item() == -0.25
    assert output.candidate_states_projected_physical[..., 2].item() == 0.0


def test_negative_depth_effect_is_not_rewritten_by_projection():
    raw = torch.zeros(1, 1, 1, 1, 6)
    raw[..., 0] = -0.25
    output = _output(raw, torch.zeros(1, 1, 1, 1))
    assert output.raw_delta_states_physical[..., 0].item() == -0.25
    assert output.candidate_states_projected_physical[..., 0].item() == 0.0


def test_managed_flow_signed_effect_is_separate_from_candidate_projection():
    raw_flow = torch.tensor([[[[-0.75]]]])
    output = _output(torch.zeros(1, 1, 1, 1, 6), raw_flow)
    assert output.raw_delta_flows_physical.item() == -0.75
    assert output.candidate_flows_projected_physical.item() == -0.75


def test_candidate_reference_exact_zero_has_zero_raw_effect():
    raw_state = torch.zeros(1, 1, 1, 1, 6)
    raw_flow = torch.zeros(1, 1, 1, 1)
    output = _output(raw_state, raw_flow)
    assert torch.equal(output.raw_delta_states_physical, torch.zeros_like(raw_state))
    assert torch.equal(output.raw_delta_flows_physical, torch.zeros_like(raw_flow))


def test_v90_model_exposes_raw_signed_effect_separately_from_projection():
    graph, basis, v80 = _model()
    # Reuse the tiny V8 fixture's frozen reference and dimensions, but enable
    # the V9 trajectory feature path without changing its capacity contract.
    base = DirectHydraulicEffectSurrogateV80(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        contract=v80.contract,
        trajectory_conditioning=True,
    )
    model = TrajectoryConditionedHydraulicEffectSurrogateV90(base)
    prepared = prepare_static_v80(graph)
    initial, rain, reference, candidate, previous_flow = _inputs(2)
    output = model(initial, rain, reference, candidate[:, 1:2], previous_flow, prepared)
    assert output.raw_delta_states_physical.shape == output.candidate_states_projected_physical.shape
    assert output.raw_delta_flows_physical.shape == output.candidate_flows_projected_physical.shape


def test_v90_predicted_reference_context_changes_joint_effect_token():
    graph, basis, v80 = _model()
    base = DirectHydraulicEffectSurrogateV80(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        contract=v80.contract,
        trajectory_conditioning=True,
    )
    model = TrajectoryConditionedHydraulicEffectSurrogateV90(base)
    prepared = prepare_static_v80(graph)
    initial, rain, reference, candidate, previous_flow = _inputs(2)
    predicted = model(initial, rain, reference, candidate[:, 1:2], previous_flow, prepared)
    oracle_states = torch.zeros(1, 72, 14, 6)
    oracle_states[..., 0] = 0.5
    oracle_flows = torch.zeros(1, 72, 109)
    oracle = model(
        initial, rain, reference, candidate[:, 1:2], previous_flow, prepared,
        reference_trajectory_context=(oracle_states, oracle_flows),
    )
    assert not torch.equal(predicted.joint_context_before_scatter, oracle.joint_context_before_scatter)
