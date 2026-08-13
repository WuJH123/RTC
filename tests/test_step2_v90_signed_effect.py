from __future__ import annotations

import torch

from test_step2_v80 import _graph, _inputs, _model
from rtc.step2_control_response_v80 import prepare_static_v80
from rtc.step2_control_response_v90 import (
    DirectHydraulicEffectSurrogateV90,
    HydraulicEffectOutputV90,
    project_candidate_states_v90,
    project_candidate_flows_v90,
)
from rtc.step2_v90_contract import DirectHydraulicEffectLossContractV90, LEVEL_C


def _output(raw_state: torch.Tensor, raw_flow: torch.Tensor) -> HydraulicEffectOutputV90:
    reference_state = torch.zeros_like(raw_state)
    reference_flow = torch.zeros_like(raw_flow)
    candidate_state = project_candidate_states_v90(
        reference_state,
        raw_state,
        invert_elevation_m=torch.zeros(raw_state.shape[-2]),
    )
    return HydraulicEffectOutputV90(
        horizon_indices=torch.empty(0, dtype=torch.long),
        raw_delta_states_physical=raw_state,
        raw_delta_flows_physical=raw_flow,
        reference_states_physical=reference_state,
        reference_flows_physical=reference_flow,
        candidate_states_projected_physical=candidate_state,
        candidate_flows_projected_physical=project_candidate_flows_v90(reference_flow, raw_flow),
        reference_flood_onset_logits=torch.empty(0),
        candidate_flood_onset_logits=torch.empty(0),
        joint_context_before_scatter=torch.empty(0),
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
    reference_flow = torch.tensor([[[[0.50]]]])
    reference_state = torch.zeros(1, 1, 1, 1, 6)
    candidate_state = project_candidate_states_v90(
        reference_state, torch.zeros_like(reference_state), invert_elevation_m=torch.zeros(1)
    )
    output = HydraulicEffectOutputV90(
        horizon_indices=torch.empty(0, dtype=torch.long),
        raw_delta_states_physical=torch.zeros_like(reference_state),
        raw_delta_flows_physical=raw_flow,
        reference_states_physical=reference_state,
        reference_flows_physical=reference_flow,
        candidate_states_projected_physical=candidate_state,
        candidate_flows_projected_physical=project_candidate_flows_v90(reference_flow, raw_flow),
        reference_flood_onset_logits=torch.empty(0),
        candidate_flood_onset_logits=torch.empty(0),
        joint_context_before_scatter=torch.empty(0),
    )
    assert output.raw_delta_flows_physical.item() == -0.75
    assert output.candidate_flows_projected_physical.item() == -0.25


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
    model = DirectHydraulicEffectSurrogateV90(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
    )
    prepared = prepare_static_v80(graph)
    initial, rain, reference, candidate, previous_flow = _inputs(2)
    output = model(initial, rain, reference, candidate[:, 1:2], previous_flow, prepared)
    assert output.raw_delta_states_physical.shape == output.candidate_states_projected_physical.shape
    assert output.raw_delta_flows_physical.shape == output.candidate_flows_projected_physical.shape


def test_v90_predicted_reference_context_changes_joint_effect_token():
    graph, basis, v80 = _model()
    model = DirectHydraulicEffectSurrogateV90(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        conditioning_level=LEVEL_C,
        contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
    )
    prepared = prepare_static_v80(graph)
    initial, rain, reference, candidate, previous_flow = _inputs(2)
    oracle_states = torch.zeros(1, 72, 14, 6)
    zero = model(
        initial, rain, reference, candidate[:, 1:2], previous_flow, prepared,
        oracle_reference_states_physical=oracle_states,
        oracle_reference_flows_physical=torch.zeros(1, 72, 109),
    )
    oracle_states[..., 0] = 0.5
    oracle = model(
        initial, rain, reference, candidate[:, 1:2], previous_flow, prepared,
        oracle_reference_states_physical=oracle_states,
        oracle_reference_flows_physical=torch.zeros(1, 72, 109),
    )
    assert not torch.equal(zero.joint_context_before_scatter, oracle.joint_context_before_scatter)


def test_v90_actuator_identity_is_explicitly_present_in_effect_encoder_input():
    """Identical physical context must remain distinguishable by facility ID.

    The frozen graph has physical-feature and endpoint collisions between real
    actuators.  Hydraulic effect tokens therefore need an explicit 109-way
    identity feature before endpoint scatter, rather than relying on accidental
    differences in physics or topology.
    """
    graph, basis, v80 = _model()
    model = DirectHydraulicEffectSurrogateV90(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
    )
    assert model.actuator_identity_embedding.num_embeddings == 109
    assert model.actuator_identity_embedding.embedding_dim > 0

    captured = []
    hook = model.actuator_effect_encoder[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        prepared = prepare_static_v80(graph)
        initial, rain, reference, candidate, previous_flow = _inputs(2)
        previous_flow.zero_()
        # Give two physically identical pump tokens exactly the same action.
        candidate[:, 1, :, 0] = reference[:, :, 0] + 0.2
        candidate[:, 1, :, 1] = reference[:, :, 1] + 0.2
        model(initial, rain, reference, candidate[:, 1:2], previous_flow, prepared)
    finally:
        hook.remove()
    # First encoder call is the non-zero action branch.  Its identity slice must
    # differ for actuator 0 and 1 even when their imposed action is identical.
    encoded = captured[0]
    identity_width = model.actuator_identity_embedding.embedding_dim
    effect_width = model.prefix.feature_count + 1
    identity = encoded[..., -(effect_width + identity_width):-effect_width]
    assert not torch.equal(identity[..., 0, :], identity[..., 1, :])


def test_v90_fixed_multiscale_diffusion_reaches_eight_hops_without_new_parameters():
    """V9 must not make every >4-hop effect structurally zero.

    The frozen H360 D2 audit found most signed response mass beyond four hops.
    The pre-registered V9 repair reuses the fixed zero-preserving blocks through
    hops 1/2/4/8 and fuses those scales; it is not a graph-depth sweep.
    """
    graph, basis, v80 = _model()
    model = DirectHydraulicEffectSurrogateV90(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
    )
    # Make a deterministic positive diffusion for this semantic test.
    with torch.no_grad():
        for block in model.graph_blocks:
            for layer in (block.self_projection, block.neighbour_projection, block.output_projection):
                layer.weight.copy_(torch.eye(16))
        model.multiscale_gate.weight.zero_()
        model.multiscale_gate.bias.zero_()
    seed = torch.zeros(1, 1, 1, 14, 16)
    seed[..., 0, 0] = 1.0
    fused, weights, scales = model._multiscale_diffuse_v90(seed, prepare_static_v80(graph))
    assert tuple(scales) == (1, 2, 4, 8)
    assert torch.count_nonzero(fused[..., 8, :]) > 0
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    zeros = torch.zeros_like(seed)
    zero_fused, _, _ = model._multiscale_diffuse_v90(zeros, prepare_static_v80(graph))
    assert torch.equal(zero_fused, zeros)
