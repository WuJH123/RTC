from __future__ import annotations

import torch

from rtc.step2_control_response_v43 import (
    DifferentiableCounterfactualResponseModelV43,
    interaction_parameter_names,
    reference_parameter_names,
    set_trainable_phase,
    single_parameter_names,
)
from rtc.step2_d3_magnitude_pathway_audit_v42 import action_descriptors_v42


def _fixture(*, actuators: int = 6, horizon: int = 8, candidates: int = 2):
    torch.manual_seed(43)
    nodes = 9
    state_dim = 6
    static = torch.randn(nodes, 4)
    physics = torch.randn(actuators, 5)
    upstream = torch.arange(actuators) % nodes
    downstream = (upstream + 2) % nodes
    edges = torch.stack(
        [
            torch.arange(nodes - 1),
            torch.arange(1, nodes),
        ]
    )
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    model = DifferentiableCounterfactualResponseModelV43(
        state_dim=state_dim,
        rainfall_dim=1,
        node_static_dim=4,
        actuator_physics_dim=5,
        hidden_dim=20,
        actuator_count=actuators,
        actuator_embedding_dim=6,
        temporal_embedding_dim=5,
        state_mean=torch.zeros(state_dim),
        state_std=torch.ones(state_dim),
        flow_std=torch.ones(actuators),
        d2_state_scale=torch.ones(state_dim),
        d3_state_scale=torch.ones(state_dim),
        d2_flow_scale=torch.ones(actuators),
        d3_flow_scale=torch.ones(actuators),
        d2_tfv_scale=100.0,
        d3_tfv_scale=1000.0,
        max_horizon_steps=horizon,
        effect_rank=4,
    )
    prepared = model.prepare_static(
        static_node_features=static,
        actuator_physics=physics,
        actuator_upstream=upstream,
        actuator_downstream=downstream,
        edge_index=edges,
        invert_elevation_m=torch.linspace(10.0, 18.0, nodes),
    )
    initial = torch.randn(1, nodes, state_dim)
    rainfall = torch.randn(1, horizon, nodes, 1)
    reference = torch.sigmoid(torch.randn(1, horizon, actuators))
    candidate = reference[:, None].expand(-1, candidates, -1, -1).clone()
    previous_flow = torch.randn(1, actuators)
    elapsed = torch.arange(horizon + 1, dtype=torch.float32).reshape(1, -1) * 300.0
    return model, prepared, initial, rainfall, reference, candidate, previous_flow, elapsed


def _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed):
    return model.forward_group(
        initial,
        rainfall,
        reference,
        candidate,
        previous,
        prepared,
        elapsed,
        source_kind="D3",
    )


def test_phase_parameter_sets_are_explicit_and_disjoint():
    model, *_ = _fixture()
    reference = set(reference_parameter_names(model))
    single = set(single_parameter_names(model))
    interaction = set(interaction_parameter_names(model))
    assert reference and single and interaction
    assert reference.isdisjoint(single)
    assert reference.isdisjoint(interaction)
    assert single.isdisjoint(interaction)
    assert reference | single | interaction == {name for name, _ in model.named_parameters()}


def test_graph_topology_changes_multi_actuator_interaction():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    baseline = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    altered_edges = torch.tensor(
        [[0, 2, 4, 6, 1, 3, 5, 7], [2, 4, 6, 8, 3, 5, 7, 8]], dtype=torch.long
    )
    altered = model.prepare_static(
        static_node_features=prepared.static_node_features,
        actuator_physics=prepared.actuator_physics,
        actuator_upstream=prepared.actuator_upstream,
        actuator_downstream=prepared.actuator_downstream,
        edge_index=altered_edges,
        invert_elevation_m=prepared.invert_elevation_m,
    )
    changed = _forward(model, altered, initial, rainfall, reference, candidate, previous, elapsed)
    assert not torch.equal(
        baseline.interaction_delta_states_physical,
        changed.interaction_delta_states_physical,
    )


def test_graph_interaction_zero_for_zero_and_single_action():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=1)
    zero = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.equal(zero.interaction_delta_states_physical, torch.zeros_like(zero.interaction_delta_states_physical))
    candidate[:, 0, 2:, 0] = (candidate[:, 0, 2:, 0] + 0.2).clamp(0.0, 1.0)
    single = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.equal(single.interaction_delta_states_physical, torch.zeros_like(single.interaction_delta_states_physical))
    candidate[:, 0, :, 1] = (candidate[:, 0, :, 1] + 0.2).clamp(0.0, 1.0)
    multi = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.any(multi.interaction_delta_states_physical != 0.0)


def test_local_hydraulic_state_changes_interaction_response():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    first = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    wet_initial = initial.clone()
    wet_initial[..., 0] += 3.0
    second = _forward(model, prepared, wet_initial, rainfall, reference, candidate, previous, elapsed)
    assert not torch.equal(first.direct_interaction_delta_tfv_m3, second.direct_interaction_delta_tfv_m3)


def test_d3_updates_do_not_change_d2_prediction():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    d2_before = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    set_trainable_phase(model, "d3")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2
    )
    d3 = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    loss = d3.direct_delta_tfv_m3.square().mean() + d3.trajectory_delta_tfv_m3.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    d2_after = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    assert torch.allclose(d2_before.delta_states_physical, d2_after.delta_states_physical, atol=1e-7, rtol=1e-6)
    assert torch.allclose(d2_before.direct_delta_tfv_m3, d2_after.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)


def test_reference_and_single_are_frozen_in_d3_phase():
    model, *_ = _fixture()
    set_trainable_phase(model, "reference")
    assert all(parameter.requires_grad == (name in set(reference_parameter_names(model))) for name, parameter in model.named_parameters())
    set_trainable_phase(model, "d2")
    assert all(parameter.requires_grad == (name in set(single_parameter_names(model))) for name, parameter in model.named_parameters())
    set_trainable_phase(model, "d3")
    assert all(parameter.requires_grad == (name in set(interaction_parameter_names(model))) for name, parameter in model.named_parameters())


def test_h72_forward_backward_finite():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(horizon=72)
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    candidate.requires_grad_(True)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0]
    assert torch.isfinite(objective)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_future_action_cannot_affect_past_graph_response():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=1)
    late = candidate.clone()
    late[:, 0, 5:, :2] = (late[:, 0, 5:, :2] + 0.2).clamp(0.0, 1.0)
    output = _forward(model, prepared, initial, rainfall, reference, late, previous, elapsed)
    assert torch.equal(
        output.interaction_delta_states_physical[:, :, :5],
        torch.zeros_like(output.interaction_delta_states_physical[:, :, :5]),
    )


def test_109_actuator_identity_preserved():
    model, *_ = _fixture(actuators=109, horizon=4, candidates=1)
    assert model.actuator_identity.num_embeddings == 109
    assert model.actuator_identity.weight.shape[0] == 109


def test_36_control_blocks_expand_to_72_steps():
    delta = torch.zeros(72, 109).numpy()
    delta[::2, :2] = 0.1
    delta[1::2, :2] = 0.1
    descriptors = action_descriptors_v42(delta)
    assert descriptors["model_step_count"] == 72
    assert descriptors["changed_control_blocks"] == 36
