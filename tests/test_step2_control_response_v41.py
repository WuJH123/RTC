from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41
from rtc.step2_d3_magnitude_pathway_audit_v42 import action_descriptors_v42


def _fixture(*, actuators: int = 20, horizon: int = 8, candidates: int = 3, magnitude: bool = False):
    torch.manual_seed(17)
    nodes = 7
    state_dim = 6
    upstream = torch.arange(actuators) % nodes
    downstream = (upstream + 1) % nodes
    edge_index = torch.stack(
        [
            torch.cat((torch.arange(nodes - 1), torch.arange(1, nodes))),
            torch.cat((torch.arange(1, nodes), torch.arange(nodes - 1))),
        ]
    )
    invert = torch.linspace(10.0, 16.0, nodes)
    static = torch.randn(nodes, 4)
    physics = torch.randn(actuators, 5)
    model = DifferentiableCounterfactualResponseModelV41(
        state_dim=state_dim,
        rainfall_dim=1,
        node_static_dim=static.shape[1],
        actuator_physics_dim=physics.shape[1],
        hidden_dim=24,
        actuator_count=actuators,
        actuator_embedding_dim=6,
        temporal_embedding_dim=5,
        state_mean=torch.zeros(state_dim),
        state_std=torch.ones(state_dim),
        flow_std=torch.ones(actuators),
        d2_state_scale=torch.ones(state_dim),
        d3_state_scale=2.0 * torch.ones(state_dim),
        d2_flow_scale=torch.ones(actuators),
        d3_flow_scale=2.0 * torch.ones(actuators),
        d2_tfv_scale=100.0,
        d3_tfv_scale=1000.0,
        max_horizon_steps=horizon,
        interaction_magnitude_features_enabled=magnitude,
    )
    prepared = model.prepare_static(
        static_node_features=static,
        actuator_physics=physics,
        actuator_upstream=upstream,
        actuator_downstream=downstream,
        edge_index=edge_index,
        invert_elevation_m=invert,
    )
    initial = torch.randn(1, nodes, state_dim)
    rainfall = torch.randn(1, horizon, nodes, 1)
    reference = torch.sigmoid(torch.randn(1, horizon, actuators))
    candidate = reference[:, None].expand(-1, candidates, -1, -1).clone()
    previous_flow = torch.randn(1, actuators)
    elapsed = torch.arange(horizon + 1, dtype=torch.float32).reshape(1, -1) * 300.0
    return model, prepared, initial, rainfall, reference, candidate, previous_flow, elapsed


def test_group_forward_encodes_reference_once_and_zero_action_is_exact_zero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    with patch.object(model, "encode_reference", wraps=model.encode_reference) as encode:
        output = model.forward_group(
            initial,
            rainfall,
            reference,
            candidate,
            previous,
            prepared,
            elapsed,
            source_kind="D2",
        )
    assert encode.call_count == 1
    assert torch.equal(output.single_delta_states_physical, torch.zeros_like(output.single_delta_states_physical))
    assert torch.equal(output.interaction_delta_states_physical, torch.zeros_like(output.interaction_delta_states_physical))
    assert torch.equal(output.delta_states_physical, torch.zeros_like(output.delta_states_physical))
    assert torch.equal(output.delta_flows_physical, torch.zeros_like(output.delta_flows_physical))
    assert torch.equal(output.direct_delta_tfv_m3, torch.zeros_like(output.direct_delta_tfv_m3))
    assert torch.equal(output.trajectory_delta_tfv_m3, torch.zeros_like(output.trajectory_delta_tfv_m3))


def test_interaction_exact_zero_for_zero_action():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=1)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(output.interaction_delta_states_physical, torch.zeros_like(output.interaction_delta_states_physical))
    assert torch.equal(output.interaction_delta_flows_physical, torch.zeros_like(output.interaction_delta_flows_physical))
    assert torch.equal(output.direct_interaction_delta_tfv_m3, torch.zeros_like(output.direct_interaction_delta_tfv_m3))


def test_interaction_magnitude_features_increase_with_action_energy():
    model, *_ = _fixture()
    small = torch.zeros(1, 1, 8, 20)
    large = torch.zeros_like(small)
    small[:, :, :, 0] = 0.05
    large[:, :, :, :5] = 0.25
    small_features = model.interaction_magnitude_features(small)
    large_features = model.interaction_magnitude_features(large)
    assert small_features.shape[-1] == len(model.interaction_magnitude_feature_names)
    assert torch.isfinite(large_features).all()
    assert large_features.abs().sum() > small_features.abs().sum()


def test_magnitude_branch_is_zero_initialized_to_preserve_v41_response():
    model, *_ = _fixture(magnitude=True)
    # The production V4.2 constructor must expose a zero-start residual branch;
    # this characterization test intentionally fails until that contract exists.
    assert hasattr(model, "interaction_magnitude_residual")
    assert torch.equal(model.interaction_magnitude_residual.weight, torch.zeros_like(model.interaction_magnitude_residual.weight))


def test_36_control_blocks_expand_to_72_model_steps():
    delta = torch.zeros(72, 109).numpy()
    delta[::2, :2] = 0.1
    delta[1::2, :2] = 0.1
    descriptors = action_descriptors_v42(delta)
    assert descriptors["model_step_count"] == 72
    assert descriptors["changed_control_blocks"] == 36
    assert len(descriptors["active_actuator_count_per_control_block"]) == 36


def test_scalar_full_train_flow_std_expands_to_all_actuators():
    model, _prepared, *_ = _fixture(actuators=5)
    scalar_model = DifferentiableCounterfactualResponseModelV41(
        state_dim=model.state_dim,
        rainfall_dim=model.rainfall_dim,
        node_static_dim=model.node_static_dim,
        actuator_physics_dim=model.actuator_physics_dim,
        hidden_dim=12,
        actuator_count=5,
        actuator_embedding_dim=4,
        temporal_embedding_dim=3,
        state_mean=torch.zeros(model.state_dim),
        state_std=torch.ones(model.state_dim),
        flow_std=torch.tensor([9.5]),
        d2_state_scale=torch.ones(model.state_dim),
        d3_state_scale=torch.ones(model.state_dim),
        d2_flow_scale=torch.ones(5),
        d3_flow_scale=torch.ones(5),
        d2_tfv_scale=1.0,
        d3_tfv_scale=1.0,
        max_horizon_steps=8,
    )
    assert scalar_model.flow_std.tolist() == pytest.approx([9.5] * 5)


def test_physical_flooding_is_nonnegative_and_head_depth_is_exact():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, 2:, 0] = (candidate[:, 0, 2:, 0] + 0.2).clamp(0, 1)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    assert torch.all(output.reference_states_physical[..., 2] >= 0.0)
    assert torch.all(output.candidate_states_physical[..., 2] >= 0.0)
    invert = prepared.invert_elevation_m.reshape(1, 1, 1, -1)
    assert torch.allclose(
        output.candidate_states_physical[..., 1],
        output.candidate_states_physical[..., 0] + invert,
        atol=1e-6,
        rtol=0.0,
    )


def test_single_active_actuator_has_exact_zero_interaction_and_is_causal():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=1)
    candidate[:, 0, 3:, 2] = (candidate[:, 0, 3:, 2] + 0.2).clamp(0, 1)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(output.interaction_delta_states_physical, torch.zeros_like(output.interaction_delta_states_physical))
    assert torch.equal(output.interaction_delta_flows_physical, torch.zeros_like(output.interaction_delta_flows_physical))
    assert torch.equal(output.delta_states_physical[:, :, :3], torch.zeros_like(output.delta_states_physical[:, :, :3]))
    assert torch.equal(output.delta_flows_physical[:, :, :3], torch.zeros_like(output.delta_flows_physical[:, :, :3]))


def test_single_effect_keeps_d2_physical_calibration_inside_d3():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(
        candidates=1
    )
    candidate[:, 0, 2:, 2] = (candidate[:, 0, 2:, 2] + 0.2).clamp(0, 1)
    d2 = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    d3 = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(d3.single_delta_states_physical, d2.single_delta_states_physical)
    assert torch.equal(d3.single_delta_flows_physical, d2.single_delta_flows_physical)
    assert torch.equal(d3.direct_single_delta_tfv_m3, d2.direct_single_delta_tfv_m3)


def test_multi_actuator_interaction_is_available_without_canceling_single_branch():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=1)
    candidate[:, 0, :, :5] = (candidate[:, 0, :, :5] + 0.15).clamp(0, 1)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.any(output.single_delta_states_physical != 0.0)
    assert torch.any(output.interaction_delta_states_physical != 0.0)
    assert torch.isfinite(output.delta_states_physical).all()


def test_actuator_identity_changes_response():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=2)
    candidate[:, 0, :, 0] = (candidate[:, 0, :, 0] + 0.15).clamp(0, 1)
    candidate[:, 1, :, 1] = (candidate[:, 1, :, 1] + 0.15).clamp(0, 1)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    assert not torch.equal(output.direct_delta_tfv_m3[:, 0], output.direct_delta_tfv_m3[:, 1])


def test_single_actuator_effect_can_reach_nodes_beyond_its_endpoints():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(
        actuators=1, candidates=1
    )
    candidate[:, 0, 1:, 0] = (candidate[:, 0, 1:, 0] + 0.2).clamp(0, 1)
    output = model.forward_group(
        initial,
        rainfall,
        reference,
        candidate,
        previous,
        prepared,
        elapsed,
        source_kind="D2",
    )
    assert output.single_delta_states_physical[..., 2:, :].abs().max() > 1e-5


def test_one_five_ten_twenty_actuator_gradients_are_finite_and_nonzero():
    for count in (1, 5, 10, 20):
        model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(candidates=1)
        candidate[:, 0, :, :count] = (candidate[:, 0, :, :count] + 0.1).clamp(0, 1)
        candidate.requires_grad_(True)
        output = model.forward_group(
            initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
        )
        objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
        gradient = torch.autograd.grad(objective, candidate)[0]
        changed = gradient[:, :, :, :count]
        assert torch.isfinite(gradient).all()
        assert torch.all(changed.abs().sum(dim=(0, 1, 2)) > 0.0)
