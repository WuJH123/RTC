"""TDD regression tests for the V5.0 candidate-manifold surrogate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_identifiability_v50 import (
    block_action_matrix_v50,
    effective_rank_v50,
    mpc_candidate_manifold_contract_v50,
)
from rtc.step2_control_response_v50 import StateConditionedCandidateManifoldModelV50
from rtc.step2_train_response_v50 import listwise_pl_loss_v50


def _tiny_model() -> tuple[StateConditionedCandidateManifoldModelV50, SimpleNamespace, object]:
    torch.manual_seed(7)
    nodes, actuators, state_dim, rain_dim, hidden = 5, 109, 6, 1, 16
    graph = SimpleNamespace(
        static_node_features=np.zeros((nodes, 26), dtype=np.float32),
        static_node_feature_names=("invert_elevation_m",) + tuple(f"f{i}" for i in range(25)),
        actuator_physics=np.zeros((actuators, 19), dtype=np.float32),
        actuator_upstream=np.arange(actuators, dtype=np.int64) % nodes,
        actuator_downstream=(np.arange(actuators, dtype=np.int64) + 1) % nodes,
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
    )
    model = StateConditionedCandidateManifoldModelV50(
        state_dim=state_dim,
        rainfall_dim=rain_dim,
        node_static_dim=26,
        actuator_physics_dim=19,
        hidden_dim=hidden,
        actuator_count=actuators,
        temporal_embedding_dim=8,
        max_horizon_steps=72,
        control_blocks=36,
        d2_tfv_scale=100.0,
        d3_tfv_scale=100.0,
    )
    prepared = model.prepare_static(
        static_node_features=torch.zeros(nodes, 26),
        actuator_physics=torch.zeros(actuators, 19),
        actuator_upstream=torch.as_tensor(graph.actuator_upstream),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream),
        edge_index=torch.as_tensor(graph.edge_index),
        invert_elevation_m=torch.zeros(nodes),
    )
    return model, graph, prepared


def test_block_action_matrix_v50_collapses_72_steps_to_36_control_blocks() -> None:
    values = torch.arange(72 * 3, dtype=torch.float32).reshape(72, 3)
    blocks = block_action_matrix_v50(values, control_blocks=36)
    assert blocks.shape == (36, 3)
    assert torch.equal(blocks[0], values[:2].mean(dim=0))
    assert torch.equal(blocks[-1], values[-2:].mean(dim=0))


def test_effective_rank_v50_reports_energy_dimensions() -> None:
    matrix = np.zeros((8, 5), dtype=np.float64)
    matrix[:, 0] = np.arange(8)
    matrix[:, 1] = np.arange(8)[::-1]
    summary = effective_rank_v50(matrix)
    assert summary["rank"] >= 1
    assert 1 <= summary["rank_90"] <= summary["rank_95"] <= summary["rank_99"] <= 5


def test_mpc_candidate_manifold_contract_is_derived_from_frozen_runtime() -> None:
    contract = mpc_candidate_manifold_contract_v50()
    assert contract["actuator_count"] == 109
    assert contract["control_blocks"] == 36
    assert contract["model_horizon_steps"] == 72
    assert contract["max_setting_delta_per_update"] == 0.5
    assert contract["candidate_family"] == "continuous_all_actuator_projected_gradient"


def test_v50_candidate_equals_reference_exact_zero() -> None:
    model, _, prepared = _tiny_model()
    batch, horizon, nodes, actuators = 1, 72, 5, 109
    initial = torch.zeros(batch, nodes, 6)
    rainfall = torch.zeros(batch, horizon, nodes, 1)
    reference = torch.full((batch, horizon, actuators), 0.4)
    candidate = reference[:, None].clone()
    previous_flow = torch.zeros(batch, actuators)
    elapsed = torch.arange(horizon + 1, dtype=torch.float32).reshape(1, -1) * 300.0
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous_flow, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(output.direct_delta_tfv_m3, torch.zeros_like(output.direct_delta_tfv_m3))
    assert torch.equal(output.delta_states_physical, torch.zeros_like(output.delta_states_physical))
    assert torch.equal(output.delta_flows_physical, torch.zeros_like(output.delta_flows_physical))


def test_v50_state_conditioning_changes_joint_representation_before_scatter() -> None:
    model, _, prepared = _tiny_model()
    batch, horizon, nodes, actuators = 1, 72, 5, 109
    initial = torch.zeros(batch, nodes, 6)
    rainfall = torch.zeros(batch, horizon, nodes, 1)
    reference = torch.full((batch, horizon, actuators), 0.4)
    candidate = reference[:, None].clone()
    candidate[:, 0, 4:, 0] = 0.8
    previous_flow = torch.zeros(batch, actuators)
    elapsed = torch.arange(horizon + 1, dtype=torch.float32).reshape(1, -1) * 300.0
    first = model.forward_group(
        initial, rainfall, reference, candidate, previous_flow, prepared, elapsed, source_kind="D3"
    )
    changed = initial.clone()
    changed[:, 0, 0] = 2.0
    second = model.forward_group(
        changed, rainfall, reference, candidate, previous_flow, prepared, elapsed, source_kind="D3"
    )
    assert not torch.allclose(first.joint_context_before_node_scatter, second.joint_context_before_node_scatter)


def test_v50_future_action_cannot_change_past_response() -> None:
    model, _, prepared = _tiny_model()
    batch, horizon, nodes, actuators = 1, 72, 5, 109
    initial = torch.zeros(batch, nodes, 6)
    rainfall = torch.zeros(batch, horizon, nodes, 1)
    reference = torch.full((batch, horizon, actuators), 0.4)
    first = reference[:, None].clone()
    second = first.clone()
    first[:, 0, 0, 0] = 0.8
    second[:, 0, 0, 0] = 0.8
    second[:, 0, 20:, 1] = 0.1
    previous_flow = torch.zeros(batch, actuators)
    elapsed = torch.arange(horizon + 1, dtype=torch.float32).reshape(1, -1) * 300.0
    a = model.forward_group(initial, rainfall, reference, first, previous_flow, prepared, elapsed, source_kind="D3")
    b = model.forward_group(initial, rainfall, reference, second, previous_flow, prepared, elapsed, source_kind="D3")
    assert torch.allclose(a.direct_delta_tfv_prefix_m3[..., :20], b.direct_delta_tfv_prefix_m3[..., :20], atol=1e-6, rtol=1e-6)


def test_v50_listwise_value_loss_is_finite_and_differentiable() -> None:
    predicted = torch.tensor([[2.0, 0.5, 1.0]], requires_grad=True)
    truth = torch.tensor([[20.0, 5.0, 10.0]])
    loss = listwise_pl_loss_v50(predicted, truth)
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert torch.count_nonzero(predicted.grad) > 0


def test_v50_value_and_hydraulic_heads_are_parameter_disjoint() -> None:
    model, _, _ = _tiny_model()
    value_names = {name for name, _ in model.control_value.named_parameters()}
    hydraulic_names = {name for name, _ in model.hydraulic_response.named_parameters()}
    assert value_names
    assert hydraulic_names
    assert value_names.isdisjoint(hydraulic_names)
    model.set_hydraulic_trainable(True)
    assert all(not parameter.requires_grad for parameter in model.control_value.parameters())
    assert all(parameter.requires_grad for parameter in model.hydraulic_response.parameters())
