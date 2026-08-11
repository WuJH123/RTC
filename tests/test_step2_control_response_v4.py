from __future__ import annotations

import numpy as np
import pytest
import torch

from rtc.step2_control_response_v4 import (
    DifferentiableCounterfactualResponseModelV4,
    direct_pair_delta_tfv,
    expand_control_blocks,
)


def _fixture(*, batch: int = 2, horizon: int = 8, nodes: int = 6, actuators: int = 5):
    torch.manual_seed(7)
    upstream = torch.arange(actuators) % nodes
    downstream = (upstream + 1) % nodes
    edge_index = torch.stack([torch.arange(nodes - 1), torch.arange(1, nodes)])
    model = DifferentiableCounterfactualResponseModelV4(
        state_dim=6,
        rainfall_dim=1,
        node_static_dim=3,
        actuator_physics_dim=4,
        hidden_dim=24,
        actuator_count=actuators,
        actuator_embedding_dim=8,
        state_effect_scale=torch.ones(6),
    )
    initial = torch.randn(batch, nodes, 6)
    rainfall = torch.randn(batch, horizon, nodes, 1)
    reference = torch.sigmoid(torch.randn(batch, horizon, actuators))
    candidate = reference.clone()
    candidate[:, 2:, 0] = (candidate[:, 2:, 0] + 0.2).clamp(0, 1)
    previous_flow = torch.randn(batch, actuators)
    static = torch.randn(nodes, 3)
    physics = torch.randn(actuators, 4)
    return model, initial, rainfall, reference, candidate, previous_flow, static, physics, upstream, downstream, edge_index


def test_zero_action_has_exact_zero_effect_and_zero_tfv_delta():
    args = _fixture()
    model, initial, rainfall, reference, _candidate, previous, static, physics, up, down, edges = args
    output = model(
        initial, rainfall, reference, reference, previous, static, physics, up, down, edges
    )
    assert torch.equal(output.delta_flows, torch.zeros_like(output.delta_flows))
    assert torch.equal(output.delta_states, torch.zeros_like(output.delta_states))
    delta = direct_pair_delta_tfv(output.candidate_flood_rate, output.reference_flood_rate, dt_seconds=300.0)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_future_action_cannot_change_earlier_response():
    args = _fixture()
    model, initial, rainfall, reference, candidate, previous, static, physics, up, down, edges = args
    output = model(
        initial, rainfall, reference, candidate, previous, static, physics, up, down, edges
    )
    assert torch.equal(output.delta_states[:, :2], torch.zeros_like(output.delta_states[:, :2]))
    assert torch.equal(output.delta_flows[:, :2], torch.zeros_like(output.delta_flows[:, :2]))
    assert torch.isfinite(output.delta_states).all()


def test_candidate_permutation_does_not_change_reference_prediction():
    args = _fixture(batch=1)
    model, initial, rainfall, reference, candidate, previous, static, physics, up, down, edges = args
    out_a = model(initial, rainfall, reference, candidate, previous, static, physics, up, down, edges)
    permuted = candidate.flip(dims=[-1])
    out_b = model(initial, rainfall, reference, permuted, previous, static, physics, up, down, edges)
    assert torch.equal(out_a.reference_states, out_b.reference_states)
    assert torch.equal(out_a.reference_flows, out_b.reference_flows)


def test_five_and_ten_actuator_action_pathways_are_differentiable():
    for actuator_count in (5, 10):
        args = _fixture(actuators=actuator_count)
        model, initial, rainfall, reference, candidate, previous, static, physics, up, down, edges = args
        candidate = candidate.clone().requires_grad_(True)
        out = model(initial, rainfall, reference, candidate, previous, static, physics, up, down, edges)
        loss = out.delta_states.square().mean() + out.delta_flows.square().mean()
        gradient = torch.autograd.grad(loss, candidate)[0]
        assert gradient.shape == candidate.shape
        assert torch.isfinite(gradient).all()


def test_control_blocks_expand_to_two_model_steps_without_reordering():
    block = torch.arange(2 * 36 * 3, dtype=torch.float32).reshape(2, 36, 3)
    expanded = expand_control_blocks(block, model_steps_per_block=2)
    assert expanded.shape == (2, 72, 3)
    assert torch.equal(expanded[:, 0::2], block)
    assert torch.equal(expanded[:, 1::2], block)


def test_direct_pair_delta_tfv_uses_candidate_minus_reference_before_reduction():
    reference = torch.tensor([[[[-1.0], [2.0]], [[3.0], [-4.0]]]])
    candidate = torch.tensor([[[[1.0], [4.0]], [[5.0], [-2.0]]]])
    delta = direct_pair_delta_tfv(candidate, reference, dt_seconds=300.0)
    assert delta.shape == (1,)
    assert torch.isfinite(delta).all()
    assert delta.item() == pytest.approx(1500.0)
