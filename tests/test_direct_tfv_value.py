from __future__ import annotations

import torch

from rtc.step2_tfv_value import DirectFacilityTFVValueModel
from rtc.step2_tfv_value_training import _pairwise_ranking_loss


def _inputs(batch: int = 2):
    torch.manual_seed(7)
    nodes, state_dim, rain_dim, physics_dim, actuators = 8, 3, 1, 5, 109
    state = torch.randn(batch, nodes, state_dim)
    rainfall = torch.randn(batch, 72, nodes, rain_dim)
    reference = torch.full((batch, 72, actuators), 0.5)
    previous_flow = torch.randn(batch, actuators)
    up = torch.arange(actuators) % nodes
    down = (up + 1) % nodes
    physics = torch.randn(actuators, physics_dim)
    model = DirectFacilityTFVValueModel(
        state_dim=state_dim,
        rainfall_dim=rain_dim,
        actuator_physics_dim=physics_dim,
        target_scale_m3=5000.0,
    )
    return model, state, rainfall, reference, previous_flow, up, down, physics


def _forward(model, state, rainfall, reference, candidate, previous_flow, up, down, physics):
    return model(
        current_state=state,
        rainfall=rainfall,
        reference_settings=reference,
        candidate_settings=candidate,
        previous_actuator_flow=previous_flow,
        actuator_upstream=up,
        actuator_downstream=down,
        actuator_physics=physics,
    )


def test_zero_action_has_exact_zero_delta_tfv() -> None:
    values = _inputs()
    model, state, rainfall, reference, previous_flow, up, down, physics = values
    output = _forward(
        model, state, rainfall, reference, reference.clone(), previous_flow, up, down, physics
    )
    assert torch.equal(output.facility_main_effect_m3, torch.zeros_like(output.facility_main_effect_m3))
    assert torch.equal(output.interaction_residual_m3, torch.zeros_like(output.interaction_residual_m3))
    assert torch.equal(output.total_delta_tfv_m3, torch.zeros_like(output.total_delta_tfv_m3))


def test_single_actuator_has_zero_interaction_and_only_one_main_effect() -> None:
    values = _inputs(batch=1)
    model, state, rainfall, reference, previous_flow, up, down, physics = values
    candidate = reference.clone()
    candidate[:, :, 17] += 0.15
    output = _forward(
        model, state, rainfall, reference, candidate, previous_flow, up, down, physics
    )
    assert torch.equal(output.interaction_residual_m3, torch.zeros_like(output.interaction_residual_m3))
    nonzero = torch.nonzero(torch.abs(output.facility_main_effect_m3[0]) > 0.0, as_tuple=False).reshape(-1)
    assert nonzero.tolist() == [17]
    torch.testing.assert_close(output.total_delta_tfv_m3, output.facility_main_effect_m3.sum(dim=-1))


def test_zero_action_value_has_finite_nonzero_action_gradient() -> None:
    values = _inputs(batch=1)
    model, state, rainfall, reference, previous_flow, up, down, physics = values
    candidate = reference.clone().requires_grad_(True)
    output = _forward(
        model, state, rainfall, reference, candidate, previous_flow, up, down, physics
    )
    gradient = torch.autograd.grad(output.total_delta_tfv_m3.sum(), candidate)[0]
    assert bool(torch.isfinite(gradient).all())
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_temporal_action_pattern_is_visible_to_value_model() -> None:
    values = _inputs(batch=1)
    model, state, rainfall, reference, previous_flow, up, down, physics = values
    early = reference.clone()
    late = reference.clone()
    early[:, :12, 5] += 0.2
    late[:, -12:, 5] += 0.2
    first = _forward(model, state, rainfall, reference, early, previous_flow, up, down, physics)
    second = _forward(model, state, rainfall, reference, late, previous_flow, up, down, physics)
    assert not torch.equal(first.facility_main_effect_m3[:, 5], second.facility_main_effect_m3[:, 5])


def test_pairwise_loss_rewards_correct_tfv_order() -> None:
    truth = torch.tensor([-1000.0, 0.0, 2000.0])
    correct = torch.tensor([-900.0, 10.0, 1800.0])
    reversed_prediction = torch.flip(correct, dims=[0])
    scale = torch.tensor(1000.0)
    assert _pairwise_ranking_loss(correct, truth, scale_m3=scale) < _pairwise_ranking_loss(
        reversed_prediction, truth, scale_m3=scale
    )
