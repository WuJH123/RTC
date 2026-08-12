from __future__ import annotations

import torch

from rtc.step2_control_response_v433 import (
    DifferentiableCounterfactualResponseModelV433,
    nodewise_residual_parameter_names_v433,
    set_trainable_nodewise_residual_v433,
    trapezoid_delta_flood_volume_per_node_v433,
)

from test_step2_state_topology_interaction_v43 import _fixture


def _v433_fixture(*, horizon: int = 8, candidates: int = 1):
    base, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(
        horizon=horizon, candidates=candidates
    )
    model = DifferentiableCounterfactualResponseModelV433(
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
        topology_blocks=base.topology_blocks,
    )
    model.load_state_dict(base.state_dict(), strict=False)
    return model, prepared, initial, rainfall, reference, candidate, previous, elapsed


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


def test_constant_one_m3s_integrates_to_expected_m3():
    rate_a = torch.ones(1, 1, 2, 3)
    elapsed_a = torch.tensor([[0.0, 300.0, 600.0]])
    rate_b = torch.ones(1, 1, 4, 3)
    elapsed_b = torch.tensor([[0.0, 150.0, 300.0, 450.0, 600.0]])
    volume_a = trapezoid_delta_flood_volume_per_node_v433(rate_a, elapsed_a)
    volume_b = trapezoid_delta_flood_volume_per_node_v433(rate_b, elapsed_b)
    torch.testing.assert_close(volume_a, torch.full((1, 1, 3), 600.0))
    torch.testing.assert_close(volume_a, volume_b)


def test_nodewise_volume_integrates_time_not_nodes():
    rate = torch.ones(1, 1, 2, 3)
    elapsed = torch.tensor([[0.0, 300.0, 600.0]])
    volume = trapezoid_delta_flood_volume_per_node_v433(rate, elapsed)
    assert volume.shape == (1, 1, 3)
    torch.testing.assert_close(volume, torch.full((1, 1, 3), 600.0))


def test_zero_latent_multi_action_residual_exact_zero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v433_fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    reference_encoding = model.encode_reference(
        initial, rainfall, reference, previous, prepared
    )
    zero_latent = torch.zeros(
        1, 1, candidate.shape[2], prepared.static_node_features.shape[0], model.effect_rank
    )
    zero_rate = torch.zeros(
        1, 1, candidate.shape[2], prepared.static_node_features.shape[0]
    )
    residual = model._nodewise_local_residual_delta_tfv(
        zero_latent,
        zero_rate,
        reference_encoding,
        elapsed,
        torch.ones(1, 1),
    )
    assert torch.equal(residual, torch.zeros_like(residual))


def test_nodewise_bias_cannot_accumulate_over_nodes():
    model, *_ = _v433_fixture()
    assert all(
        torch.equal(parameter, torch.zeros_like(parameter))
        for parameter in model.nodewise_residual_correction[-1].parameters()
    )


def test_zero_initialized_residual_matches_old_global_head():
    base, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    model, *_ = _v433_fixture()
    old = base.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    corrected = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    torch.testing.assert_close(
        corrected.direct_interaction_delta_tfv_m3,
        old.direct_interaction_delta_tfv_m3,
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        corrected.direct_delta_tfv_m3,
        old.direct_delta_tfv_m3,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.equal(
        corrected.nodewise_local_residual_delta_tfv_m3,
        torch.zeros_like(corrected.nodewise_local_residual_delta_tfv_m3),
    )


def test_nodewise_is_residual_not_replacement():
    base, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    model, *_ = _v433_fixture()
    with torch.no_grad():
        model.nodewise_residual_correction[-1].bias.fill_(0.1)
    old = base.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    corrected = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    torch.testing.assert_close(
        corrected.direct_interaction_delta_tfv_m3,
        old.direct_interaction_delta_tfv_m3 + corrected.nodewise_local_residual_delta_tfv_m3,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.any(corrected.nodewise_local_residual_delta_tfv_m3 != 0.0)


def test_zero_action_and_single_action_residual_zero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v433_fixture()
    zero = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.equal(
        zero.nodewise_local_residual_delta_tfv_m3,
        torch.zeros_like(zero.nodewise_local_residual_delta_tfv_m3),
    )
    candidate[:, 0, 2:, 0] = (candidate[:, 0, 2:, 0] + 0.2).clamp(0.0, 1.0)
    single = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.equal(
        single.nodewise_local_residual_delta_tfv_m3,
        torch.zeros_like(single.nodewise_local_residual_delta_tfv_m3),
    )


def test_nodewise_future_action_causality_and_h72_finite():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _v433_fixture(
        horizon=72
    )
    candidate[:, 0, 36:, :2] = (candidate[:, 0, 36:, :2] + 0.2).clamp(0.0, 1.0)
    candidate.requires_grad_(True)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert torch.isfinite(output.direct_delta_tfv_m3).all()
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0
    assert torch.equal(
        output.interaction_delta_states_physical[:, :, :36],
        torch.zeros_like(output.interaction_delta_states_physical[:, :, :36]),
    )


def test_nodewise_residual_parameter_set_is_explicit():
    model, *_ = _v433_fixture()
    names = nodewise_residual_parameter_names_v433(model)
    assert names
    set_trainable_nodewise_residual_v433(model, enabled=True)
    assert all(
        parameter.requires_grad == (name in set(names))
        for name, parameter in model.named_parameters()
    )
