from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_tfv_value_ng1 import (
    NG1ProcessAwareDirectTFVValueModel,
    build_control_interaction_graph,
    d2_magnitude_strata,
    d2_magnitude_weights,
)


def _graph() -> SimpleNamespace:
    actuator_count = 109
    return SimpleNamespace(
        actuator_ids=tuple(f"A{i:03d}" for i in range(actuator_count)),
        actuator_upstream=np.arange(actuator_count, dtype=np.int64) % 8,
        actuator_downstream=(np.arange(actuator_count, dtype=np.int64) + 1) % 8,
        actuator_physics=np.column_stack(
            (
                np.linspace(0.0, 1.0, actuator_count),
                np.linspace(1.0, 0.0, actuator_count),
                np.arange(actuator_count) % 4,
            )
        ).astype(np.float32),
        edge_index=np.vstack(
            (
                np.arange(8, dtype=np.int64),
                (np.arange(8, dtype=np.int64) + 1) % 8,
            )
        ),
    )


def _inputs(batch: int = 1):
    torch.manual_seed(11)
    nodes, state_dim, rain_dim, physics_dim, actuators = 8, 3, 1, 3, 109
    state = torch.randn(batch, nodes, state_dim)
    rainfall = torch.rand(batch, 72, nodes, rain_dim)
    reference = torch.full((batch, 72, actuators), 0.5)
    previous_flow = torch.randn(batch, actuators)
    graph = _graph()
    model = NG1ProcessAwareDirectTFVValueModel(
        state_dim=state_dim,
        rainfall_dim=rain_dim,
        actuator_physics_dim=physics_dim,
        target_scale_m3=5000.0,
        interaction_graph=build_control_interaction_graph(graph),
    )
    return (
        model,
        state,
        rainfall,
        reference,
        previous_flow,
        torch.as_tensor(graph.actuator_upstream),
        torch.as_tensor(graph.actuator_downstream),
        torch.as_tensor(graph.actuator_physics),
    )


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


def test_complete_pair_graph_is_deterministic_and_label_free() -> None:
    graph = _graph()
    first = build_control_interaction_graph(graph)
    second = build_control_interaction_graph(graph)
    assert first["pair_indices"].shape == (109 * 108 // 2, 2)
    assert first["pair_features"].shape[0] == 109 * 108 // 2
    assert first["pair_indices"].dtype == np.int64
    assert np.array_equal(first["pair_indices"], second["pair_indices"])
    assert np.array_equal(first["pair_features"], second["pair_features"])
    assert first["sha256"] == second["sha256"]
    assert first["physics_features_standardized"] is True
    assert "truth" not in first
    assert "tfv" not in first


def test_ng1_exact_zero_and_single_facility_interaction_gate() -> None:
    values = _inputs()
    model, state, rainfall, reference, previous_flow, up, down, physics = values
    zero = _forward(
        model, state, rainfall, reference, reference.clone(), previous_flow, up, down, physics
    )
    assert torch.equal(
        zero.interaction_residual_m3, torch.zeros_like(zero.interaction_residual_m3)
    )
    assert torch.equal(zero.total_delta_tfv_m3, torch.zeros_like(zero.total_delta_tfv_m3))

    single = reference.clone()
    single[:, :, 17] += 0.15
    output = _forward(
        model, state, rainfall, reference, single, previous_flow, up, down, physics
    )
    assert torch.equal(
        output.interaction_residual_m3,
        torch.zeros_like(output.interaction_residual_m3),
    )
    assert torch.equal(
        output.total_delta_tfv_m3, output.facility_main_effect_m3.sum(dim=-1)
    )


def test_adaptive_connectivity_keeps_changed_to_context_pairs() -> None:
    values = _inputs()
    model = values[0]
    changed_one = torch.zeros(1, 109, dtype=torch.bool)
    changed_one[:, 5] = True
    mask_one = model._pair_activity_mask(changed_one)
    assert int(mask_one.sum()) == 108

    changed_two = changed_one.clone()
    changed_two[:, 17] = True
    mask_two = model._pair_activity_mask(changed_two)
    assert int(mask_two.sum()) == 215

    changed_three = changed_two.clone()
    changed_three[:, 31] = True
    mask_three = model._pair_activity_mask(changed_three)
    assert int(mask_three.sum()) == 321


def test_ng1_candidate_reference_swap_is_exactly_antisymmetric() -> None:
    values = _inputs()
    model, state, rainfall, reference, previous_flow, up, down, physics = values
    candidate = reference.clone()
    candidate[:, :24, 5] += 0.15
    candidate[:, 20:50, 17] -= 0.10
    forward = _forward(
        model, state, rainfall, reference, candidate, previous_flow, up, down, physics
    )
    reverse = _forward(
        model, state, rainfall, candidate, reference, previous_flow, up, down, physics
    )
    torch.testing.assert_close(
        forward.interaction_residual_m3,
        -reverse.interaction_residual_m3,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        forward.total_delta_tfv_m3,
        -reverse.total_delta_tfv_m3,
        rtol=0.0,
        atol=0.0,
    )


def test_d2_magnitude_partition_is_mutually_exclusive_complete_and_mean_weight_one() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 5.0, 9.0, 12.0, 30.0], dtype=np.float64)
    strata = d2_magnitude_strata(values)
    masks = strata["masks"]
    assert np.all(sum(mask.astype(np.int64) for mask in masks.values()) == 1)
    weights = d2_magnitude_weights(values, strata)
    assert np.isclose(float(weights.mean()), 1.0)
    assert np.isfinite(weights).all()
