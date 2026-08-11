from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtc.step2_control_response_v4 import (
    CounterfactualResponseV4,
    DifferentiableCounterfactualResponseModelV4,
)
from rtc.step2_response_calibration_audit_v41 import (
    cumulative_trapezoid_pair_delta_tfv,
    current_v4_loss_components,
    gradient_cosine,
    head_depth_consistency,
    magnitude_statistics,
    parameter_group_parameter_counts,
    reference_forward_accounting,
)


def test_magnitude_statistics_reports_robust_scale_and_zero_fraction():
    stats = magnitude_statistics(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert stats["rms"] == pytest.approx(math.sqrt(2.0))
    assert stats["median_abs"] == pytest.approx(1.0)
    assert stats["iqr_abs"] == pytest.approx(1.0)
    assert stats["p90_abs"] == pytest.approx(2.0)
    assert stats["p95_abs"] == pytest.approx(2.0)
    assert stats["p99_abs"] == pytest.approx(2.0)
    assert stats["max_abs"] == pytest.approx(2.0)
    assert stats["zero_fraction"] == pytest.approx(0.2)


def test_cumulative_trapezoid_pair_delta_tfv_includes_same_prefix_initial_rate():
    reference_initial = np.asarray([[0.0]])
    candidate_initial = np.asarray([[0.0]])
    reference_future = np.asarray([[[0.0], [0.0]]])
    candidate_future = np.asarray([[[1.0], [3.0]]])
    elapsed = np.asarray([[0.0, 2.0, 4.0]])
    cumulative = cumulative_trapezoid_pair_delta_tfv(
        candidate_initial,
        reference_initial,
        candidate_future,
        reference_future,
        elapsed,
    )
    assert cumulative.shape == (1, 2)
    assert cumulative[0].tolist() == pytest.approx([1.0, 5.0])


def test_gradient_cosine_is_explicit_for_orthogonal_and_zero_vectors():
    assert gradient_cosine(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == pytest.approx(0.0)
    assert gradient_cosine(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])) == pytest.approx(1.0)
    assert math.isnan(gradient_cosine(np.zeros(2), np.ones(2)))


def test_reference_forward_accounting_exposes_pairwise_duplication():
    row = reference_forward_accounting(candidate_count=24)
    assert row == {
        "candidate_count": 24,
        "current_reference_forward_rows": 48,
        "deduplicated_reference_forward_rows": 1,
        "reference_forward_reduction": 48.0,
    }


def test_head_depth_consistency_uses_graph_invert_elevation():
    states = np.zeros((2, 3, 2, 6), dtype=np.float32)
    invert = np.asarray([10.0, 20.0], dtype=np.float32)
    states[..., 0] = np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]])
    states[..., 1] = states[..., 0] + invert
    result = head_depth_consistency(states, invert)
    assert result["max_abs_residual_m"] == pytest.approx(0.0)
    assert result["p99_abs_residual_m"] == pytest.approx(0.0)
    assert result["within_1e_5_fraction"] == pytest.approx(1.0)


def test_parameter_groups_expose_missing_v4_tfv_head_and_flooding_rows():
    model = DifferentiableCounterfactualResponseModelV4(
        state_dim=6,
        rainfall_dim=1,
        node_static_dim=3,
        actuator_physics_dim=4,
        hidden_dim=8,
        actuator_count=5,
        actuator_embedding_dim=3,
        state_effect_scale=torch.ones(6),
    )
    counts = parameter_group_parameter_counts(model)
    assert set(counts) == {
        "reference_encoder",
        "actuator_encoder",
        "action_effect_encoder",
        "trajectory_effect_head",
        "tfv_head",
        "flooding_head",
    }
    assert counts["tfv_head"] == 0
    assert counts["flooding_head"] > 0


def test_current_v4_loss_audit_exposes_rate_surrogate_vs_authoritative_tfv():
    shape = (2, 2, 1, 6)
    reference_states = torch.zeros(shape)
    candidate_states = torch.zeros(shape)
    candidate_states[1, :, 0, 2] = torch.tensor([0.5, 1.0])
    zero_flow = torch.zeros((2, 2, 1))
    output = CounterfactualResponseV4(
        reference_states=reference_states,
        candidate_states=candidate_states,
        delta_states=candidate_states - reference_states,
        reference_flows=zero_flow,
        candidate_flows=zero_flow,
        delta_flows=zero_flow,
        reference_flood_rate=reference_states[..., 2],
        candidate_flood_rate=candidate_states[..., 2],
    )
    target = torch.zeros(shape)
    target[1, :, 0, 2] = torch.tensor([1.0, 3.0])
    batch = {
        "target_states": target,
        "target_actuator_flows": zero_flow,
        "target_states_physical": target,
        "elapsed_seconds": torch.tensor([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]]),
        "exact_node_flood_volume_m3": torch.tensor([[0.0], [100.0]]),
    }
    norm = SimpleNamespace(state_mean=np.zeros(6), state_std=np.ones(6))
    components, diagnostic = current_v4_loss_components(output, batch, norm)
    assert set(components) >= {
        "absolute_state",
        "delta_state",
        "delta_tfv_rate_rectangle",
        "ranking_sign",
        "authoritative_exact_tfv_diagnostic",
        "effect_energy_diagnostic",
    }
    assert diagnostic["true_rate_rectangle_delta_tfv_m3"].item() == pytest.approx(8.0)
    assert diagnostic["true_authoritative_delta_tfv_m3"].item() == pytest.approx(100.0)
