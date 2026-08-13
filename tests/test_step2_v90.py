from __future__ import annotations

from types import SimpleNamespace

import torch
import numpy as np

from rtc.step2_control_response_v90 import (
    project_candidate_flows_v90,
    project_candidate_states_v90,
)
from rtc.step2_hydraulic_eval_v90 import (
    _bucket_effect_records,
    _effect_record,
    _horizon_bucket_masks,
    _timing_record,
    _top_overlap,
    decide_state_sufficiency_v90,
)
from rtc.step2_v90_contract import LEVEL_A, LEVEL_B, LEVEL_C


def _reference_states() -> torch.Tensor:
    # [B,C,J,N,S] = depth, head, flooding, volume, inflow, outflow
    reference = torch.zeros(1, 1, 1, 2, 6)
    reference[..., 0, 0] = 0.02
    reference[..., 0, 1] = 10.02
    reference[..., 0, 2] = 0.01
    reference[..., 0, 3] = 0.50
    reference[..., 0, 4] = 0.02
    reference[..., 0, 5] = 0.03
    reference[..., 1, 1] = 20.0
    return reference


def test_v90_negative_flooding_effect_survives_physical_projection():
    reference = _reference_states()
    raw_delta = torch.zeros_like(reference)
    raw_delta[..., 0, 2] = -0.25
    projected = project_candidate_states_v90(
        reference,
        raw_delta,
        invert_elevation_m=torch.tensor([10.0, 20.0]),
    )
    # Absolute candidate flooding is physically projected to zero, but the signed
    # primary effect is NOT recomputed from the projected candidate.
    assert projected[..., 0, 2].item() == 0.0
    assert torch.isclose(raw_delta[..., 0, 2], torch.tensor(-0.25)).item()


def test_v90_negative_depth_effect_survives_physical_projection():
    reference = _reference_states()
    raw_delta = torch.zeros_like(reference)
    raw_delta[..., 0, 0] = -0.20
    raw_delta[..., 0, 1] = -0.20
    projected = project_candidate_states_v90(
        reference,
        raw_delta,
        invert_elevation_m=torch.tensor([10.0, 20.0]),
    )
    assert projected[..., 0, 0].item() == 0.0
    assert projected[..., 0, 1].item() == 10.0
    assert torch.isclose(raw_delta[..., 0, 0], torch.tensor(-0.20)).item()
    assert torch.isclose(raw_delta[..., 0, 1], torch.tensor(-0.20)).item()


def test_v90_signed_flow_semantics_are_not_projection_semantics():
    reference = torch.tensor([[[[0.02, 0.30]]]])
    raw_delta = torch.tensor([[[[-0.25, -0.10]]]])
    projected = project_candidate_flows_v90(reference, raw_delta)
    # Orifice/weir authoritative flows can be negative; V9 must not impose a
    # global non-negative projection on signed managed-flow semantics.
    assert torch.allclose(projected, torch.tensor([[[[-0.23, 0.20]]]]))
    assert torch.allclose(raw_delta, torch.tensor([[[[-0.25, -0.10]]]]))


def test_v90_zero_action_raw_effect_is_exact_zero_after_projection():
    reference = _reference_states()
    raw_delta = torch.zeros_like(reference)
    projected = project_candidate_states_v90(
        reference,
        raw_delta,
        invert_elevation_m=torch.tensor([10.0, 20.0]),
    )
    assert torch.equal(raw_delta, torch.zeros_like(raw_delta))
    assert torch.equal(projected, reference)


def test_v90_objective_binds_primary_loss_to_raw_signed_effect(monkeypatch):
    import rtc.step2_hydraulic_objective_v90 as objective

    raw_state = torch.tensor([-3.0])
    raw_flow = torch.tensor([-4.0])
    output = SimpleNamespace(
        horizon_indices=torch.tensor([0]),
        reference_states_physical=torch.tensor([1.0]),
        raw_delta_states_physical=raw_state,
        candidate_states_projected_physical=torch.tensor([0.0]),
        reference_flows_physical=torch.tensor([1.0]),
        raw_delta_flows_physical=raw_flow,
        candidate_flows_projected_physical=torch.tensor([0.0]),
        reference_flood_onset_logits=torch.tensor([0.0]),
        candidate_flood_onset_logits=torch.tensor([0.0]),
    )
    captured = {}

    def fake_v80(proxy, *args, **kwargs):
        captured["state"] = proxy.delta_states_physical
        captured["flow"] = proxy.delta_flows_physical
        return torch.tensor(0.0), {"loss": 0.0}

    monkeypatch.setattr(objective, "hydraulic_effect_loss_v80", fake_v80)
    objective.hydraulic_effect_loss_v90(
        output,
        None,
        None,
        None,
        onset_positive_weight=1.0,
    )
    assert captured["state"] is raw_state
    assert captured["flow"] is raw_flow


def _ladder(skills):
    keys = (
        "delta_depth_m_skill_vs_zero",
        "delta_flood_m3s_skill_vs_zero",
        "delta_storage_m3_skill_vs_zero",
        "delta_managed_flow_m3s_skill_vs_zero",
    )
    return {"overall": dict(zip(keys, skills, strict=True))}


def test_v90_ladder_selects_predicted_reference_when_b_closes_oracle_gap():
    result = decide_state_sufficiency_v90({
        LEVEL_A: _ladder([0.00, 0.00, 0.00, 0.00]),
        LEVEL_B: _ladder([0.18, 0.16, 0.20, 0.14]),
        LEVEL_C: _ladder([0.20, 0.18, 0.22, 0.16]),
    })
    assert result["decision"] == "PREDICTED_REFERENCE_TRAJECTORY_SUFFICIENT"


def test_v90_ladder_never_calls_oracle_failure_reference_bottleneck():
    result = decide_state_sufficiency_v90({
        LEVEL_A: _ladder([0.00, 0.00, 0.00, 0.00]),
        LEVEL_B: _ladder([0.02, 0.01, 0.03, 0.02]),
        LEVEL_C: _ladder([0.30, 0.25, 0.28, 0.24]),
    })
    assert result["decision"] == "REFERENCE_HYDRAULIC_ACCURACY_PRIMARY_BOTTLENECK"


def test_v90_markov_insufficiency_requires_oracle_at_or_below_zero():
    result = decide_state_sufficiency_v90({
        LEVEL_A: _ladder([-0.10, -0.10, -0.10, -0.10]),
        LEVEL_B: _ladder([-0.08, -0.05, -0.06, -0.07]),
        LEVEL_C: _ladder([-0.01, 0.0, -0.02, -0.03]),
    })
    assert result["decision"] == "MARKOV_INSUFFICIENCY_SUPPORTED"


def test_v90_managed_flow_active_metrics_use_per_actuator_scales():
    """A global median must not relabel heterogeneous actuator effects."""
    truth = np.asarray([[0.30, 15.0]], dtype=np.float64)
    predicted = np.asarray([[-0.30, 15.0]], dtype=np.float64)
    metrics = _effect_record(
        predicted,
        truth,
        scale=np.asarray([1.0, 100.0], dtype=np.float64),
        active_fraction=0.25,
        prefix="flow",
    )
    # Only actuator 0 is active after its own normalization (0.30 / 1.0).
    # A median scale would instead label actuator 1 active and incorrectly report
    # a perfect sign score.
    assert metrics["flow_active_fraction"] == 0.5
    assert metrics["flow_active_sign"] == 0.0


def test_v90_top_overlap_zero_truth_support_is_not_applicable():
    result = _top_overlap(
        np.asarray([0.0, 2.0, 0.0, 1.0]),
        np.zeros(4, dtype=np.float64),
        2,
    )
    assert np.isnan(result)


def test_v90_top_overlap_zero_prediction_on_true_support_is_zero():
    result = _top_overlap(
        np.zeros(4, dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        1,
    )
    assert result == 0.0


def test_v90_top_overlap_uses_meaningful_active_support():
    result = _top_overlap(
        np.asarray([0.0, 4.0, 0.0, 1.0]),
        np.asarray([0.0, 2.0, 0.0, 3.0]),
        2,
    )
    assert result == 1.0


def test_v90_sparse_onset_uses_normalized_spatial_response_not_network_mean():
    truth = np.zeros((2, 100), dtype=np.float64)
    predicted = np.zeros_like(truth)
    truth[0, 0] = 1.0
    truth[1, :] = 0.5
    predicted[0, 0] = 100.0
    predicted[1, :] = 0.6
    metrics = _timing_record(
        predicted,
        truth,
        retained_indices=np.asarray([0, 1], dtype=np.int64),
        scale=1.0,
        active_fraction=0.25,
        prefix="depth",
    )
    # Peak timing intentionally remains its distinct mass-style statistic, while
    # onset must identify the true local time-0 activation.
    assert metrics["depth_peak_effect_timing_error_min"] == 5.0
    assert metrics["depth_response_onset_timing_error_min"] == 0.0


def test_v90_horizon_buckets_are_fixed_and_partition_h360_exactly():
    masks = _horizon_bucket_masks(np.arange(72, dtype=np.int64))
    assert tuple(masks) == ("0_30_min", "30_120_min", "120_360_min")
    assert [int(mask.sum()) for mask in masks.values()] == [6, 18, 48]
    assert np.array_equal(np.logical_or.reduce(tuple(masks.values())), np.ones(72, dtype=bool))


def test_v90_horizon_bucket_effect_metrics_include_primary_sparse_diagnostics():
    values = np.ones((72, 2), dtype=np.float64)
    buckets = _bucket_effect_records(
        values,
        values,
        retained_indices=np.arange(72, dtype=np.int64),
        scale=1.0,
        active_fraction=0.25,
        prefix="depth",
    )
    for metrics in buckets.values():
        assert metrics["depth_skill_vs_zero"] == 1.0
        assert metrics["depth_response_ratio"] == 1.0
        assert metrics["depth_active_sign"] == 1.0
