from __future__ import annotations

import numpy as np
import pytest
import torch

from rtc.step2_train_response_v4 import ResponsePairV4
from rtc.step2_train_response_v41 import (
    ScaleDeltaBlockV41,
    balanced_magnitude_stratum_weights,
    derive_counterfactual_delta_scales_v41,
    group_metrics_v41,
    magnitude_strata_metrics_v41,
    calibration_selection_score_v41,
    source_parameter_is_trainable,
    stack_response_group_v41,
    tfv_loss_components_v41,
    weighted_pairwise_ranking_loss,
)


def test_counterfactual_scales_are_source_specific_and_use_physical_delta_rms():
    d2 = ScaleDeltaBlockV41(
        source_kind="D2",
        delta_states=np.asarray([[[[3.0, 0.0], [4.0, 0.0]]]], dtype=np.float32),
        delta_flows=np.asarray([[[2.0, 0.0]]], dtype=np.float32),
        delta_tfv_m3=np.asarray([10.0], dtype=np.float32),
    )
    d3 = ScaleDeltaBlockV41(
        source_kind="D3",
        delta_states=np.asarray([[[[30.0, 0.0], [40.0, 0.0]]]], dtype=np.float32),
        delta_flows=np.asarray([[[20.0, 0.0]]], dtype=np.float32),
        delta_tfv_m3=np.asarray([100.0], dtype=np.float32),
    )
    scales = derive_counterfactual_delta_scales_v41(
        [d2, d3], source_manifest_sha256="a" * 64
    )
    assert scales.by_source["D2"].state_scale.tolist() == pytest.approx(
        [np.sqrt(12.5), 1e-6]
    )
    assert scales.by_source["D3"].state_scale.tolist() == pytest.approx(
        [np.sqrt(1250.0), 1e-6]
    )
    assert scales.by_source["D2"].tfv_scale_m3 == pytest.approx(10.0)
    assert scales.by_source["D3"].tfv_scale_m3 == pytest.approx(100.0)
    assert scales.by_source["D2"].state_scale[0] != scales.by_source["D3"].state_scale[0]


def test_pairwise_ranking_uses_all_meaningful_weighted_pairs():
    truth = torch.tensor([[0.0, 10.0, 10.000001]])
    correct = torch.tensor([[0.0, 9.0, 9.5]], requires_grad=True)
    reversed_prediction = torch.tensor([[9.0, 0.0, 0.5]])
    correct_loss, details = weighted_pairwise_ranking_loss(
        correct, truth, group_scale=torch.tensor([10.0]), minimum_normalized_gap=1e-4
    )
    reversed_loss, _ = weighted_pairwise_ranking_loss(
        reversed_prediction,
        truth,
        group_scale=torch.tensor([10.0]),
        minimum_normalized_gap=1e-4,
    )
    assert details["meaningful_pair_count"] == 2
    assert correct_loss < reversed_loss
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()
    assert torch.count_nonzero(correct.grad) == 3


def test_tfv_loss_uses_authoritative_exact_target_and_centered_group_term():
    predicted_direct = torch.tensor([[10.0, 30.0]], requires_grad=True)
    predicted_trajectory = torch.tensor([[12.0, 28.0]], requires_grad=True)
    authoritative = torch.tensor([[20.0, 40.0]])
    losses = tfv_loss_components_v41(
        predicted_direct,
        predicted_trajectory,
        authoritative,
        source_scale=torch.tensor(100.0),
    )
    assert set(losses) >= {
        "absolute_direct",
        "group_centered_direct",
        "authoritative_trajectory",
        "direct_trajectory_consistency",
        "ranking",
    }
    assert losses["group_centered_direct"].item() == pytest.approx(0.0)
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(predicted_direct.grad).all()
    assert torch.isfinite(predicted_trajectory.grad).all()


def test_tfv_loss_has_fixed_d3_magnitude_calibration_and_balanced_strata():
    predicted_direct = torch.tensor([[10.0, 20.0, 50.0, 120.0]], requires_grad=True)
    predicted_trajectory = torch.tensor([[12.0, 18.0, 55.0, 110.0]], requires_grad=True)
    authoritative = torch.tensor([[5.0, 25.0, 60.0, 240.0]])
    losses = tfv_loss_components_v41(
        predicted_direct,
        predicted_trajectory,
        authoritative,
        source_scale=torch.tensor(100.0),
        magnitude_calibration=True,
        magnitude_q33=30.0,
        magnitude_q67=100.0,
    )
    assert "log_magnitude_calibration" in losses
    assert torch.isfinite(losses["log_magnitude_calibration"])
    assert losses["magnitude_stratum_weight_mean"].item() == pytest.approx(1.0)
    total = losses["log_magnitude_calibration"] + losses["absolute_direct"]
    total.backward()
    assert torch.isfinite(predicted_direct.grad).all()


def test_balanced_magnitude_weights_equalize_unbalanced_strata_per_group():
    authoritative = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0, 40.0, 100.0]])
    result = balanced_magnitude_stratum_weights(authoritative, q33=20.0, q67=100.0)
    weights = result["weights"]
    assert result["counts"].tolist() == [[8, 2, 1]]
    totals = [weights[result[f"{name}_mask"]].sum().item() for name in ("small", "medium", "large")]
    assert totals[0] == pytest.approx(totals[1])
    assert totals[1] == pytest.approx(totals[2])
    assert weights.mean().item() == pytest.approx(1.0)


def test_balanced_magnitude_weights_are_independent_for_each_batch_group():
    authoritative = torch.tensor(
        [[1.0, 2.0, 30.0, 100.0, 101.0], [0.0, 1.0, 30.0, 31.0, 100.0]]
    )
    result = balanced_magnitude_stratum_weights(authoritative, q33=20.0, q67=100.0)
    assert result["counts"].tolist() == [[2, 1, 2], [2, 2, 1]]
    for batch_index in range(2):
        weights = result["weights"][batch_index]
        totals = [
            weights[result[f"{name}_mask"][batch_index]].sum().item()
            for name in ("small", "medium", "large")
        ]
        assert totals[0] == pytest.approx(totals[1])
        assert totals[1] == pytest.approx(totals[2])
        assert weights.mean().item() == pytest.approx(1.0)


def test_balanced_magnitude_weights_handle_missing_stratum_without_nan():
    authoritative = torch.tensor([[1.0, 2.0, 100.0, 110.0]])
    result = balanced_magnitude_stratum_weights(authoritative, q33=20.0, q67=100.0)
    assert result["counts"].tolist() == [[2, 0, 2]]
    assert torch.isfinite(result["weights"]).all()
    assert result["weights"].mean().item() == pytest.approx(1.0)
    assert result["weights"][result["small_mask"]].sum().item() == pytest.approx(
        result["weights"][result["large_mask"]].sum().item()
    )


def test_balanced_magnitude_weights_use_exact_boundary_partition():
    authoritative = torch.tensor([[29.0, 30.0, 99.0, 100.0]])
    result = balanced_magnitude_stratum_weights(authoritative, q33=30.0, q67=100.0)
    assert result["small_mask"].tolist() == [[True, False, False, False]]
    assert result["medium_mask"].tolist() == [[False, True, True, False]]
    assert result["large_mask"].tolist() == [[False, False, False, True]]


def test_magnitude_reporting_partition_is_exhaustive_and_non_overlapping():
    rows = [
        {"source_kind": "D3", "true_delta_tfv_m3": value, "predicted_final_delta_tfv_m3": value, "group": "g1"}
        for value in (29.0, 30.0, 99.0, 100.0)
    ]
    result = magnitude_strata_metrics_v41(rows, q33=30.0, q67=100.0)
    assert sum(result[name]["count"] for name in ("small", "medium", "large")) == len(rows)
    assert [result[name]["count"] for name in ("small", "medium", "large")] == [1, 2, 1]


def test_balanced_magnitude_loss_has_finite_nonzero_gradient_for_multiple_groups():
    predicted_direct = torch.tensor([[10.0, 20.0, 50.0, 120.0], [5.0, 15.0, 80.0, 130.0]], requires_grad=True)
    predicted_trajectory = torch.tensor([[12.0, 18.0, 55.0, 110.0], [6.0, 13.0, 75.0, 125.0]], requires_grad=True)
    authoritative = torch.tensor([[5.0, 25.0, 60.0, 240.0], [5.0, 30.0, 70.0, 220.0]])
    losses = tfv_loss_components_v41(
        predicted_direct,
        predicted_trajectory,
        authoritative,
        source_scale=torch.tensor(100.0),
        magnitude_calibration=True,
        magnitude_q33=30.0,
        magnitude_q67=100.0,
    )
    total = losses["absolute_direct"] + losses["authoritative_trajectory"] + losses["log_magnitude_calibration"]
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(predicted_direct.grad).all()
    assert torch.count_nonzero(predicted_direct.grad) > 0


def test_magnitude_strata_metrics_are_reported_separately():
    rows = [
        {"source_kind": "D3", "true_delta_tfv_m3": 5.0, "predicted_final_delta_tfv_m3": 4.0, "group": "g1"},
        {"source_kind": "D3", "true_delta_tfv_m3": 50.0, "predicted_final_delta_tfv_m3": 40.0, "group": "g1"},
        {"source_kind": "D3", "true_delta_tfv_m3": 150.0, "predicted_final_delta_tfv_m3": 100.0, "group": "g1"},
    ]
    result = magnitude_strata_metrics_v41(rows, q33=20.0, q67=100.0)
    assert set(result) == {"small", "medium", "large"}
    assert result["small"]["count"] == 1
    assert result["medium"]["count"] == 1
    assert result["large"]["count"] == 1
    assert result["large"]["response_ratio"] == pytest.approx(100.0 / 150.0)


def test_rank_first_selection_prioritizes_group_ranking_over_signed_bias():
    rank_good = {"spread_ratio": 1.5, "rank": 1.0, "pairwise": 1.0, "sign": 0.5, "top1": True, "normalized_mae": 0.8}
    rank_bad = {"spread_ratio": 1.0, "rank": 0.7, "pairwise": 0.7, "sign": 1.0, "top1": True, "normalized_mae": 0.1}
    assert calibration_selection_score_v41(rank_good, policy="rank_first") < calibration_selection_score_v41(rank_bad, policy="rank_first")


def test_d3_magnitude_selection_penalizes_large_effect_compression():
    better_large = {
        "spread_ratio": 1.0,
        "rank": 0.45,
        "pairwise": 0.67,
        "sign": 0.7,
        "top1": False,
        "normalized_mae": 0.4,
        "d3_magnitude_strata": {"large": {"response_ratio": 0.52, "rank": 0.24, "pairwise": 0.61}},
    }
    worse_large = {
        "spread_ratio": 1.0,
        "rank": 0.51,
        "pairwise": 0.69,
        "sign": 0.7,
        "top1": False,
        "normalized_mae": 0.4,
        "d3_magnitude_strata": {"large": {"response_ratio": 0.20, "rank": 0.46, "pairwise": 0.67}},
    }
    assert calibration_selection_score_v41(better_large, policy="d3_magnitude") < calibration_selection_score_v41(worse_large, policy="d3_magnitude")


def _row(value: float) -> dict[str, np.ndarray]:
    return {
        "initial_state": np.full((2, 3), value, dtype=np.float32),
        "rainfall": np.full((4, 2, 1), value, dtype=np.float32),
        "settings": np.full((4, 5), value, dtype=np.float32),
        "previous_actuator_flow": np.full(5, value, dtype=np.float32),
        "target_states_physical": np.full((4, 2, 3), value, dtype=np.float32),
        "target_actuator_flows_physical": np.full((4, 5), value, dtype=np.float32),
        "elapsed_seconds": np.arange(5, dtype=np.float32),
        "exact_node_flood_volume_m3": np.full(2, value, dtype=np.float32),
    }


def test_group_stack_keeps_one_reference_and_all_candidates():
    reference = _row(1.0)
    pairs = [
        ResponsePairV4("D2", "D2::g", reference, _row(2.0)),
        ResponsePairV4("D2", "D2::g", reference, _row(3.0)),
    ]
    group = stack_response_group_v41(pairs, torch.device("cpu"))
    assert group.reference_settings.shape == (1, 4, 5)
    assert group.candidate_settings.shape == (1, 2, 4, 5)
    assert group.true_delta_states_physical.shape == (1, 2, 4, 2, 3)
    assert group.true_delta_tfv_m3.tolist() == [[2.0, 4.0]]


def test_source_parameter_partition_prevents_d3_from_retraining_single_effect():
    assert source_parameter_is_trainable("single_effect_encoder.0.weight", "D2")
    assert not source_parameter_is_trainable("interaction_encoder.0.weight", "D2")
    assert not source_parameter_is_trainable("single_effect_encoder.0.weight", "D3")
    assert source_parameter_is_trainable("interaction_encoder.0.weight", "D3")
    assert source_parameter_is_trainable("reference_encoder.0.weight", "D2")
    assert source_parameter_is_trainable("reference_encoder.0.weight", "D3")


def test_d2_single_branch_parameters_are_not_trainable_during_d3_update():
    for name in (
        "single_effect_encoder.0.weight",
        "single_flow_head.weight",
        "single_state_head.weight",
        "direct_single_tfv_head.0.weight",
    ):
        assert source_parameter_is_trainable(name, "D2")
        assert not source_parameter_is_trainable(name, "D3")


def test_group_metrics_report_rank_pairwise_top1_and_regret():
    rows = group_metrics_v41(
        predicted=np.asarray([2.0, 1.0, 4.0]),
        truth=np.asarray([3.0, 1.0, 5.0]),
        group="D2::g",
        source_kind="D2",
    )
    assert rows["rank"] == pytest.approx(1.0)
    assert rows["pairwise"] == pytest.approx(1.0)
    assert rows["top1"] is True
    assert rows["regret_m3"] == pytest.approx(0.0)
    assert rows["spread_ratio"] == pytest.approx(3.0 / 4.0)


def test_group_metrics_reports_nan_pairwise_when_all_true_effects_tie():
    rows = group_metrics_v41(
        predicted=np.asarray([1.0, 2.0, 3.0]),
        truth=np.asarray([4.0, 4.0, 4.0]),
        group="D2::ties",
        source_kind="D2",
    )
    assert np.isnan(rows["rank"])
    assert np.isnan(rows["pairwise"])
