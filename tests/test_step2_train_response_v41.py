from __future__ import annotations

import numpy as np
import pytest
import torch

from rtc.step2_train_response_v4 import ResponsePairV4
from rtc.step2_train_response_v41 import (
    ScaleDeltaBlockV41,
    derive_counterfactual_delta_scales_v41,
    group_metrics_v41,
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
