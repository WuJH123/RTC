from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_direct_tfv_policy_return_current.py"


def _module():
    spec = importlib.util.spec_from_file_location("train_policy_return_decision_aligned", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(**updates: float) -> dict[str, float]:
    base = {
        "selected_action_false_beneficial_fraction": 0.10,
        "selected_action_false_reject_fraction": 0.20,
        "within_query_pairwise_rank_accuracy": 0.70,
        "within_query_candidate_top1_accuracy": 0.60,
        "selected_action_mean_regret_m3": 100.0,
        "hold_aware_decision_accuracy": 0.65,
        "event_balanced_sign_accuracy": 0.75,
        "event_balanced_mae_m3": 500.0,
    }
    base.update(updates)
    return base


def test_selection_prioritizes_false_beneficial_before_every_other_metric() -> None:
    module = _module()
    safer = _metrics(
        selected_action_false_beneficial_fraction=0.0,
        selected_action_false_reject_fraction=0.9,
        within_query_pairwise_rank_accuracy=0.1,
        event_balanced_mae_m3=5000.0,
    )
    riskier = _metrics(
        selected_action_false_beneficial_fraction=0.1,
        selected_action_false_reject_fraction=0.0,
        within_query_pairwise_rank_accuracy=1.0,
        event_balanced_mae_m3=1.0,
    )
    assert module._validation_selection_key(safer) < module._validation_selection_key(riskier)


def test_selection_prioritizes_false_reject_before_rank_or_mae() -> None:
    module = _module()
    fewer_rejects = _metrics(
        selected_action_false_reject_fraction=0.0,
        within_query_pairwise_rank_accuracy=0.1,
        event_balanced_mae_m3=5000.0,
    )
    more_rejects = _metrics(
        selected_action_false_reject_fraction=0.1,
        within_query_pairwise_rank_accuracy=1.0,
        event_balanced_mae_m3=1.0,
    )
    assert module._validation_selection_key(fewer_rejects) < module._validation_selection_key(more_rejects)


def test_query_loss_prefers_correct_hold_aware_action_ordering() -> None:
    module = _module()
    scale = torch.tensor(10.0)
    mixed_truth = torch.tensor([-20.0, 10.0])
    good_mixed = module._query_loss(torch.tensor([-18.0, 8.0]), mixed_truth, scale=scale)
    bad_mixed = module._query_loss(torch.tensor([8.0, -18.0]), mixed_truth, scale=scale)
    assert float(good_mixed) < float(bad_mixed)

    harmful_truth = torch.tensor([10.0, 20.0])
    good_hold = module._query_loss(torch.tensor([8.0, 18.0]), harmful_truth, scale=scale)
    unsafe_execute = module._query_loss(torch.tensor([-18.0, -8.0]), harmful_truth, scale=scale)
    assert float(good_hold) < float(unsafe_execute)


def test_query_loss_rejects_empty_or_misaligned_candidate_sets() -> None:
    module = _module()
    with pytest.raises(ValueError, match="zero candidates"):
        module._query_loss(torch.tensor([]), torch.tensor([]), scale=torch.tensor(10.0))
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        module._query_loss(torch.tensor([1.0]), torch.tensor([1.0, 2.0]), scale=torch.tensor(10.0))


def test_trainer_keeps_epoch_zero_and_single_candidate_hold_decisions_in_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "validation_baseline_metrics" in text
    assert "validation_selected_epoch" in text
    assert "baseline_preserving_model_selection" in text
    assert "fine_tuning_improved_over_epoch0" in text
    assert "decision_query_set_count" in text
    assert "single_candidate_queries_included_in_hold_aware_selection_metrics" in text
    assert "CROSS_ENTROPY_ARGMIN_OVER_HOLD_ZERO_PLUS_CANDIDATES" in text
    assert "selected_action_false_reject_fraction" in text
