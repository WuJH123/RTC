from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.direct_tfv_policy_return_query_margin import (
    DIRECT_TFV_QUERY_MARGIN_CONTRACT,
    QueryConditionedPolicyReturnAdapter,
    build_query_margin_features,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_direct_tfv_policy_return_query_margin_current.py"
SCORE = ROOT / "scripts" / "score_direct_tfv_policy_return_query_margin_calibration_current.py"
STEP3 = ROOT / "src" / "rtc" / "step3_tfv_value_mpc_v14.py"
RUNNER = ROOT / "scripts" / "run_policy_direct_tfv_query_margin_development.py"


def _zero(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)


def test_query_margin_decomposition_is_invariant_to_common_rank_shift() -> None:
    adapter = QueryConditionedPolicyReturnAdapter(target_scale_m3=1000.0)
    _zero(adapter)
    context = torch.zeros(11)
    candidate = torch.zeros((3, 9))
    first = adapter(
        raw_rank_scores_m3=torch.tensor([30.0, -20.0, 10.0]),
        context_features=context,
        candidate_features=candidate,
    )
    shifted = adapter(
        raw_rank_scores_m3=torch.tensor([5030.0, 4980.0, 5010.0]),
        context_features=context,
        candidate_features=candidate,
    )
    assert torch.allclose(
        first.relative_rank_normalized,
        shifted.relative_rank_normalized,
        atol=1e-6,
    )
    assert int(torch.argmin(first.predicted_returns_m3)) == 1
    assert float(first.predicted_returns_m3[1]) == pytest.approx(
        float(first.query_best_margin_m3)
    )


def test_query_margin_sign_is_separate_from_candidate_rank() -> None:
    adapter = QueryConditionedPolicyReturnAdapter(target_scale_m3=1000.0)
    _zero(adapter)
    # With all hidden weights zero, changing only the query-margin output bias moves the whole query
    # across HOLD while preserving candidate ordering exactly.
    adapter.margin_head[-1].bias.data.fill_(-0.25)
    out = adapter(
        raw_rank_scores_m3=torch.tensor([100.0, -50.0]),
        context_features=torch.zeros(11),
        candidate_features=torch.zeros((2, 9)),
    )
    assert int(torch.argmin(out.predicted_returns_m3)) == 1
    assert float(out.query_best_margin_m3) == pytest.approx(-250.0)
    assert float(out.predicted_returns_m3[1]) == pytest.approx(-250.0)

    adapter.margin_head[-1].bias.data.fill_(0.25)
    hold = adapter(
        raw_rank_scores_m3=torch.tensor([100.0, -50.0]),
        context_features=torch.zeros(11),
        candidate_features=torch.zeros((2, 9)),
    )
    assert int(torch.argmin(hold.predicted_returns_m3)) == 1
    assert float(hold.query_best_margin_m3) == pytest.approx(250.0)


def test_query_feature_contract_enforces_82_of_109_and_three_families() -> None:
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    active = torch.zeros(109)
    targets = torch.zeros((2, 109))
    targets[0, 0] = 0.2
    targets[1, 1] = 0.3
    context, candidates = build_query_margin_features(
        current_state=torch.zeros((4, 3)),
        rainfall_scenarios=torch.zeros((3, 72, 4, 1)),
        previous_actuator_flow=torch.zeros(109),
        active_target=active,
        candidate_targets=targets,
        base_step2_scores_m3=torch.tensor([-10.0, -20.0]),
        candidate_sources=[
            "STEP2_H10_PROBE_SCALE_0.50",
            "TYPE_AWARE_HYDRAULIC_PRESSURE",
        ],
        supervisory_mask=mask,
        target_scale_m3=1000.0,
    )
    assert tuple(context.shape) == (11,)
    assert tuple(candidates.shape) == (2, 9)

    bad = targets.clone()
    bad[0, 100] = 0.5
    with pytest.raises(ValueError, match="passive"):
        build_query_margin_features(
            current_state=torch.zeros((4, 3)),
            rainfall_scenarios=torch.zeros((3, 72, 4, 1)),
            previous_actuator_flow=torch.zeros(109),
            active_target=active,
            candidate_targets=bad,
            base_step2_scores_m3=torch.tensor([-10.0, -20.0]),
            candidate_sources=[
                "STEP2_H10_PROBE_SCALE_0.50",
                "TYPE_AWARE_HYDRAULIC_PRESSURE",
            ],
            supervisory_mask=mask,
            target_scale_m3=1000.0,
        )


def test_training_protocol_is_fixed_and_fresh_validation_is_not_model_selected() -> None:
    text = TRAIN.read_text(encoding="utf-8")
    assert "RANK_EPOCHS = 8" in text
    assert "MARGIN_EPOCHS = 40" in text
    assert "--deprecated-validation-dataset" in text
    assert "fresh_validation_used_for_hyperparameter_selection\": False" in text
    assert "validation_groups & deprecated_groups" in text
    assert "ready_for_calibration\": accepted" in text


def test_runtime_ranks_before_conformal_and_keeps_policy_lock_closed() -> None:
    text = STEP3.read_text(encoding="utf-8")
    assert "selected_index = int(np.argmin(relative))" in text
    assert "upper = score + margin" in text
    assert text.index("selected_index = int(np.argmin(relative))") < text.index(
        "upper = score + margin"
    )
    assert "conformal uncertainty is deliberately not used to rerank" in text

    runner = RUNNER.read_text(encoding="utf-8")
    assert '"ready_for_policy_lock": False' in runner
    assert '"online_swmm_candidate_search": False' in runner
    module = __import__(
        "rtc.direct_tfv_policy_return_query_margin",
        fromlist=["DIRECT_TFV_QUERY_MARGIN_CONTRACT"],
    )
    assert (
        DIRECT_TFV_QUERY_MARGIN_CONTRACT
        in module.DIRECT_TFV_QUERY_MARGIN_CONTRACT
    )


def test_calibration_scorer_is_query_joint_not_row_independent() -> None:
    text = SCORE.read_text(encoding="utf-8")
    assert "by_query" in text
    assert "query_conditioned_best_candidate_margin_m3" in text
    assert '"calibration_used_for_training": False' in text
