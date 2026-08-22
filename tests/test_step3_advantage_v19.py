from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from rtc.direct_tfv_policy_return_advantage_v19 import (
    ZeroAnchoredAdvantagePreprocessorV19,
    ZeroAnchoredAdvantageRegressorV19,
    build_zero_anchored_advantage_parts_v19,
)


def _trainer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("train_step3_advantage_v19_current")


def _preprocessor(context_dim: int = 3, action_dim: int = 4) -> ZeroAnchoredAdvantagePreprocessorV19:
    return ZeroAnchoredAdvantagePreprocessorV19(
        context_mean=torch.zeros(context_dim),
        context_std=torch.ones(context_dim),
        context_components=torch.eye(context_dim)[:2],
        action_scale=torch.ones(action_dim),
        action_components=torch.eye(action_dim)[:2],
        explicit_scale=torch.ones(11),
    )


def test_v19_hold_identity_is_exact_zero_for_any_query_context() -> None:
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    active = torch.linspace(0.0, 1.0, 109)
    candidate = active.clone()
    context = torch.tensor([7.0, -3.0, 4.0])
    candidate_features = torch.arange(13, dtype=torch.float32)

    parts = build_zero_anchored_advantage_parts_v19(
        context_features=context,
        candidate_features=candidate_features,
        raw_step2_score_m3=torch.tensor(-99999.0),
        active_target=active,
        candidate_target=candidate,
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
        supervisory_mask=mask,
        target_scale_m3=10000.0,
    )
    assert torch.count_nonzero(parts.action_dense) == 0
    assert torch.count_nonzero(parts.explicit) == 0

    preprocessor = _preprocessor()
    regressor = ZeroAnchoredAdvantageRegressorV19(
        preprocessor=preprocessor,
        weight=torch.linspace(-2.0, 3.0, preprocessor.output_dim),
        target_scale_m3=10000.0,
    )
    output = regressor.predict(parts)
    assert float(output.coordinate) == 0.0
    assert float(output.advantage_m3) == 0.0
    assert bool(output.execute) is False


def test_v19_nonzero_action_keeps_signed_and_context_interaction_signal() -> None:
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    active = torch.zeros(109)
    candidate = active.clone()
    candidate[0] = 0.5
    candidate[1] = 0.25
    candidate_features = torch.zeros(13)
    candidate_features[9:] = torch.tensor([0.5, -0.2, 0.3, 0.1])

    parts_a = build_zero_anchored_advantage_parts_v19(
        context_features=torch.tensor([1.0, 0.0, 0.0]),
        candidate_features=candidate_features,
        raw_step2_score_m3=torch.tensor(-500.0),
        active_target=active,
        candidate_target=candidate,
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
        supervisory_mask=mask,
        target_scale_m3=1000.0,
    )
    parts_b = build_zero_anchored_advantage_parts_v19(
        context_features=torch.tensor([0.0, 1.0, 0.0]),
        candidate_features=candidate_features,
        raw_step2_score_m3=torch.tensor(-500.0),
        active_target=active,
        candidate_target=candidate,
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
        supervisory_mask=mask,
        target_scale_m3=1000.0,
    )
    assert float(parts_a.explicit[5]) > 0.0
    assert float(parts_a.explicit[6]) > 0.0
    assert float(parts_a.explicit[7]) == 0.0

    preprocessor = _preprocessor()
    design_a = preprocessor(parts_a)
    design_b = preprocessor(parts_b)
    assert not torch.allclose(design_a, design_b)


def test_v19_oof_is_query_group_disjoint_and_uses_all_candidate_records() -> None:
    trainer = _trainer()
    query_count = 12
    records_per_query = 2
    n = query_count * records_per_query
    queries = np.asarray(
        [f"Q{query:02d}" for query in range(query_count) for _ in range(records_per_query)]
    )
    selected_mask = np.asarray(
        [local == 0 for _query in range(query_count) for local in range(records_per_query)],
        dtype=bool,
    )
    returns = []
    for query in range(query_count):
        selected_return = 2.0 + query if query < 6 else -(2.0 + query)
        sibling_return = selected_return * 0.5
        returns.extend((selected_return, sibling_return))
    returns_array = np.asarray(returns, dtype=float)

    context = np.zeros((n, 3), dtype=float)
    action = np.zeros((n, 4), dtype=float)
    explicit = np.zeros((n, 11), dtype=float)
    for index, value in enumerate(returns_array):
        query = index // records_per_query
        context[index] = [query / 10.0, (query % 3) / 2.0, 1.0]
        sign = 1.0 if value >= 0.0 else -1.0
        action[index] = [sign, sign * 0.5, 0.2 + query / 100.0, 0.1]
        explicit[index, 0] = sign
        explicit[index, 1] = 0.1 + query / 100.0
        explicit[index, 4] = 0.2

    rows = {
        "context": context,
        "action": action,
        "explicit": explicit,
        "returns": returns_array,
        "queries": queries,
        "sources": np.asarray(["TYPE_AWARE_HYDRAULIC_PRESSURE"] * n),
        "selected_mask": selected_mask,
        "rank_scores": np.tile(np.asarray([0.0, 1.0]), query_count),
    }
    result = trainer._crossfit(rows, scale=10.0, seed=42)
    assert result["query_group_disjoint"] is True
    assert len(result["fold_query_counts"]) == 6
    assert sum(result["fold_query_counts"]) == query_count
    assert rows["returns"].size == 2 * int(rows["selected_mask"].sum())
    assert result["selected"]["ridge"] in trainer.RIDGE_GRID
    assert np.isfinite(result["selected"]["auc"])


def test_v19_weighted_ridge_has_no_intercept_and_zero_design_predicts_zero() -> None:
    trainer = _trainer()
    design = np.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.0],
            [-1.0, 0.0],
            [-0.5, 0.0],
        ],
        dtype=float,
    )
    returns = np.asarray([10.0, 5.0, -10.0, -5.0], dtype=float)
    queries = np.asarray(["A", "B", "C", "D"])
    weight = trainer._fit_ridge(
        design,
        returns,
        queries,
        scale=10.0,
        ridge=0.1,
    )
    assert weight.shape == (2,)
    assert float(np.zeros(2) @ weight) == 0.0
