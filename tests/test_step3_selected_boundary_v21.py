from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from rtc.direct_tfv_policy_return_selected_boundary_v21 import (
    SelectedBoundaryCalibratorV21,
    SelectedBoundaryPartsV21,
    SelectedBoundaryPreprocessorV21,
    build_selected_portfolio_feature_v21,
)


def _trainer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("train_step3_selected_boundary_v21_current")


def test_v21_hold_selected_candidate_is_exact_zero_even_with_nonzero_siblings() -> None:
    candidate_features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 0.5],
        ],
        dtype=torch.float32,
    )
    parts = build_selected_portfolio_feature_v21(
        candidate_features=candidate_features,
        rank_scores=torch.tensor([-1.0, 0.5]),
        selected_index=0,
        selected_action_mass=0.0,
    )
    assert torch.count_nonzero(parts.feature) == 0


def test_v21_sibling_features_change_portfolio_context_without_using_truth() -> None:
    base = torch.tensor([[1.0, 0.5], [0.2, 0.1]], dtype=torch.float32)
    changed = torch.tensor([[1.0, 0.5], [2.0, -1.0]], dtype=torch.float32)
    rank = torch.tensor([-1.0, 0.0], dtype=torch.float32)
    left = build_selected_portfolio_feature_v21(
        candidate_features=base,
        rank_scores=rank,
        selected_index=0,
        selected_action_mass=0.25,
    ).feature
    right = build_selected_portfolio_feature_v21(
        candidate_features=changed,
        rank_scores=rank,
        selected_index=0,
        selected_action_mass=0.25,
    ).feature
    assert not torch.allclose(left, right)
    assert torch.allclose(left[:2], right[:2])


def test_v21_noncentered_svd_and_no_intercept_preserve_zero_boundary() -> None:
    trainer = _trainer()
    features = np.asarray(
        [
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 0.5],
            [-2.0, 0.0, -1.0],
            [-1.0, 0.0, -0.5],
            [1.5, 0.2, 0.7],
            [-1.5, -0.2, -0.7],
            [1.2, 0.1, 0.6],
            [-1.2, -0.1, -0.6],
            [0.8, 0.1, 0.4],
            [-0.8, -0.1, -0.4],
            [1.8, 0.3, 0.9],
            [-1.8, -0.3, -0.9],
        ],
        dtype=float,
    )
    scale, components = trainer._fit_preprocessor(features)
    assert np.allclose(np.zeros((1, 3)) / scale @ components.T, 0.0)
    pre = SelectedBoundaryPreprocessorV21(
        feature_scale=torch.as_tensor(scale, dtype=torch.float32),
        components=torch.as_tensor(components, dtype=torch.float32),
    )
    calibrator = SelectedBoundaryCalibratorV21(
        preprocessor=pre,
        boundary_weight=torch.ones(pre.output_dim),
        magnitude_weight=torch.ones(pre.output_dim),
        target_scale_m3=1000.0,
    )
    out = calibrator.predict(SelectedBoundaryPartsV21(feature=torch.zeros(3)))
    assert float(out.hold_score) == 0.0
    assert float(out.advantage_m3) == 0.0
    assert bool(out.execute) is False


def test_v21_crossfit_operates_on_one_selected_label_per_query() -> None:
    trainer = _trainer()
    n = 24
    queries = np.asarray([f"Q{i:02d}" for i in range(n)])
    returns = np.asarray([5.0 + i if i < 12 else -(5.0 + i) for i in range(n)], dtype=float)
    sign = np.where(returns >= 0.0, 1.0, -1.0)
    features = np.stack(
        (
            sign,
            sign * np.linspace(0.5, 1.5, n),
            np.linspace(-1.0, 1.0, n),
            sign * np.linspace(-1.0, 1.0, n),
        ),
        axis=1,
    )
    rows = {
        "features": features,
        "returns": returns,
        "queries": queries,
        "selected_sources": np.asarray(["TYPE_AWARE_HYDRAULIC_PRESSURE"] * n),
        "oracle_sources": np.asarray(["TYPE_AWARE_HYDRAULIC_PRESSURE"] * n),
        "selected_indices": np.arange(n),
        "oracle_indices": np.arange(n),
    }
    result = trainer._crossfit(rows, seed=42)
    assert result["query_group_disjoint"] is True
    assert len(result["fold_query_counts"]) == 6
    assert sum(result["fold_query_counts"]) == n
    assert result["selected"]["ridge"] in trainer.RIDGE_GRID
    assert np.isfinite(result["selected"]["auc"])
