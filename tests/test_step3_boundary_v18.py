from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from rtc.direct_tfv_policy_return_query_margin_v18 import (
    BoundaryPreprocessorV18,
    LinearBoundaryCalibratorV18,
    build_boundary_feature_parts_v18,
)


def _trainer_helpers():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    module = importlib.import_module("train_step3_query_margin_v18_current")
    return module._auc_hold, module._choose_threshold


def test_v18_boundary_features_preserve_query_and_selected_candidate_signal() -> None:
    context = torch.tensor([1.0, 2.0, 3.0])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    parts = build_boundary_feature_parts_v18(
        context_features=context,
        candidate_features=candidates,
        raw_rank_scores_m3=torch.tensor([100.0, -200.0]),
        rank_scores_normalized=torch.tensor([0.5, -0.5]),
        selected_candidate_index=torch.tensor(1),
        target_scale_m3=1_000.0,
    )
    assert tuple(parts.dense.shape) == (9,)
    assert tuple(parts.explicit.shape) == (5,)
    assert torch.allclose(parts.dense[:3], context)
    assert torch.allclose(parts.dense[3:5], candidates[1])
    assert float(parts.explicit[2]) == 1.0


def test_v18_train_oof_threshold_is_not_hard_coded_to_zero() -> None:
    auc_hold, choose_threshold = _trainer_helpers()
    scores = np.asarray([-3.0, -2.0, -1.0, 0.2, 0.4, 0.6])
    returns = np.asarray([-10.0, -8.0, -6.0, 2.0, 4.0, 5.0])
    threshold, metrics = choose_threshold(scores, returns)
    assert threshold != 0.0
    assert metrics["fb"] == 0.0
    assert metrics["fr"] == 0.0
    assert metrics["collapse"] is False
    assert auc_hold(scores, returns) == 1.0


def test_v18_calibrator_uses_frozen_threshold_not_action_quota() -> None:
    preprocessor = BoundaryPreprocessorV18(
        dense_mean=torch.zeros(2),
        dense_std=torch.ones(2),
        components=torch.eye(2),
        explicit_mean=torch.zeros(1),
        explicit_std=torch.ones(1),
    )
    calibrator = LinearBoundaryCalibratorV18(
        preprocessor=preprocessor,
        boundary_weight=torch.tensor([1.0, 0.0, 0.0]),
        boundary_bias=0.0,
        decision_threshold=0.5,
        magnitude_weight=torch.zeros(3),
        magnitude_bias=1.0,
        target_scale_m3=1_000.0,
    )
    action_parts = type(
        "Parts",
        (),
        {"dense": torch.tensor([0.0, 0.0]), "explicit": torch.tensor([0.0])},
    )()
    hold_parts = type(
        "Parts",
        (),
        {"dense": torch.tensor([1.0, 0.0]), "explicit": torch.tensor([0.0])},
    )()
    action = calibrator.predict(
        parts=action_parts,
        relative_rank_normalized=torch.tensor([0.0, 1.0]),
        selected_candidate_index=torch.tensor(0),
    )
    hold = calibrator.predict(
        parts=hold_parts,
        relative_rank_normalized=torch.tensor([0.0, 1.0]),
        selected_candidate_index=torch.tensor(0),
    )
    assert float(action.boundary_distance) < 0.0
    assert float(action.query_best_margin_m3) < 0.0
    assert float(hold.boundary_distance) > 0.0
    assert float(hold.query_best_margin_m3) > 0.0
