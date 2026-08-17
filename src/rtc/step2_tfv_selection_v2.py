"""Development HOLD/action threshold calibration for Direct-TFV.

The previous finite-sample residual margin was intentionally conservative, but with only 16 D3
calibration groups and alpha=0.10 its non-interpolated rank became the sample maximum.  That made
the guard mathematically prone to collapsing to HOLD even when the raw HOLD-reference model carried
useful action value.

This module calibrates the quantity the runtime actually needs: the minimum *predicted improvement*
required before a model-best non-HOLD action is admitted.  Calibration is lexicographic on a
rainfall-disjoint Development subset:

1. minimise the number of harmful selected actions;
2. among equally safe thresholds, minimise mean authoritative selected delta TFV;
3. then prefer lower regret and higher action rate.

It is an empirical Development decision calibration, not a formal probabilistic safety guarantee.
The value model and candidate ordering are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT = "PROJECT7_DIRECT_TFV_HOLD_ACTION_THRESHOLD_V2"


@dataclass(frozen=True)
class DirectTFVSelectionThreshold:
    contract: str
    minimum_predicted_improvement_m3: float
    calibration_groups: int
    harmful_selected_count: int
    action_selected_count: int
    selected_true_delta_tfv_m3: float
    selected_regret_m3: float
    candidate_threshold_count: int


def _vectors(
    *,
    best_candidate_prediction_m3: Sequence[float],
    best_candidate_truth_m3: Sequence[float],
    oracle_truth_m3: Sequence[float],
    oracle_is_hold: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(tuple(best_candidate_prediction_m3), dtype=np.float64)
    truth = np.asarray(tuple(best_candidate_truth_m3), dtype=np.float64)
    oracle = np.asarray(tuple(oracle_truth_m3), dtype=np.float64)
    oracle_hold = np.asarray(tuple(oracle_is_hold), dtype=np.bool_)
    if not (prediction.shape == truth.shape == oracle.shape == oracle_hold.shape):
        raise ValueError("selection vectors must have identical shapes")
    if prediction.ndim != 1 or prediction.size == 0:
        raise ValueError("selection calibration requires a non-empty 1-D group vector")
    if not bool(np.isfinite(prediction).all() and np.isfinite(truth).all() and np.isfinite(oracle).all()):
        raise ValueError("selection calibration vectors must be finite")
    return prediction, truth, oracle, oracle_hold


def evaluate_selection_threshold(
    *,
    best_candidate_prediction_m3: Sequence[float],
    best_candidate_truth_m3: Sequence[float],
    oracle_truth_m3: Sequence[float],
    oracle_is_hold: Sequence[bool],
    minimum_predicted_improvement_m3: float,
    practical_zero_m3: float = 1.0,
) -> dict[str, float | int]:
    prediction, truth, oracle, oracle_hold = _vectors(
        best_candidate_prediction_m3=best_candidate_prediction_m3,
        best_candidate_truth_m3=best_candidate_truth_m3,
        oracle_truth_m3=oracle_truth_m3,
        oracle_is_hold=oracle_is_hold,
    )
    threshold = float(minimum_predicted_improvement_m3)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum predicted improvement must be finite and non-negative")
    if not math.isfinite(float(practical_zero_m3)) or float(practical_zero_m3) < 0.0:
        raise ValueError("practical zero must be finite and non-negative")
    accept = prediction < -threshold
    selected_truth = np.where(accept, truth, 0.0)
    regret = selected_truth - oracle
    harmful = selected_truth > float(practical_zero_m3)
    beneficial = selected_truth < -float(practical_zero_m3)
    false_action = accept & oracle_hold
    oracle_hold_count = int(np.sum(oracle_hold))
    return {
        "groups": int(prediction.size),
        "minimum_predicted_improvement_m3": threshold,
        "action_selected_count": int(np.sum(accept)),
        "action_selected_fraction": float(np.mean(accept)),
        "hold_selected_fraction": float(np.mean(~accept)),
        "oracle_hold_groups": oracle_hold_count,
        "false_action_when_hold_oracle_count": int(np.sum(false_action)),
        "false_action_when_hold_oracle_fraction": (
            float(np.sum(false_action) / oracle_hold_count) if oracle_hold_count else 0.0
        ),
        "selected_beneficial_count": int(np.sum(beneficial)),
        "selected_beneficial_fraction": float(np.mean(beneficial)),
        "selected_harmful_count": int(np.sum(harmful)),
        "selected_harmful_fraction": float(np.mean(harmful)),
        "selected_true_delta_tfv_m3": float(np.mean(selected_truth)),
        "selected_regret_m3": float(np.mean(regret)),
        "mean_best_candidate_prediction_m3": float(np.mean(prediction)),
        "mean_best_candidate_truth_m3": float(np.mean(truth)),
    }


def calibrate_minimum_predicted_improvement(
    *,
    best_candidate_prediction_m3: Sequence[float],
    best_candidate_truth_m3: Sequence[float],
    oracle_truth_m3: Sequence[float],
    oracle_is_hold: Sequence[bool],
    practical_zero_m3: float = 1.0,
) -> DirectTFVSelectionThreshold:
    """Choose the least harmful empirically useful HOLD/action threshold.

    Candidate thresholds are exactly the observed predicted-improvement magnitudes plus zero.  The
    strict admission rule ``prediction < -threshold`` means a threshold equal to a group's predicted
    improvement excludes that group.  A HOLD-all solution is therefore available, but it is not
    chosen when a zero-harm threshold retains actions with negative mean authoritative delta TFV.
    """

    prediction, truth, oracle, oracle_hold = _vectors(
        best_candidate_prediction_m3=best_candidate_prediction_m3,
        best_candidate_truth_m3=best_candidate_truth_m3,
        oracle_truth_m3=oracle_truth_m3,
        oracle_is_hold=oracle_is_hold,
    )
    magnitudes = [0.0]
    magnitudes.extend(float(max(0.0, -value)) for value in prediction.tolist())
    thresholds = sorted(set(magnitudes))
    records: list[tuple[tuple[float, ...], float, dict[str, float | int]]] = []
    for threshold in thresholds:
        metrics = evaluate_selection_threshold(
            best_candidate_prediction_m3=prediction,
            best_candidate_truth_m3=truth,
            oracle_truth_m3=oracle,
            oracle_is_hold=oracle_hold,
            minimum_predicted_improvement_m3=float(threshold),
            practical_zero_m3=float(practical_zero_m3),
        )
        # Lexicographic objective.  Mean selected delta includes HOLD=0, so a useful zero-harm
        # threshold with negative benefit dominates the trivial HOLD-all solution.
        key = (
            float(metrics["selected_harmful_count"]),
            float(metrics["false_action_when_hold_oracle_count"]),
            float(metrics["selected_true_delta_tfv_m3"]),
            float(metrics["selected_regret_m3"]),
            -float(metrics["action_selected_count"]),
            float(threshold),
        )
        records.append((key, float(threshold), metrics))
    _, threshold, metrics = min(records, key=lambda item: item[0])
    return DirectTFVSelectionThreshold(
        contract=DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT,
        minimum_predicted_improvement_m3=float(threshold),
        calibration_groups=int(prediction.size),
        harmful_selected_count=int(metrics["selected_harmful_count"]),
        action_selected_count=int(metrics["action_selected_count"]),
        selected_true_delta_tfv_m3=float(metrics["selected_true_delta_tfv_m3"]),
        selected_regret_m3=float(metrics["selected_regret_m3"]),
        candidate_threshold_count=int(len(thresholds)),
    )


__all__ = [
    "DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT",
    "DirectTFVSelectionThreshold",
    "calibrate_minimum_predicted_improvement",
    "evaluate_selection_threshold",
]
