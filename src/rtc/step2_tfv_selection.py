"""Development-only HOLD/action calibration for Project7 Direct-TFV.

The Direct-TFV value model can learn useful candidate ordering while still being too eager to
execute an action.  This module therefore keeps the learned value model unchanged and calibrates a
one-sided margin on the *selected action residual*:

    residual = true_delta_tfv - predicted_delta_tfv

For a future group, the best predicted non-HOLD candidate is admitted only when

    predicted_delta_tfv + calibrated_margin < 0.

Otherwise the policy returns HOLD (exact delta TFV = 0 by contract).

The margin is a finite-sample upper quantile computed from a calibration split that must be
separate from model fitting and from the audit split.  This is a Development selection-calibration
mechanism, not a claim of formal conformal coverage under arbitrary distribution shift.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DIRECT_TFV_SELECTION_CONTRACT = "PROJECT7_DIRECT_TFV_HOLD_ACTION_SELECTION_GUARD_V1"


@dataclass(frozen=True)
class DirectTFVSelectionDesign:
    alpha: float = 0.10
    minimum_margin_m3: float = 0.0
    practical_zero_m3: float = 1.0

    def validate(self) -> None:
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("selection alpha must lie strictly between zero and one")
        for value in (self.minimum_margin_m3, self.practical_zero_m3):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("selection design margins must be finite and non-negative")


@dataclass(frozen=True)
class DirectTFVSelectionCalibration:
    contract: str
    alpha: float
    margin_m3: float
    calibration_groups: int
    calibration_action_groups: int
    residual_quantile_rank: int
    residuals_m3: tuple[float, ...]


def finite_sample_upper_quantile(values: Sequence[float], *, alpha: float) -> tuple[float, int]:
    """Return the split-conformal-style finite-sample upper quantile and one-based rank.

    For n finite values, the rank is ceil((n + 1) * (1 - alpha)), clipped to [1, n].
    The helper is deterministic and deliberately avoids interpolation.
    """

    data = np.asarray(tuple(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("cannot calibrate an upper quantile from no finite residuals")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    ordered = np.sort(data)
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1]), rank


def calibrate_selected_action_margin(
    *,
    best_candidate_prediction_m3: Sequence[float],
    best_candidate_truth_m3: Sequence[float],
    design: DirectTFVSelectionDesign = DirectTFVSelectionDesign(),
) -> DirectTFVSelectionCalibration:
    """Calibrate an upper error margin on actions the unguarded model would execute.

    A calibration group contributes when its best predicted candidate is below HOLD (prediction
    < 0), because only those groups expose selected-action optimism.  HOLD-only calibration groups
    do not dilute the residual distribution with artificial zero residuals.
    """

    design.validate()
    prediction = np.asarray(tuple(best_candidate_prediction_m3), dtype=np.float64)
    truth = np.asarray(tuple(best_candidate_truth_m3), dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("selection calibration prediction/truth arrays must be aligned 1-D vectors")
    finite = np.isfinite(prediction) & np.isfinite(truth)
    if not bool(np.all(finite)):
        raise ValueError("selection calibration inputs must be finite")
    action_mask = prediction < 0.0
    if int(np.sum(action_mask)) < 5:
        raise ValueError("selection calibration requires at least five raw action-selected groups")
    residuals = truth[action_mask] - prediction[action_mask]
    quantile, rank = finite_sample_upper_quantile(residuals, alpha=design.alpha)
    margin = max(float(design.minimum_margin_m3), 0.0, float(quantile))
    return DirectTFVSelectionCalibration(
        contract=DIRECT_TFV_SELECTION_CONTRACT,
        alpha=float(design.alpha),
        margin_m3=margin,
        calibration_groups=int(len(prediction)),
        calibration_action_groups=int(np.sum(action_mask)),
        residual_quantile_rank=int(rank),
        residuals_m3=tuple(float(value) for value in residuals.tolist()),
    )


def evaluate_selection_margin(
    *,
    best_candidate_prediction_m3: Sequence[float],
    best_candidate_truth_m3: Sequence[float],
    oracle_truth_m3: Sequence[float],
    oracle_is_hold: Sequence[bool],
    margin_m3: float,
    practical_zero_m3: float = 1.0,
) -> dict[str, float | int]:
    """Evaluate a HOLD guard without changing candidate ranking.

    The candidate identity is always the model's best non-HOLD candidate.  The margin only decides
    whether that candidate is admitted or replaced by HOLD, so this diagnostic isolates selection
    calibration from ranking quality.
    """

    prediction = np.asarray(tuple(best_candidate_prediction_m3), dtype=np.float64)
    truth = np.asarray(tuple(best_candidate_truth_m3), dtype=np.float64)
    oracle = np.asarray(tuple(oracle_truth_m3), dtype=np.float64)
    oracle_hold = np.asarray(tuple(oracle_is_hold), dtype=np.bool_)
    if not (prediction.shape == truth.shape == oracle.shape == oracle_hold.shape):
        raise ValueError("selection evaluation vectors must have identical shapes")
    if prediction.ndim != 1 or prediction.size == 0:
        raise ValueError("selection evaluation requires a non-empty 1-D group vector")
    if not bool(np.isfinite(prediction).all() and np.isfinite(truth).all() and np.isfinite(oracle).all()):
        raise ValueError("selection evaluation values must be finite")
    if not math.isfinite(float(margin_m3)) or float(margin_m3) < 0.0:
        raise ValueError("selection margin must be finite and non-negative")
    if not math.isfinite(float(practical_zero_m3)) or float(practical_zero_m3) < 0.0:
        raise ValueError("practical zero must be finite and non-negative")

    accept_action = prediction + float(margin_m3) < 0.0
    selected_truth = np.where(accept_action, truth, 0.0)
    regret = selected_truth - oracle
    harmful = selected_truth > float(practical_zero_m3)
    beneficial = selected_truth < -float(practical_zero_m3)
    false_action = accept_action & oracle_hold
    oracle_hold_count = int(np.sum(oracle_hold))
    return {
        "groups": int(prediction.size),
        "margin_m3": float(margin_m3),
        "action_selected_fraction": float(np.mean(accept_action)),
        "hold_selected_fraction": float(np.mean(~accept_action)),
        "oracle_hold_groups": oracle_hold_count,
        "false_action_when_hold_oracle_count": int(np.sum(false_action)),
        "false_action_when_hold_oracle_fraction": (
            float(np.sum(false_action) / oracle_hold_count) if oracle_hold_count else 0.0
        ),
        "selected_beneficial_fraction": float(np.mean(beneficial)),
        "selected_harmful_fraction": float(np.mean(harmful)),
        "selected_true_delta_tfv_m3": float(np.mean(selected_truth)),
        "selected_regret_m3": float(np.mean(regret)),
        "mean_best_candidate_prediction_m3": float(np.mean(prediction)),
        "mean_best_candidate_truth_m3": float(np.mean(truth)),
    }


__all__ = [
    "DIRECT_TFV_SELECTION_CONTRACT",
    "DirectTFVSelectionCalibration",
    "DirectTFVSelectionDesign",
    "calibrate_selected_action_margin",
    "evaluate_selection_margin",
    "finite_sample_upper_quantile",
]
