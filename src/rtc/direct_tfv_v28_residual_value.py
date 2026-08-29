"""Frozen-Q27 residual value correction for the Project7 V28 Development lane.

V28 does not refit the V27 value surface.  It fits a small deterministic ridge model on
features that describe how an already-supported q95 action differs from its raw proposal.
The residual is in the same exact policy-return units as Q27, so the deployed score is
``Q28 = Q27 + residual`` and HOLD remains exactly zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


V28_RESIDUAL_MODEL_CONTRACT = (
    "PROJECT7_STEP3_V28_Q95_MATCHED_RESIDUAL_VALUE_RIDGE_V1"
)
V28_RESIDUAL_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_V28_Q95_MATCHED_RESIDUAL_VALUE_CHECKPOINT_V1"
)
V28_RESIDUAL_RIDGE_GRID = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)

V28_RESIDUAL_FEATURE_NAMES = (
    "q27_supported_score_m3",
    "q95_contraction_scale",
    "q95_max_support_ratio",
    "q95_support_binding",
    "raw_first_move_l1",
    "supported_first_move_l1",
    "raw_to_supported_first_move_l1",
    "raw_to_supported_h120_l1",
    "raw_to_supported_total_variation_l1",
    "changed_facility_count",
    "network_stress_q75",
    "rain_level",
    "strong_storm_blend",
    "candidate_family_probe_050",
    "candidate_family_probe_100",
    "candidate_family_hydraulic",
    "candidate_family_auto_rbc_shadow",
)


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _family_one_hot(candidate_source: str) -> tuple[float, float, float, float]:
    source = str(candidate_source).upper()
    return (
        float("PROBE_SCALE_0.50" in source or "PROBE_050" in source),
        float("PROBE_SCALE_1.00" in source or "PROBE_100" in source),
        float("HYDRAULIC" in source),
        float("AUTO_RBC_SHADOW" in source),
    )


def build_v28_residual_features(
    *,
    q27_score_m3: float,
    q95_scale: float,
    q95_max_ratio: float,
    q95_binding: bool,
    raw_first_move_l1: float,
    supported_first_move_l1: float,
    raw_to_supported_first_move_l1: float,
    raw_to_supported_h120_l1: float,
    raw_to_supported_tv_l1: float,
    changed_facility_count: int,
    network_stress_q75: float,
    rain_level: float,
    strong_storm_blend: float,
    candidate_source: str,
) -> np.ndarray:
    """Build the event-independent V28 residual feature vector.

    No event identifier is accepted by this function.  All values are current causal-prefix or
    candidate-geometry diagnostics; candidate family is represented by a fixed one-hot code.
    """

    one_hot = _family_one_hot(candidate_source)
    values = np.asarray(
        [
            q27_score_m3,
            q95_scale,
            q95_max_ratio,
            float(bool(q95_binding)),
            raw_first_move_l1,
            supported_first_move_l1,
            raw_to_supported_first_move_l1,
            raw_to_supported_h120_l1,
            raw_to_supported_tv_l1,
            float(changed_facility_count),
            network_stress_q75,
            rain_level,
            strong_storm_blend,
            *one_hot,
        ],
        dtype=np.float64,
    )
    if values.shape != (len(V28_RESIDUAL_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("V28 residual feature vector is invalid")
    return values


def q28_score_m3(q27_score_m3: float, residual_m3: float) -> float:
    result = float(q27_score_m3) + float(residual_m3)
    if not math.isfinite(result):
        raise ValueError("V28 Q28 score is non-finite")
    return result


@dataclass(frozen=True)
class V28ResidualValueModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    intercept: float
    q27_checkpoint_sha256: str
    ridge: float

    @property
    def feature_width(self) -> int:
        return int(self.weight.size)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim == 1:
            if array.size != self.feature_width:
                raise ValueError("V28 residual feature width drifted")
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.feature_width or not np.isfinite(array).all():
            raise ValueError("V28 residual feature shape/content is invalid")
        return (array - self.feature_mean) / self.feature_scale

    def predict_many(self, features: np.ndarray) -> np.ndarray:
        standardized = self._standardize(features)
        values = standardized @ self.weight + float(self.intercept)
        if not np.isfinite(values).all():
            raise ValueError("V28 residual prediction is non-finite")
        return values

    def predict_m3(self, features: np.ndarray) -> float:
        return float(self.predict_many(np.asarray(features, dtype=np.float64))[0])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(prediction: np.ndarray, truth: np.ndarray) -> float:
    if len(prediction) < 2 or np.std(prediction) <= 1.0e-12 or np.std(truth) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(_rank(prediction), _rank(truth))[0, 1])
    return value if math.isfinite(value) else 0.0


def _pairwise_indices(decision_units: Sequence[str]) -> list[tuple[int, int]]:
    grouped: dict[str, list[int]] = {}
    for index, unit in enumerate(decision_units):
        grouped.setdefault(str(unit), []).append(index)
    pairs: list[tuple[int, int]] = []
    for unit in sorted(grouped):
        members = grouped[unit]
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                pairs.append((members[left], members[right]))
    return pairs


def _decision_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    decision_units: Sequence[str],
) -> dict[str, float | int]:
    selected: list[float] = []
    oracle: list[float] = []
    action_count = 0
    beneficial = 0
    harmful = 0
    top1_correct = 0
    for unit in sorted({str(value) for value in decision_units}):
        mask = np.asarray([str(value) == unit for value in decision_units], dtype=bool)
        index = int(np.argmin(prediction[mask]))
        local_truth = truth[mask]
        local_prediction = prediction[mask]
        truth_with_hold = np.minimum(local_truth, 0.0)
        top1_correct += int(index == int(np.argmin(truth_with_hold)))
        if float(local_prediction[index]) < 0.0:
            value = float(local_truth[index])
            action_count += 1
            if value < 0.0:
                beneficial += 1
            elif value > 0.0:
                harmful += 1
        else:
            value = 0.0
        selected.append(value)
        oracle.append(float(min(0.0, float(np.min(local_truth)))))
    selected_array = np.asarray(selected, dtype=np.float64)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    return {
        "query_count": int(len(selected_array)),
        "action_count": int(action_count),
        "hold_count": int(len(selected_array) - action_count),
        "beneficial_action_count": int(beneficial),
        "harmful_action_count": int(harmful),
        "beneficial_action_precision": float(beneficial / max(1, action_count)),
        "top1_accuracy": float(top1_correct / max(1, len(selected))),
        "mean_selected_true_delta_tfv_m3": float(selected_array.mean()) if len(selected_array) else 0.0,
        "mean_regret_m3": float(np.mean(selected_array - oracle_array)) if len(selected_array) else 0.0,
    }


def evaluate_v28_residual(
    model: V28ResidualValueModel,
    features: np.ndarray,
    q27_scores_m3: np.ndarray,
    truth_m3: np.ndarray,
    decision_units: Sequence[str],
) -> dict[str, Any]:
    residual = model.predict_many(features)
    q28 = np.asarray(q27_scores_m3, dtype=np.float64) + residual
    truth = _finite_vector(truth_m3, name="truth_m3")
    if q28.size != truth.size or len(decision_units) != truth.size:
        raise ValueError("V28 evaluation inputs are not aligned")
    error = q28 - truth
    pair_count = 0
    pair_correct = 0
    for left, right in _pairwise_indices(decision_units):
        true_delta = truth[left] - truth[right]
        if abs(float(true_delta)) <= 1.0e-6:
            continue
        pair_count += 1
        pair_correct += int(((q28[left] - q28[right]) < 0.0) == (true_delta < 0.0))
    sign = (q28 < 0.0) == (truth < 0.0)
    return {
        "residual_metrics": {
            "mae_m3": float(np.mean(np.abs(residual - (truth - q27_scores_m3)))),
            "rmse_m3": float(np.sqrt(np.mean(np.square(residual - (truth - q27_scores_m3))))),
            "spearman": _spearman(residual, truth - q27_scores_m3),
        },
        "q28_metrics": {
            "mae_m3": float(np.mean(np.abs(error))),
            "rmse_m3": float(np.sqrt(np.mean(np.square(error)))),
            "spearman": _spearman(q28, truth),
            "sign_accuracy": float(np.mean(sign)),
            "prediction_min_m3": float(q28.min()),
            "prediction_max_m3": float(q28.max()),
            "prediction_std_m3": float(q28.std()),
        },
        "pairwise": {
            "pair_count": int(pair_count),
            "pairwise_rank_accuracy": float(pair_correct / pair_count) if pair_count else 0.5,
        },
        "decision": _decision_metrics(q28, truth, decision_units),
    }


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> V28ResidualValueModel:
    x = _finite_matrix(features, name="features")
    y = _finite_vector(target, name="target")
    if x.shape[0] != y.size:
        raise ValueError("V28 residual fit arrays are not aligned")
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("V28 ridge must be finite and non-negative")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    z = (x - mean) / scale
    design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return V28ResidualValueModel(
        feature_mean=mean,
        feature_scale=scale,
        weight=beta[:-1],
        intercept=float(beta[-1]),
        q27_checkpoint_sha256="",
        ridge=float(ridge),
    )


def fit_v28_residual_value_model(
    *,
    train_features: np.ndarray,
    train_q27_scores_m3: np.ndarray,
    train_truth_m3: np.ndarray,
    train_groups: Sequence[str],
    train_units: Sequence[str],
    validation_features: np.ndarray,
    validation_q27_scores_m3: np.ndarray,
    validation_truth_m3: np.ndarray,
    validation_units: Sequence[str],
    q27_checkpoint_sha256: str,
    ridge_grid: Sequence[float] = V28_RESIDUAL_RIDGE_GRID,
) -> tuple[V28ResidualValueModel, dict[str, Any]]:
    """Fit residuals on Train and select ridge on Validation only."""

    x_train = _finite_matrix(train_features, name="train_features")
    q27_train = _finite_vector(train_q27_scores_m3, name="train_q27_scores_m3")
    y_train = _finite_vector(train_truth_m3, name="train_truth_m3")
    x_val = _finite_matrix(validation_features, name="validation_features")
    q27_val = _finite_vector(validation_q27_scores_m3, name="validation_q27_scores_m3")
    y_val = _finite_vector(validation_truth_m3, name="validation_truth_m3")
    if x_train.shape[0] != len(y_train) or len(q27_train) != len(y_train) or len(train_groups) != len(y_train):
        raise ValueError("V28 Train arrays are not aligned")
    if x_val.shape[0] != len(y_val) or len(q27_val) != len(y_val) or len(validation_units) != len(y_val):
        raise ValueError("V28 Validation arrays are not aligned")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("V28 Train/Validation feature widths differ")
    residual_train = y_train - q27_train
    unique_groups = sorted(set(str(value) for value in train_groups))
    if len(unique_groups) < 2:
        raise ValueError("V28 residual fit requires at least two Train groups")
    group_order = sorted(
        unique_groups,
        key=lambda value: hashlib.sha256(f"v28|{value}".encode("utf-8")).hexdigest(),
    )
    folds = min(5, len(group_order))
    assignment = {group: index % folds for index, group in enumerate(group_order)}
    cv_rows: list[dict[str, Any]] = []
    for ridge in ridge_grid:
        fold_metrics: list[dict[str, Any]] = []
        for fold in range(folds):
            holdout = np.asarray([assignment[str(value)] == fold for value in train_groups], dtype=bool)
            if not holdout.any() or (~holdout).sum() == 0:
                continue
            fitted = _fit_ridge(x_train[~holdout], residual_train[~holdout], float(ridge))
            report = evaluate_v28_residual(
                fitted,
                x_train[holdout],
                q27_train[holdout],
                y_train[holdout],
                [train_units[index] for index in np.flatnonzero(holdout)],
            )
            fold_metrics.append(report)
        if not fold_metrics:
            continue
        cv_rows.append(
            {
                "ridge": float(ridge),
                "fold_count": len(fold_metrics),
                "mean_decision_regret_m3": float(np.mean([row["decision"]["mean_regret_m3"] for row in fold_metrics])),
                "mean_selected_true_delta_tfv_m3": float(np.mean([row["decision"]["mean_selected_true_delta_tfv_m3"] for row in fold_metrics])),
                "mean_pairwise_rank_accuracy": float(np.mean([row["pairwise"]["pairwise_rank_accuracy"] for row in fold_metrics])),
                "mean_q28_rmse_m3": float(np.mean([row["q28_metrics"]["rmse_m3"] for row in fold_metrics])),
                "fold_reports": fold_metrics,
            }
        )
    if not cv_rows:
        raise RuntimeError("V28 residual CV produced no configuration")
    cv_rows.sort(key=lambda row: (float(row["mean_decision_regret_m3"]), float(row["mean_q28_rmse_m3"]), float(row["ridge"])))
    validation_rows: list[dict[str, Any]] = []
    fitted_by_ridge: dict[float, V28ResidualValueModel] = {}
    for row in cv_rows:
        ridge = float(row["ridge"])
        fitted = _fit_ridge(x_train, residual_train, ridge)
        fitted_by_ridge[ridge] = fitted
        validation_rows.append(
            {
                "ridge": ridge,
                "validation": evaluate_v28_residual(
                    fitted,
                    x_val,
                    q27_val,
                    y_val,
                    validation_units,
                ),
            }
        )
    selected_row = min(
        validation_rows,
        key=lambda row: (
            float(row["validation"]["decision"]["mean_regret_m3"]),
            float(row["validation"]["q28_metrics"]["rmse_m3"]),
            float(row["ridge"]),
        ),
    )
    selected = fitted_by_ridge[float(selected_row["ridge"])]
    selected = V28ResidualValueModel(
        feature_mean=selected.feature_mean,
        feature_scale=selected.feature_scale,
        weight=selected.weight,
        intercept=selected.intercept,
        q27_checkpoint_sha256=str(q27_checkpoint_sha256).lower(),
        ridge=selected.ridge,
    )
    return selected, {
        "selection_contract": "V28_TRAIN_GROUP_CV_THEN_VALIDATION_DECISION_REGRET_V1",
        "cv_folds": int(folds),
        "cv_configuration_count": len(cv_rows),
        "cv_ranked_configurations": cv_rows,
        "validation_candidates": validation_rows,
        "selected_ridge": float(selected.ridge),
        "selected_validation_metrics": selected_row["validation"],
        "test_used_for_training_or_model_selection": False,
        "scientific_metrics_block_runtime": False,
    }


def checkpoint_payload(
    model: V28ResidualValueModel,
    *,
    lineage: Mapping[str, Any],
    training_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": V28_RESIDUAL_CHECKPOINT_CONTRACT,
        "model_contract": V28_RESIDUAL_MODEL_CONTRACT,
        "feature_names": list(V28_RESIDUAL_FEATURE_NAMES),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "weight": model.weight.tolist(),
        "intercept": float(model.intercept),
        "q27_checkpoint_sha256": model.q27_checkpoint_sha256,
        "ridge": float(model.ridge),
        "lineage": dict(lineage),
        "training_report": dict(training_report),
        "validation_report": dict(validation_report),
        "test_report": dict(test_report),
        "development_only": True,
        "formal_evidence": False,
    }


def load_v28_residual_value_model(
    path: str,
    *,
    expected_q27_checkpoint_sha256: str,
    expected_dataset_manifest_sha256: str | None = None,
) -> tuple[V28ResidualValueModel, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != V28_RESIDUAL_CHECKPOINT_CONTRACT:
        raise ValueError("V28 runtime requires a V28 residual checkpoint")
    names = tuple(str(value) for value in payload.get("feature_names", ()))
    if names != V28_RESIDUAL_FEATURE_NAMES:
        raise ValueError("V28 residual feature contract drifted")
    q27_sha = str(payload.get("q27_checkpoint_sha256", "")).lower()
    if q27_sha != str(expected_q27_checkpoint_sha256).lower():
        raise ValueError("V28 residual checkpoint is bound to another Q27 checkpoint")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V28 residual checkpoint lacks lineage")
    if expected_dataset_manifest_sha256 is not None and str(lineage.get("dataset_manifest_sha256", "")).lower() != str(expected_dataset_manifest_sha256).lower():
        raise ValueError("V28 residual checkpoint/dataset lineage mismatch")
    if payload.get("development_only") is not True or payload.get("formal_evidence") is not False:
        raise ValueError("V28 residual checkpoint is not Development-only")
    model = V28ResidualValueModel(
        feature_mean=_finite_vector(payload.get("feature_mean"), name="feature_mean"),
        feature_scale=np.maximum(_finite_vector(payload.get("feature_scale"), name="feature_scale"), 1.0e-6),
        weight=_finite_vector(payload.get("weight"), name="weight"),
        intercept=float(payload.get("intercept", 0.0)),
        q27_checkpoint_sha256=q27_sha,
        ridge=float(payload.get("ridge", 0.0)),
    )
    if model.feature_width != len(V28_RESIDUAL_FEATURE_NAMES) or model.feature_mean.size != model.feature_width or model.feature_scale.size != model.feature_width:
        raise ValueError("V28 residual checkpoint feature width is invalid")
    return model, payload


__all__ = [
    "V28_RESIDUAL_CHECKPOINT_CONTRACT",
    "V28_RESIDUAL_FEATURE_NAMES",
    "V28_RESIDUAL_MODEL_CONTRACT",
    "V28_RESIDUAL_RIDGE_GRID",
    "V28ResidualValueModel",
    "build_v28_residual_features",
    "checkpoint_payload",
    "evaluate_v28_residual",
    "fit_v28_residual_value_model",
    "load_v28_residual_value_model",
    "q28_score_m3",
]
