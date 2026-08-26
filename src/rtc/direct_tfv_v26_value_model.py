"""Compact exact-policy-return model for Project7 Step3 V26.

The model predicts system-wide cumulative TFV difference for *each* engineering-feasible candidate
relative to HOLD.  It intentionally has no UCB/admission safety head: model quality is measured on
Validation/Test, while online selection simply minimizes predicted exact return together with the
physical HOLD value of zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


V26_VALUE_MODEL_CONTRACT = "PROJECT7_STEP3_V26_EXACT_RETURN_RIDGE_VALUE_MODEL_V1"
V26_VALUE_CHECKPOINT_CONTRACT = "PROJECT7_STEP3_V26_EXACT_RETURN_VALUE_CHECKPOINT_V1"
V26_RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] < 1 or out.shape[1] < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return out


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return out


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


def _spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) <= 1.0e-12 or np.std(truth) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(_rank(pred), _rank(truth))[0, 1])
    return value if math.isfinite(value) else 0.0


def _auc_beneficial(prediction: np.ndarray, truth: np.ndarray) -> float:
    labels = truth < 0.0
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = _rank(-prediction)
    positive_rank_sum = float(ranks[labels].sum())
    return float(
        (positive_rank_sum - positives * (positives - 1) / 2.0)
        / (positives * negatives)
    )


def candidate_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    pred = _finite_vector(prediction, name="prediction")
    y = _finite_vector(truth, name="truth")
    if pred.size != y.size:
        raise ValueError("candidate metrics are not aligned")
    beneficial = y < 0.0
    action = pred < 0.0
    return {
        "count": int(len(y)),
        "mae_m3": float(np.mean(np.abs(pred - y))),
        "rmse_m3": float(np.sqrt(np.mean(np.square(pred - y)))),
        "spearman": _spearman(pred, y),
        "auc_beneficial_vs_nonbeneficial": _auc_beneficial(pred, y),
        "sign_accuracy": float(np.mean((pred < 0.0) == beneficial)),
        "beneficial_count": int(beneficial.sum()),
        "predicted_action_count": int(action.sum()),
        "prediction_min_m3": float(pred.min()),
        "prediction_max_m3": float(pred.max()),
        "prediction_std_m3": float(pred.std()),
    }


def decision_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: Sequence[str],
) -> dict[str, float | int]:
    """Evaluate the actual V26 decision: argmin(candidate predictions, HOLD=0)."""
    pred = _finite_vector(prediction, name="prediction")
    y = _finite_vector(truth, name="truth")
    if pred.size != y.size or len(query_ids) != pred.size:
        raise ValueError("decision metrics are not aligned")
    selected_truth: list[float] = []
    oracle_truth: list[float] = []
    action_count = 0
    beneficial_actions = 0
    harmful_actions = 0
    for query in sorted({str(value) for value in query_ids}):
        mask = np.asarray([str(value) == query for value in query_ids], dtype=bool)
        local_pred = pred[mask]
        local_truth = y[mask]
        index = int(np.argmin(local_pred))
        if float(local_pred[index]) < 0.0:
            realized = float(local_truth[index])
            action_count += 1
            if realized < 0.0:
                beneficial_actions += 1
            elif realized > 0.0:
                harmful_actions += 1
        else:
            realized = 0.0
        selected_truth.append(realized)
        oracle_truth.append(float(min(0.0, float(np.min(local_truth)))))
    selected = np.asarray(selected_truth, dtype=np.float64)
    oracle = np.asarray(oracle_truth, dtype=np.float64)
    return {
        "query_count": int(len(selected)),
        "action_count": int(action_count),
        "hold_count": int(len(selected) - action_count),
        "beneficial_action_count": int(beneficial_actions),
        "harmful_action_count": int(harmful_actions),
        "beneficial_action_precision": float(beneficial_actions / max(1, action_count)),
        "mean_selected_true_delta_tfv_m3": float(selected.mean()) if len(selected) else 0.0,
        "sum_selected_true_delta_tfv_m3": float(selected.sum()),
        "mean_oracle_delta_tfv_m3": float(oracle.mean()) if len(oracle) else 0.0,
        "mean_regret_m3": float(np.mean(selected - oracle)) if len(selected) else 0.0,
    }


def _fit_ridge(
    features: np.ndarray,
    transformed_truth: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = _finite_matrix(features, name="features")
    y = _finite_vector(transformed_truth, name="transformed_truth")
    if x.shape[0] != y.size:
        raise ValueError("ridge inputs are not aligned")
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
    return beta[:-1], float(beta[-1]), mean, scale


@dataclass(frozen=True)
class FittedV26ValueModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    intercept: float
    target_scale_m3: float
    ridge: float

    @property
    def feature_width(self) -> int:
        return int(self.weight.size)

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or int(x.shape[1]) != self.feature_width or not np.isfinite(x).all():
            raise ValueError("V26 prediction feature shape/content is invalid")
        coordinate = ((x - self.feature_mean) / self.feature_scale) @ self.weight + self.intercept
        return np.sinh(np.clip(coordinate, -8.0, 8.0)) * float(self.target_scale_m3)


class V26ValueModelModule(nn.Module):
    def __init__(self, fitted: FittedV26ValueModel) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(fitted.feature_mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(fitted.feature_scale, dtype=torch.float32))
        self.register_buffer("weight", torch.as_tensor(fitted.weight, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(float(fitted.intercept), dtype=torch.float32))
        self.register_buffer("target_scale_m3", torch.tensor(float(fitted.target_scale_m3), dtype=torch.float32))

    def predict(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 1 or int(feature.numel()) != int(self.weight.numel()):
            raise ValueError("V26 runtime feature width drifted")
        x = feature.to(dtype=self.weight.dtype, device=self.weight.device)
        coordinate = torch.dot((x - self.feature_mean) / self.feature_scale, self.weight) + self.intercept
        return torch.sinh(torch.clamp(coordinate, -8.0, 8.0)) * self.target_scale_m3


def fit_v26_value_model(
    train_features: np.ndarray,
    train_truth: np.ndarray,
    validation_features: np.ndarray,
    validation_truth: np.ndarray,
    validation_query_ids: Sequence[str],
    *,
    ridge_grid: Sequence[float] = V26_RIDGE_GRID,
) -> tuple[FittedV26ValueModel, dict[str, Any]]:
    x_train = _finite_matrix(train_features, name="train_features")
    y_train = _finite_vector(train_truth, name="train_truth")
    x_val = _finite_matrix(validation_features, name="validation_features")
    y_val = _finite_vector(validation_truth, name="validation_truth")
    if x_train.shape[0] != y_train.size or x_val.shape[0] != y_val.size:
        raise ValueError("V26 train/validation arrays are not aligned")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("V26 train/validation feature widths differ")
    target_scale = float(max(np.median(np.abs(y_train)), 1.0))
    transformed = np.arcsinh(y_train / target_scale)
    candidates: list[dict[str, Any]] = []
    best: tuple[float, float, float, FittedV26ValueModel] | None = None
    for ridge_value in ridge_grid:
        ridge = float(ridge_value)
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("V26 ridge grid must contain finite positive values")
        weight, intercept, mean, scale = _fit_ridge(x_train, transformed, ridge=ridge)
        fitted = FittedV26ValueModel(
            feature_mean=mean,
            feature_scale=scale,
            weight=weight,
            intercept=intercept,
            target_scale_m3=target_scale,
            ridge=ridge,
        )
        val_pred = fitted.predict_numpy(x_val)
        cand_metrics = candidate_metrics(val_pred, y_val)
        dec_metrics = decision_metrics(val_pred, y_val, validation_query_ids)
        candidates.append({"ridge": ridge, "candidate": cand_metrics, "decision": dec_metrics})
        # Standard Validation model selection: minimize realized exact TFV under the downstream V26
        # decision rule, then use candidate RMSE and ridge only as deterministic tie-breakers.
        key = (
            float(dec_metrics["mean_selected_true_delta_tfv_m3"]),
            float(cand_metrics["rmse_m3"]),
            ridge,
        )
        if best is None or key < best[:3]:
            best = (key[0], key[1], key[2], fitted)
    assert best is not None
    selected_model = best[3]
    selected_pred = selected_model.predict_numpy(x_val)
    report = {
        "selected_ridge": float(selected_model.ridge),
        "target_scale_m3": float(selected_model.target_scale_m3),
        "validation_candidate_metrics": candidate_metrics(selected_pred, y_val),
        "validation_decision_metrics": decision_metrics(selected_pred, y_val, validation_query_ids),
        "ridge_candidates": candidates,
        "selection_rule": "MIN_VALIDATION_REALIZED_EXACT_TFV_THEN_RMSE_THEN_RIDGE",
        "scientific_metrics_block_runtime": False,
    }
    return selected_model, report


def checkpoint_payload(
    fitted: FittedV26ValueModel,
    *,
    lineage: Mapping[str, Any],
    training_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": V26_VALUE_CHECKPOINT_CONTRACT,
        "model_contract": V26_VALUE_MODEL_CONTRACT,
        "feature_mean": fitted.feature_mean.tolist(),
        "feature_scale": fitted.feature_scale.tolist(),
        "weight": fitted.weight.tolist(),
        "intercept": float(fitted.intercept),
        "target_scale_m3": float(fitted.target_scale_m3),
        "ridge": float(fitted.ridge),
        "lineage": dict(lineage),
        "training_report": dict(training_report),
        "test_report": dict(test_report),
        "development_only": True,
        "formal_evidence": False,
    }


def load_v26_value_model(
    path: str,
    *,
    device: torch.device,
    expected_lineage: Mapping[str, Any],
) -> tuple[V26ValueModelModule, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != V26_VALUE_CHECKPOINT_CONTRACT:
        raise ValueError("V26 runtime requires a V26 value checkpoint")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V26 checkpoint lacks lineage")
    for key, expected in expected_lineage.items():
        actual = lineage.get(key)
        if isinstance(expected, str):
            if not isinstance(actual, str) or actual.lower() != expected.lower():
                raise ValueError(f"V26 checkpoint lineage mismatch: {key}")
        elif actual != expected:
            raise ValueError(f"V26 checkpoint lineage mismatch: {key}")
    fitted = FittedV26ValueModel(
        feature_mean=_finite_vector(payload["feature_mean"], name="feature_mean"),
        feature_scale=np.maximum(_finite_vector(payload["feature_scale"], name="feature_scale"), 1.0e-6),
        weight=_finite_vector(payload["weight"], name="weight"),
        intercept=float(payload["intercept"]),
        target_scale_m3=float(payload["target_scale_m3"]),
        ridge=float(payload["ridge"]),
    )
    if fitted.feature_mean.size != fitted.weight.size or fitted.feature_scale.size != fitted.weight.size:
        raise ValueError("V26 checkpoint feature width mismatch")
    if fitted.target_scale_m3 <= 0.0 or not math.isfinite(fitted.target_scale_m3):
        raise ValueError("V26 checkpoint target scale is invalid")
    model = V26ValueModelModule(fitted).to(device)
    model.eval()
    return model, payload


__all__ = [
    "FittedV26ValueModel",
    "V26_RIDGE_GRID",
    "V26_VALUE_CHECKPOINT_CONTRACT",
    "V26_VALUE_MODEL_CONTRACT",
    "V26ValueModelModule",
    "candidate_metrics",
    "checkpoint_payload",
    "decision_metrics",
    "fit_v26_value_model",
    "load_v26_value_model",
]
