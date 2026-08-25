"""Small, conservative Development-only H120 selected-action value calibrator for V25."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


V25_VALUE_CALIBRATOR_CONTRACT = (
    "PROJECT7_STEP3_V25_MATCHED_SELECTED_H120_VALUE_CALIBRATION_V1"
)
V25_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_V25_VALUE_CALIBRATOR_CHECKPOINT_V1"
)
V25_VALUE_FEATURE_CONTRACT = (
    "PROJECT7_STEP3_V25_SELECTED_V23_PORTFOLIO_V15_RANK_V21_FEATURE_V1"
)
V25_VALUE_ERROR_QUANTILE = 0.90
V25_VALUE_RIDGE = 1.0
V25_VALUE_OOF_FOLDS = 6


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
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
    x = _rank(pred)
    y = _rank(truth)
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else 0.0


def _auc_beneficial(prediction: np.ndarray, truth: np.ndarray) -> float:
    """AUC for beneficial (negative delta TFV) versus harmful/non-beneficial actions."""

    labels = truth < 0.0
    positives = int(np.sum(labels))
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    # Larger -prediction means more likely beneficial; ties receive average ranks.
    ranks = _rank(-prediction)
    positive_rank_sum = float(np.sum(ranks[labels]))
    auc = (positive_rank_sum - positives * (positives - 1) / 2.0) / (positives * negatives)
    return float(auc) if math.isfinite(float(auc)) else 0.5


def _fit_ridge(
    features: np.ndarray,
    truth: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    ridge: float = V25_VALUE_RIDGE,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = _finite_vector(truth, name="truth")
    if x.ndim != 2 or x.shape[0] != y.size or not np.isfinite(x).all():
        raise ValueError("ridge features must be a finite [N,D] matrix aligned with truth")
    if mean is None:
        mean = x.mean(axis=0)
    if scale is None:
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
    return beta[:-1], float(beta[-1]), np.asarray(mean), np.asarray(scale)


def _fold_ids(groups: Sequence[str], folds: int) -> np.ndarray:
    unique = sorted({str(group) for group in groups})
    if len(unique) < folds:
        raise ValueError("OOF folds exceed the number of rainfall groups")
    mapping = {group: index % folds for index, group in enumerate(unique)}
    return np.asarray([mapping[str(group)] for group in groups], dtype=np.int64)


def _event_balanced_binary(
    predicted_action: np.ndarray,
    truth: np.ndarray,
    groups: Sequence[str],
) -> dict[str, float]:
    values: list[float] = []
    harmful: list[float] = []
    for group in sorted({str(value) for value in groups}):
        mask = np.asarray([str(value) == group for value in groups], dtype=bool)
        values.append(float(np.mean(predicted_action[mask] == (truth[mask] < 0.0))))
        actions = predicted_action[mask]
        harmful.append(float(np.mean(truth[mask][actions] >= 0.0)) if actions.any() else 0.0)
    return {
        "group_count": float(len(values)),
        "mean_accuracy": float(np.mean(values)) if values else 0.0,
        "mean_harmful_action_rate": float(np.mean(harmful)) if harmful else 0.0,
    }


def value_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    margin: float,
    groups: Sequence[str],
) -> dict[str, Any]:
    pred = _finite_vector(prediction, name="prediction")
    y = _finite_vector(truth, name="truth")
    if pred.size != y.size or len(groups) != pred.size:
        raise ValueError("value metrics inputs are not aligned")
    upper = pred + float(margin)
    action = upper < 0.0
    beneficial = y < 0.0
    harmful_action = action & ~beneficial
    precision = float(np.mean(beneficial[action])) if action.any() else 0.0
    recall = float(np.sum(action & beneficial) / max(1, int(np.sum(beneficial))))
    return {
        "count": int(len(y)),
        "mae_m3": float(np.mean(np.abs(pred - y))),
        "rmse_m3": float(np.sqrt(np.mean(np.square(pred - y)))),
        "spearman": _spearman(pred, y),
        "auc_beneficial_vs_harmful": _auc_beneficial(pred, y),
        "prediction_std_m3": float(np.std(pred)),
        "sign_accuracy": float(np.mean((pred < 0.0) == beneficial)),
        "beneficial_action_precision": precision,
        "beneficial_action_recall": recall,
        "false_admitted_harmful_action_rate": float(
            np.sum(harmful_action) / max(1, int(np.sum(action)))
        ),
        "harmful_action_count": int(np.sum(harmful_action)),
        "action_count": int(np.sum(action)),
        "hold_count": int(np.sum(~action)),
        "beneficial_count": int(np.sum(beneficial)),
        "calibration_coverage": float(np.mean(y <= upper)),
        "event_balanced": _event_balanced_binary(action, y, groups),
        "collapse": bool(np.all(action) or not np.any(action)),
        "upper_bound_margin_m3": float(margin),
    }


@dataclass(frozen=True)
class FittedV25ValueCalibrator:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    intercept: float
    one_sided_error_margin_m3: float
    ridge: float
    error_quantile: float

    @property
    def feature_width(self) -> int:
        return int(self.weight.size)

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or int(x.shape[1]) != self.feature_width:
            raise ValueError("V25 feature width drifted")
        if not np.isfinite(x).all():
            raise ValueError("V25 features contain non-finite values")
        z = (x - self.feature_mean) / self.feature_scale
        return z @ self.weight + float(self.intercept)

    def predict_tensor(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 1 or int(features.numel()) != self.feature_width:
            raise ValueError("V25 tensor feature width drifted")
        mean = torch.as_tensor(self.feature_mean, dtype=features.dtype, device=features.device)
        scale = torch.as_tensor(self.feature_scale, dtype=features.dtype, device=features.device)
        weight = torch.as_tensor(self.weight, dtype=features.dtype, device=features.device)
        return torch.dot((features - mean) / scale, weight) + features.new_tensor(float(self.intercept))


class V25ValueCalibratorModule(nn.Module):
    """Read-only runtime form of the fitted linear value model."""

    def __init__(self, fitted: FittedV25ValueCalibrator) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(fitted.feature_mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(fitted.feature_scale, dtype=torch.float32))
        self.register_buffer("weight", torch.as_tensor(fitted.weight, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(float(fitted.intercept), dtype=torch.float32))
        self.register_buffer(
            "one_sided_error_margin_m3",
            torch.tensor(float(fitted.one_sided_error_margin_m3), dtype=torch.float32),
        )

    def predict(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 1 or int(feature.numel()) != int(self.weight.numel()):
            raise ValueError("V25 runtime feature width drifted")
        value = torch.dot(
            (feature.to(self.feature_mean.device) - self.feature_mean) / self.feature_scale,
            self.weight,
        ) + self.intercept
        upper = value + self.one_sided_error_margin_m3
        return value, upper


def fit_v25_value_calibrator(
    features: np.ndarray,
    truth: np.ndarray,
    groups: Sequence[str],
    *,
    ridge: float = V25_VALUE_RIDGE,
    folds: int = V25_VALUE_OOF_FOLDS,
    error_quantile: float = V25_VALUE_ERROR_QUANTILE,
) -> tuple[FittedV25ValueCalibrator, dict[str, Any], np.ndarray]:
    """Fit deterministic group-disjoint OOF calibration and a final train-only model."""

    x = np.asarray(features, dtype=np.float64)
    y = _finite_vector(truth, name="truth")
    if x.ndim != 2 or x.shape[0] != y.size or len(groups) != y.size:
        raise ValueError("V25 fit inputs are not aligned")
    if not 0.5 < float(error_quantile) < 1.0:
        raise ValueError("V25 error quantile must be in (0.5,1)")
    fold_ids = _fold_ids(groups, int(folds))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(int(folds)):
        train = fold_ids != fold
        test = ~train
        weight, intercept, mean, scale = _fit_ridge(x[train], y[train], ridge=ridge)
        oof[test] = ((x[test] - mean) / scale) @ weight + intercept
    if not np.isfinite(oof).all():
        raise RuntimeError("V25 group-disjoint OOF did not cover every query")
    residual = y - oof
    margin = max(0.0, float(np.quantile(residual, float(error_quantile), method="linear")))
    weight, intercept, mean, scale = _fit_ridge(x, y, ridge=ridge)
    fitted = FittedV25ValueCalibrator(
        feature_mean=mean,
        feature_scale=scale,
        weight=weight,
        intercept=intercept,
        one_sided_error_margin_m3=margin,
        ridge=float(ridge),
        error_quantile=float(error_quantile),
    )
    metrics = value_metrics(oof, y, margin, groups)
    metrics["train_oof_supported"] = bool(
        metrics["auc_beneficial_vs_harmful"] >= 0.60
        and metrics["prediction_std_m3"] > 1.0e-6
        and not metrics["collapse"]
        and metrics["harmful_action_count"] == 0
        and metrics["beneficial_action_precision"] >= 0.50
    )
    metrics["support_contract"] = (
        "AUC>=0.60;prediction_std>1e-6;no_action_or_hold_collapse;"
        "harmful_admitted_action_count=0;beneficial_action_precision>=0.50"
    )
    metrics.update(
        {
            "oof_folds": int(folds),
            "oof_group_count": int(len(set(str(group) for group in groups))),
            "oof_fold_assignment": {str(group): int(fold) for group, fold in sorted({str(g): int(f) for g, f in zip(groups, fold_ids)}.items())},
            "error_residual_quantile_m3": float(margin),
            "error_quantile": float(error_quantile),
            "ridge": float(ridge),
        }
    )
    return fitted, metrics, oof


def fitted_payload(
    fitted: FittedV25ValueCalibrator,
    *,
    lineage: Mapping[str, Any],
    oof_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": V25_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT,
        "calibrator_contract": V25_VALUE_CALIBRATOR_CONTRACT,
        "feature_contract": V25_VALUE_FEATURE_CONTRACT,
        "development_only": True,
        "formal_evidence": False,
        "feature_mean": fitted.feature_mean.tolist(),
        "feature_scale": fitted.feature_scale.tolist(),
        "weight": fitted.weight.tolist(),
        "intercept": float(fitted.intercept),
        "one_sided_error_margin_m3": float(fitted.one_sided_error_margin_m3),
        "ridge": float(fitted.ridge),
        "error_quantile": float(fitted.error_quantile),
        "oof_metrics": dict(oof_metrics),
        "lineage": dict(lineage),
    }


def load_v25_value_calibrator(
    path: str,
    *,
    device: torch.device,
    expected_lineage: Mapping[str, Any],
) -> tuple[V25ValueCalibratorModule, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != V25_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT:
        raise ValueError("V25 runtime requires the V25 value calibrator checkpoint")
    if payload.get("development_only") is not True or payload.get("formal_evidence") is not False:
        raise ValueError("V25 calibrator must remain Development-only")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V25 calibrator lacks lineage")
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(f"V25 calibrator lineage mismatch: {key}")
    fitted = FittedV25ValueCalibrator(
        feature_mean=_finite_vector(payload["feature_mean"], name="feature_mean"),
        feature_scale=np.maximum(_finite_vector(payload["feature_scale"], name="feature_scale"), 1.0e-6),
        weight=_finite_vector(payload["weight"], name="weight"),
        intercept=float(payload["intercept"]),
        one_sided_error_margin_m3=float(payload["one_sided_error_margin_m3"]),
        ridge=float(payload["ridge"]),
        error_quantile=float(payload["error_quantile"]),
    )
    if fitted.feature_mean.size != fitted.weight.size or fitted.feature_scale.size != fitted.weight.size:
        raise ValueError("V25 calibrator feature width mismatch")
    if fitted.one_sided_error_margin_m3 < 0.0 or not math.isfinite(fitted.one_sided_error_margin_m3):
        raise ValueError("V25 calibrator error margin is invalid")
    model = V25ValueCalibratorModule(fitted).to(device)
    model.eval()
    return model, payload


__all__ = [
    "FittedV25ValueCalibrator",
    "V25_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT",
    "V25_VALUE_CALIBRATOR_CONTRACT",
    "V25_VALUE_ERROR_QUANTILE",
    "V25_VALUE_FEATURE_CONTRACT",
    "V25_VALUE_OOF_FOLDS",
    "V25_VALUE_RIDGE",
    "V25ValueCalibratorModule",
    "fit_v25_value_calibrator",
    "fitted_payload",
    "load_v25_value_calibrator",
    "value_metrics",
]
