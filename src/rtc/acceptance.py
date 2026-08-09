from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    metrics: dict[str, float]
    failed_metrics: tuple[str, ...]


def rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(truth, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("RMSE arrays must have identical non-zero shape")
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(predicted: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(truth, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("MAE arrays must have identical non-zero shape")
    return float(np.mean(np.abs(p - y)))


def nse(predicted: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float).reshape(-1)
    y = np.asarray(truth, dtype=float).reshape(-1)
    if p.shape != y.shape or p.size < 2:
        raise ValueError("NSE arrays must have identical length >= 2")
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((p - y) ** 2) / denom)


def _rank(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    # Average tied ranks.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, count in enumerate(counts):
            if count > 1:
                tied = np.where(inverse == idx)[0]
                ranks[tied] = ranks[tied].mean()
    return ranks


def rank_correlation(predicted: np.ndarray, truth: np.ndarray) -> float:
    p = _rank(predicted)
    y = _rank(truth)
    if p.size < 2 or np.std(p) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(p, y)[0, 1])


def sign_accuracy(predicted_effect: np.ndarray, true_effect: np.ndarray, *, tolerance: float = 1e-9) -> float:
    p = np.asarray(predicted_effect, dtype=float).reshape(-1)
    y = np.asarray(true_effect, dtype=float).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("effect arrays must have equal shape")
    mask = np.abs(y) > tolerance
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(p[mask]) == np.sign(y[mask])))


def event_balanced_metric(
    predicted: np.ndarray,
    truth: np.ndarray,
    event_ids: np.ndarray,
    metric,
) -> float:
    p = np.asarray(predicted)
    y = np.asarray(truth)
    events = np.asarray(event_ids).astype(str).reshape(-1)
    if p.shape[0] != events.size or y.shape[0] != events.size:
        raise ValueError("event ids must align with sample axis")
    scores: list[float] = []
    for event in sorted(set(events.tolist())):
        idx = events == event
        value = float(metric(p[idx], y[idx]))
        if np.isfinite(value):
            scores.append(value)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def evaluate_step1_depth(
    *,
    predicted_depth: np.ndarray,
    true_depth: np.ndarray,
    event_ids: np.ndarray,
    unobserved_node_indices: np.ndarray,
    priority_node_indices: np.ndarray,
    wet_depth_threshold_m: float = 0.05,
) -> dict[str, float]:
    """Acceptance family required before Step2 can claim reconstructed-state inputs."""

    pred = np.asarray(predicted_depth, dtype=float)
    truth = np.asarray(true_depth, dtype=float)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Step1 depth arrays must be [sample,node] with equal shape")
    u = np.asarray(unobserved_node_indices, dtype=int)
    p = np.asarray(priority_node_indices, dtype=int)
    wet = truth[:, u] >= float(wet_depth_threshold_m)
    wet_rmse = rmse(pred[:, u][wet], truth[:, u][wet]) if np.any(wet) else float("nan")
    return {
        "unobserved_depth_rmse_m": rmse(pred[:, u], truth[:, u]),
        "unobserved_depth_nse": nse(pred[:, u], truth[:, u]),
        "unobserved_depth_event_balanced_rmse_m": event_balanced_metric(
            pred[:, u], truth[:, u], event_ids, rmse
        ),
        "priority_depth_rmse_m": rmse(pred[:, p], truth[:, p]),
        "priority_depth_nse": nse(pred[:, p], truth[:, p]),
        "wet_unobserved_depth_rmse_m": wet_rmse,
    }


def evaluate_action_effects(
    *,
    predicted_delta_pfv: np.ndarray,
    true_delta_pfv: np.ndarray,
    predicted_delta_tfv: np.ndarray,
    true_delta_tfv: np.ndarray,
    event_ids: np.ndarray,
) -> dict[str, float]:
    """Same-state control-effect acceptance for Step2/Step3 readiness."""

    return {
        "delta_pfv_mae_m3": mae(predicted_delta_pfv, true_delta_pfv),
        "delta_tfv_mae_m3": mae(predicted_delta_tfv, true_delta_tfv),
        "delta_pfv_sign_accuracy": sign_accuracy(predicted_delta_pfv, true_delta_pfv),
        "delta_tfv_sign_accuracy": sign_accuracy(predicted_delta_tfv, true_delta_tfv),
        "candidate_pfv_rank_correlation": rank_correlation(predicted_delta_pfv, true_delta_pfv),
        "candidate_tfv_rank_correlation": rank_correlation(predicted_delta_tfv, true_delta_tfv),
        "delta_tfv_event_balanced_mae_m3": event_balanced_metric(
            np.asarray(predicted_delta_tfv).reshape(-1, 1),
            np.asarray(true_delta_tfv).reshape(-1, 1),
            event_ids,
            mae,
        ),
    }


def apply_metric_thresholds(
    metrics: Mapping[str, float],
    *,
    minimum: Mapping[str, float] | None = None,
    maximum: Mapping[str, float] | None = None,
) -> AcceptanceResult:
    """Fail closed on missing/non-finite metrics or violated frozen thresholds."""

    minimum = dict(minimum or {})
    maximum = dict(maximum or {})
    failures: list[str] = []
    for name, threshold in minimum.items():
        value = metrics.get(name)
        if value is None or not np.isfinite(value) or value < threshold:
            failures.append(name)
    for name, threshold in maximum.items():
        value = metrics.get(name)
        if value is None or not np.isfinite(value) or value > threshold:
            failures.append(name)
    return AcceptanceResult(
        passed=not failures,
        metrics={k: float(v) for k, v in metrics.items()},
        failed_metrics=tuple(sorted(set(failures))),
    )
