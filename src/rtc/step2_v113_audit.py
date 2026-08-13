"""Small, deterministic helpers for the V11.3 Train-only audit.

This module deliberately contains no model code.  It provides fail-closed
statistics used by the signed-magnitude and influence-prior audits so that the
same definitions are exercised by unit tests and the report scripts.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


def deterministic_event_split_v113(
    event_ids: Sequence[str], *, devfit_count: int = 10
) -> dict[str, object]:
    """Split forcing identifiers only, independent of hydraulic outcomes."""
    events = sorted({str(x) for x in event_ids})
    if len(events) < 2 or not 0 < int(devfit_count) < len(events):
        raise ValueError("V113 nested split requires both non-empty event sides")
    ordered = sorted(events, key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    devfit = ordered[: int(devfit_count)]
    devcheck = ordered[int(devfit_count) :]
    return {
        "selection_rule": "sha256(event_id) ascending, first devfit_count",
        "devfit_events": devfit,
        "devcheck_events": devcheck,
        "event_overlap": sorted(set(devfit) & set(devcheck)),
        "event_digest": hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest(),
    }


def _finite_pair(values: np.ndarray, active: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x)
    if active is None:
        a = np.ones(x.shape, dtype=bool)
    else:
        a = np.asarray(active, dtype=bool).reshape(-1)
        if a.shape != x.shape:
            raise ValueError("V113 values and active mask are misaligned")
    return x[mask], a[mask]


def signed_distribution_v113(
    values: np.ndarray, active: np.ndarray | None = None
) -> dict[str, float | int]:
    """Summarize unclipped signed effects in their physical units."""
    x, a = _finite_pair(values, active)
    if x.size == 0:
        return {
            "count": 0,
            "p50_abs": None,
            "p75_abs": None,
            "p90_abs": None,
            "p95_abs": None,
            "p99_abs": None,
            "max_abs": None,
            "signed_mean": None,
            "positive_active_fraction": None,
            "negative_active_fraction": None,
            "active_fraction": None,
            "effect_rms": None,
        }
    signed = x[a]
    return {
        "count": int(x.size),
        "p50_abs": float(np.quantile(np.abs(x), 0.50)),
        "p75_abs": float(np.quantile(np.abs(x), 0.75)),
        "p90_abs": float(np.quantile(np.abs(x), 0.90)),
        "p95_abs": float(np.quantile(np.abs(x), 0.95)),
        "p99_abs": float(np.quantile(np.abs(x), 0.99)),
        "max_abs": float(np.max(np.abs(x))),
        "signed_mean": float(np.mean(x)),
        "positive_active_fraction": float(np.mean(signed > 0)) if signed.size else float("nan"),
        "negative_active_fraction": float(np.mean(signed < 0)) if signed.size else float("nan"),
        "active_fraction": float(np.mean(a)),
        "effect_rms": float(np.sqrt(np.mean(np.square(x)))),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    if n_pos == 0 or n_pos == positive.size:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y = positive[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / float(n_pos)
    precision = tp / np.maximum(tp + fp, 1.0)
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def atlas_binary_metrics_v113(
    support_probability: np.ndarray,
    truth_abs_effect: np.ndarray,
    truth_active: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Measure support occurrence and absolute-effect mass capture."""
    support = np.asarray(support_probability, dtype=np.float64).reshape(-1)
    magnitude = np.asarray(truth_abs_effect, dtype=np.float64).reshape(-1)
    active = np.asarray(truth_active, dtype=bool).reshape(-1)
    if support.shape != magnitude.shape or support.shape != active.shape:
        raise ValueError("V113 Atlas/truth arrays are misaligned")
    finite = np.isfinite(support) & np.isfinite(magnitude)
    support, magnitude, active = support[finite], magnitude[finite], active[finite]
    predicted = support >= float(threshold)
    true_count = int(active.sum())
    predicted_count = int(predicted.sum())
    captured_mass = float(magnitude[predicted].sum())
    total_mass = float(magnitude.sum())
    return {
        "support_threshold": float(threshold),
        "sample_count": int(support.size),
        "true_active_count": true_count,
        "predicted_support_count": predicted_count,
        "active_recall": float((predicted & active).sum() / true_count) if true_count else float("nan"),
        "active_precision": float((predicted & active).sum() / predicted_count) if predicted_count else float("nan"),
        "auroc": _auroc(support, active),
        "auprc": _auprc(support, active),
        "effect_mass_recall": float(captured_mass / total_mass) if total_mass > 0 else float("nan"),
    }


__all__ = [
    "atlas_binary_metrics_v113",
    "deterministic_event_split_v113",
    "signed_distribution_v113",
]
