"""Event-level evidence for Project7 whole-system TFV control.

"Stable TFV reduction" is deliberately not defined from monotonic improvement at every
10-minute decision.  Sewer storage shifts water through time, so an optimal action may
look neutral or temporarily adverse before reducing event-total overflow.  Stability is
therefore evaluated on paired event outcomes against No-control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

V122_TFV_EVIDENCE_CONTRACT = "PROJECT7_V122_EVENT_BALANCED_PAIRED_TFV_EVIDENCE_V1"


@dataclass(frozen=True)
class PairedTFVSummaryV122:
    event_count: int
    mean_reduction_fraction: float
    median_reduction_fraction: float
    nonworse_fraction: float
    worst_reduction_fraction: float
    best_reduction_fraction: float
    mean_saved_volume_m3: float
    median_saved_volume_m3: float
    worst_extra_volume_m3: float
    bootstrap_lower_95_mean_reduction_fraction: float
    bootstrap_upper_95_mean_reduction_fraction: float


def paired_tfv_summary_v122(
    proposed_tfv_m3: Sequence[float] | np.ndarray,
    no_control_tfv_m3: Sequence[float] | np.ndarray,
    *,
    noninferiority_tolerance_m3: float = 0.0,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> PairedTFVSummaryV122:
    proposed = np.asarray(proposed_tfv_m3, dtype=np.float64).reshape(-1)
    baseline = np.asarray(no_control_tfv_m3, dtype=np.float64).reshape(-1)
    if proposed.shape != baseline.shape or proposed.size == 0:
        raise ValueError("paired TFV vectors must be non-empty and aligned")
    if not np.isfinite(proposed).all() or not np.isfinite(baseline).all():
        raise ValueError("paired TFV vectors contain non-finite values")
    if np.any(proposed < 0.0) or np.any(baseline < 0.0):
        raise ValueError("TFV cannot be negative")
    if noninferiority_tolerance_m3 < 0.0:
        raise ValueError("non-inferiority tolerance cannot be negative")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    saved = baseline - proposed
    # A zero-No-control event has no meaningful relative reduction.  It remains in the
    # absolute-volume evidence but is excluded from the relative-effect denominator.
    relative = np.divide(
        saved,
        baseline,
        out=np.zeros_like(saved),
        where=baseline > 0.0,
    )
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, proposed.size, size=(int(bootstrap_samples), proposed.size))
    boot_mean = relative[index].mean(axis=1)
    lower, upper = np.quantile(boot_mean, [0.025, 0.975])
    nonworse = proposed <= baseline + float(noninferiority_tolerance_m3)
    return PairedTFVSummaryV122(
        event_count=int(proposed.size),
        mean_reduction_fraction=float(np.mean(relative)),
        median_reduction_fraction=float(np.median(relative)),
        nonworse_fraction=float(np.mean(nonworse)),
        worst_reduction_fraction=float(np.min(relative)),
        best_reduction_fraction=float(np.max(relative)),
        mean_saved_volume_m3=float(np.mean(saved)),
        median_saved_volume_m3=float(np.median(saved)),
        worst_extra_volume_m3=float(max(0.0, -float(np.min(saved)))),
        bootstrap_lower_95_mean_reduction_fraction=float(lower),
        bootstrap_upper_95_mean_reduction_fraction=float(upper),
    )


def stable_reduction_gate_v122(
    summary: PairedTFVSummaryV122,
    *,
    minimum_nonworse_fraction: float,
    maximum_worst_extra_volume_m3: float,
    require_positive_mean: bool = True,
    require_positive_median: bool = True,
) -> tuple[bool, tuple[str, ...]]:
    """Apply *pre-registered* stability tolerances without embedding arbitrary defaults.

    The caller must supply the event-wise non-inferiority tolerance and acceptable
    nonworse fraction from the study protocol.  This prevents changing a 5%, 1%, or
    '5-out-of-6' criterion after inspecting results.
    """

    if not 0.0 <= minimum_nonworse_fraction <= 1.0:
        raise ValueError("minimum_nonworse_fraction must lie in [0,1]")
    if maximum_worst_extra_volume_m3 < 0.0:
        raise ValueError("maximum_worst_extra_volume_m3 cannot be negative")
    reasons: list[str] = []
    if require_positive_mean and summary.mean_reduction_fraction <= 0.0:
        reasons.append("event-balanced mean TFV reduction is not positive")
    if require_positive_median and summary.median_reduction_fraction <= 0.0:
        reasons.append("median paired TFV reduction is not positive")
    if summary.nonworse_fraction < minimum_nonworse_fraction:
        reasons.append("paired nonworse-event fraction is below the frozen requirement")
    if summary.worst_extra_volume_m3 > maximum_worst_extra_volume_m3:
        reasons.append("worst-event TFV degradation exceeds the frozen tolerance")
    return (not reasons, tuple(reasons))


__all__ = [
    "PairedTFVSummaryV122",
    "V122_TFV_EVIDENCE_CONTRACT",
    "paired_tfv_summary_v122",
    "stable_reduction_gate_v122",
]
