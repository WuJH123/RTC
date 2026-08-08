from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyResult:
    admissible: bool
    worst_site_flood_deterioration_ucb_m3: float
    aggregate_priority_flood_deterioration_ucb_m3: float
    worst_site_depth_deterioration_ucb_m: float
    nonpriority_new_flood_ucb_m3: float | None


def integrate_rate(rate_m3s: np.ndarray, dt_seconds: float) -> np.ndarray:
    rate = np.asarray(rate_m3s, dtype=float)
    if rate.ndim < 2:
        raise ValueError("expected at least [time, node]")
    return np.clip(rate, 0.0, None).sum(axis=-2) * float(dt_seconds)


def total_flood_volume(rate_m3s: np.ndarray, dt_seconds: float) -> np.ndarray:
    return integrate_rate(rate_m3s, dt_seconds).sum(axis=-1)


def priority_flood_volume(
    rate_m3s: np.ndarray, priority_indices: np.ndarray, dt_seconds: float
) -> np.ndarray:
    volumes = integrate_rate(rate_m3s, dt_seconds)
    return volumes[..., np.asarray(priority_indices, dtype=int)].sum(axis=-1)


def cvar(values: np.ndarray, alpha: float = 0.9) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot compute CVaR of an empty sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    threshold = np.quantile(x, alpha)
    return float(x[x >= threshold].mean())


def one_sided_ucb(values: np.ndarray, quantile: float = 0.95) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot compute UCB of an empty sample")
    return float(np.quantile(x, quantile))


def assess_priority_safety(
    *,
    candidate_flood_rate: np.ndarray,
    fallback_flood_rate: np.ndarray,
    candidate_depth: np.ndarray,
    fallback_depth: np.ndarray,
    priority_indices: np.ndarray,
    dt_seconds: float,
    per_site_flood_budget_m3: float,
    per_site_depth_budget_m: float,
    aggregate_priority_flood_budget_m3: float | None = None,
    quantile: float = 0.95,
    nonpriority_new_flood_budget_m3: float | None = None,
) -> SafetyResult:
    """Hard safety admission with no compensation between priority locations."""

    p = np.asarray(priority_indices, dtype=int)
    cand_v = integrate_rate(candidate_flood_rate, dt_seconds)
    base_v = integrate_rate(fallback_flood_rate, dt_seconds)
    site_delta = cand_v[..., p] - base_v[..., p]
    # Ensemble/scenario axis is assumed to be the leading axis.
    site_ucb = np.quantile(site_delta, quantile, axis=0)
    worst_site_flood_ucb = float(np.max(site_ucb))
    aggregate_ucb = one_sided_ucb(site_delta.sum(axis=-1), quantile)

    cand_depth = np.asarray(candidate_depth, dtype=float)[..., p]
    base_depth = np.asarray(fallback_depth, dtype=float)[..., p]
    # Per scenario and site, compare the worst time-local depth deterioration.
    depth_delta = (cand_depth - base_depth).max(axis=-2)
    depth_site_ucb = np.quantile(depth_delta, quantile, axis=0)
    worst_site_depth_ucb = float(np.max(depth_site_ucb))

    new_flood_ucb: float | None = None
    if nonpriority_new_flood_budget_m3 is not None:
        mask = np.ones(cand_v.shape[-1], dtype=bool)
        mask[p] = False
        added = np.clip(cand_v[..., mask] - base_v[..., mask], 0.0, None).sum(axis=-1)
        new_flood_ucb = one_sided_ucb(added, quantile)

    admissible = (
        worst_site_flood_ucb <= per_site_flood_budget_m3
        and worst_site_depth_ucb <= per_site_depth_budget_m
        and (
            aggregate_priority_flood_budget_m3 is None
            or aggregate_ucb <= aggregate_priority_flood_budget_m3
        )
        and (
            new_flood_ucb is None
            or new_flood_ucb <= float(nonpriority_new_flood_budget_m3)
        )
    )
    return SafetyResult(
        admissible=bool(admissible),
        worst_site_flood_deterioration_ucb_m3=worst_site_flood_ucb,
        aggregate_priority_flood_deterioration_ucb_m3=aggregate_ucb,
        worst_site_depth_deterioration_ucb_m=worst_site_depth_ucb,
        nonpriority_new_flood_ucb_m3=new_flood_ucb,
    )
