from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyResult:
    admissible: bool
    priority_flood_deterioration_ucb_m3: float
    priority_depth_deterioration_ucb_m: float
    nonpriority_new_flood_ucb_m3: float | None


def integrate_rate(rate_m3s: np.ndarray, dt_seconds: float) -> np.ndarray:
    """Integrate a non-negative flooding-rate trajectory along the time axis (-2)."""

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
    """Upper-tail empirical CVaR: lower is better for a loss such as TFV."""

    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot compute CVaR of an empty sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    threshold = np.quantile(x, alpha)
    tail = x[x >= threshold]
    return float(tail.mean())


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
    priority_flood_budget_m3: float,
    priority_depth_budget_m: float,
    quantile: float = 0.95,
    nonpriority_new_flood_budget_m3: float | None = None,
) -> SafetyResult:
    """Scenario/ensemble safety admission relative to the operational fallback.

    Arrays may include a leading ensemble/scenario dimension. The same fallback and
    rainfall scenarios must be used for the counterfactual comparison.
    """

    p = np.asarray(priority_indices, dtype=int)
    cand_pfv = priority_flood_volume(candidate_flood_rate, p, dt_seconds)
    base_pfv = priority_flood_volume(fallback_flood_rate, p, dt_seconds)
    pfv_ucb = one_sided_ucb(cand_pfv - base_pfv, quantile)

    cand_depth = np.asarray(candidate_depth, dtype=float)[..., p]
    base_depth = np.asarray(fallback_depth, dtype=float)[..., p]
    # Max deterioration at any priority location/time per scenario.
    depth_delta = (cand_depth - base_depth).max(axis=(-2, -1))
    depth_ucb = one_sided_ucb(depth_delta, quantile)

    new_flood_ucb: float | None = None
    if nonpriority_new_flood_budget_m3 is not None:
        node_count = np.asarray(candidate_flood_rate).shape[-1]
        mask = np.ones(node_count, dtype=bool)
        mask[p] = False
        cand_v = integrate_rate(candidate_flood_rate, dt_seconds)[..., mask]
        base_v = integrate_rate(fallback_flood_rate, dt_seconds)[..., mask]
        # Only newly transferred/additional flooding is penalised.
        added = np.clip(cand_v - base_v, 0.0, None).sum(axis=-1)
        new_flood_ucb = one_sided_ucb(added, quantile)

    admissible = (
        pfv_ucb <= priority_flood_budget_m3
        and depth_ucb <= priority_depth_budget_m
        and (
            new_flood_ucb is None
            or new_flood_ucb <= float(nonpriority_new_flood_budget_m3)
        )
    )
    return SafetyResult(
        admissible=bool(admissible),
        priority_flood_deterioration_ucb_m3=pfv_ucb,
        priority_depth_deterioration_ucb_m=depth_ucb,
        nonpriority_new_flood_ucb_m3=new_flood_ucb,
    )
