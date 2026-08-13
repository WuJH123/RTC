"""Read-only diagnostics for hydraulic effects when flooding is unchanged.

This module intentionally contains no model or loss code.  It defines the
fixed numerical and horizon-bucket contracts used by the V10 D2 audit.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np


NO_FLOOD_EPSILON_V100 = 1e-8
HORIZON_BUCKETS_V100 = (
    "0_30_min",
    "30_120_min",
    "120_360_min",
)


def no_flood_mask_v100(values: np.ndarray, *, epsilon: float = NO_FLOOD_EPSILON_V100) -> np.ndarray:
    """Return the fixed inclusive no-flood numerical mask."""
    if not np.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("flood deltas must be finite")
    return np.abs(array) <= float(epsilon)


def bucket_slices_v100(horizon_steps: int, sample_seconds: int = 300) -> Mapping[str, np.ndarray]:
    """Return the pre-registered 0-30/30-120/120-360 minute index buckets."""
    horizon = int(horizon_steps)
    dt = int(sample_seconds)
    if horizon <= 0 or dt <= 0:
        raise ValueError("horizon_steps and sample_seconds must be positive")
    minutes = (np.arange(horizon, dtype=np.int64) + 1) * dt / 60.0
    masks = OrderedDict(
        (
            ("0_30_min", (minutes > 0.0) & (minutes <= 30.0)),
            ("30_120_min", (minutes > 30.0) & (minutes <= 120.0)),
            ("120_360_min", (minutes > 120.0) & (minutes <= 360.0)),
        )
    )
    result = OrderedDict((name, np.flatnonzero(mask)) for name, mask in masks.items())
    joined = np.concatenate(tuple(result.values())) if result else np.empty(0, dtype=np.int64)
    if not np.array_equal(joined, np.arange(horizon, dtype=np.int64)):
        raise RuntimeError("V100 horizon buckets are not exhaustive/disjoint for this horizon")
    return result


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "median": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def summarize_no_flood_values_v100(
    values: np.ndarray,
    no_flood_mask: np.ndarray,
    *,
    epsilon: float = NO_FLOOD_EPSILON_V100,
) -> dict[str, float | int]:
    """Summarize values for complete sample rows with no flooding delta.

    The first dimension is the sample dimension.  All remaining dimensions
    (time/node/actuator) must be identical in ``values`` and the mask.  A
    complete row is used deliberately: it answers whether the whole sampled
    hydraulic field changed without a flooding response.  Cell-level reports
    can use :func:`no_flood_mask_v100` directly.
    """
    del epsilon  # The mask is already constructed under the fixed contract.
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(no_flood_mask, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError("values and no_flood_mask must have the same shape")
    if array.ndim == 0:
        raise ValueError("values must have a sample dimension")
    row_mask = mask.reshape(mask.shape[0], -1).all(axis=1)
    selected = array[row_mask]
    result = _distribution(selected)
    result["no_flood_count"] = int(row_mask.sum())
    result["sample_count"] = int(mask.shape[0])
    result["no_flood_fraction"] = float(row_mask.mean()) if row_mask.size else float("nan")
    result["active_fraction"] = (
        float(np.mean(np.abs(selected) > NO_FLOOD_EPSILON_V100))
        if selected.size
        else float("nan")
    )
    return result


def joint_no_flood_fractions_v100(
    delta_depth: np.ndarray,
    delta_volume: np.ndarray,
    delta_flow: np.ndarray,
    delta_flood: np.ndarray,
    *,
    epsilon: float = NO_FLOOD_EPSILON_V100,
) -> dict[str, float]:
    """Compute active-effect AND no-flood fractions.

    Node channels use cell-level denominators (candidate/time/node cells).
    Managed flow uses candidate/time samples and is paired with a time step if
    at least one node has no flooding delta.  This is diagnostic only.
    """
    flood = no_flood_mask_v100(delta_flood, epsilon=epsilon)
    depth_arr = np.asarray(delta_depth, dtype=np.float64)
    volume_arr = np.asarray(delta_volume, dtype=np.float64)
    flow_arr = np.asarray(delta_flow, dtype=np.float64)
    node_denominator = max(int(flood.size), 1)
    flow_no_flood = flood.any(axis=-1) if flood.ndim >= 2 else flood
    flow_denominator = max(int(flow_arr.size), 1)
    return {
        "depth_active_and_flood_inactive": float(np.count_nonzero((np.abs(depth_arr) > epsilon) & flood) / node_denominator),
        "volume_active_and_flood_inactive": float(np.count_nonzero((np.abs(volume_arr) > epsilon) & flood) / node_denominator),
        "flow_active_and_flood_inactive": float(np.count_nonzero((np.abs(flow_arr) > epsilon) & flow_no_flood) / flow_denominator),
        "depth_denominator": node_denominator,
        "volume_denominator": node_denominator,
        "flow_denominator": flow_denominator,
        "depth_numerator": int(np.count_nonzero((np.abs(depth_arr) > epsilon) & flood)),
        "volume_numerator": int(np.count_nonzero((np.abs(volume_arr) > epsilon) & flood)),
        "flow_numerator": int(np.count_nonzero((np.abs(flow_arr) > epsilon) & flow_no_flood)),
    }
