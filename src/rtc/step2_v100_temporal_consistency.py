"""Fixed temporal slope/bucket helpers for the V10 post-training diagnostic."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

HORIZON_BUCKETS_V100 = ("0_30_min", "30_120_min", "120_360_min")


def slope_pair_v100(values: np.ndarray, timestamps_seconds: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if values.ndim < 1 or timestamps.ndim != 1 or values.shape[-1] != timestamps.size:
        raise ValueError("values must end with the timestamp dimension")
    dt = np.diff(timestamps)
    if dt.size == 0 or not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return np.diff(values, axis=-1) / dt.reshape((1,) * (values.ndim - 1) + (dt.size,))


def horizon_bucket_indices_v100(retained_indices: np.ndarray, timestamps_seconds: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.asarray(retained_indices, dtype=np.int64)
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if indices.ndim != 1 or timestamps.shape != indices.shape or indices.size < 2:
        raise ValueError("retained indices and timestamps must be aligned vectors")
    slope_times = timestamps[1:]
    minute = slope_times / 60.0
    masks = OrderedDict((
        ("0_30_min", (minute > 0.0) & (minute <= 30.0)),
        ("30_120_min", (minute > 30.0) & (minute <= 120.0)),
        ("120_360_min", (minute > 120.0) & (minute <= 360.0)),
    ))
    result = OrderedDict((name, np.flatnonzero(mask)) for name, mask in masks.items())
    joined = np.concatenate(tuple(result.values()))
    if not np.array_equal(joined, np.arange(indices.size - 1)):
        raise RuntimeError("temporal slope buckets are not exhaustive/disjoint")
    return result

