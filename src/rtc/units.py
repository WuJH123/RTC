from __future__ import annotations

import numpy as np

# Convert any SWMM flow-rate unit to SI m3/s before integrating flood volume.
_FLOW_TO_M3S = {
    "CMS": 1.0,
    "LPS": 1.0e-3,
    "MLD": 1000.0 / 86400.0,
    "CFS": 0.028316846592,
    "GPM": 0.003785411784 / 60.0,
    "MGD": 3785.411784 / 86400.0,
}


def flow_rate_to_m3s(values: np.ndarray | float, flow_units: str) -> np.ndarray:
    """Convert a SWMM flow-rate quantity to m3/s.

    PySWMM exposes rates in the simulation's configured ``flow_units``. Scientific
    volume metrics must therefore never assume that a raw ``Node.flooding`` value is
    already in m3/s.
    """

    units = str(flow_units).upper()
    if units not in _FLOW_TO_M3S:
        raise ValueError(f"unsupported SWMM flow units: {flow_units!r}")
    return np.asarray(values, dtype=float) * _FLOW_TO_M3S[units]


def integrate_piecewise_constant_rate(
    elapsed_seconds: np.ndarray,
    rate: np.ndarray,
    *,
    flow_units: str,
) -> np.ndarray:
    """Integrate sampled rates using the preceding sample over each interval.

    ``elapsed_seconds`` is one-dimensional and ``rate`` has time on axis 0. The
    returned value keeps every non-time dimension. At least two samples are needed.
    """

    t = np.asarray(elapsed_seconds, dtype=float).reshape(-1)
    q = flow_rate_to_m3s(rate, flow_units)
    if q.shape[0] != t.size:
        raise ValueError("time and rate length mismatch")
    if t.size < 2:
        raise ValueError("at least two samples are required for integration")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("elapsed_seconds must be strictly increasing")
    shape = (dt.size,) + (1,) * (q.ndim - 1)
    return (np.clip(q[:-1], 0.0, None) * dt.reshape(shape)).sum(axis=0)
