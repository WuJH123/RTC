from __future__ import annotations

import numpy as np

_FLOW_TO_M3S = {
    "CMS": 1.0,
    "LPS": 1.0e-3,
    "MLD": 1000.0 / 86400.0,
    "CFS": 0.028316846592,
    "GPM": 0.003785411784 / 60.0,
    "MGD": 3785.411784 / 86400.0,
}


def flow_rate_to_m3s(values: np.ndarray | float, flow_units: str) -> np.ndarray:
    units = str(flow_units).upper()
    if units not in _FLOW_TO_M3S:
        raise ValueError(f"unsupported SWMM flow units: {flow_units!r}")
    return np.asarray(values, dtype=float) * _FLOW_TO_M3S[units]


def length_to_m(values: np.ndarray | float, system_units: str) -> np.ndarray:
    units = str(system_units).upper()
    if units == "SI":
        factor = 1.0
    elif units == "US":
        factor = 0.3048
    else:
        raise ValueError(f"unsupported SWMM system units: {system_units!r}")
    return np.asarray(values, dtype=float) * factor


def volume_to_m3(values: np.ndarray | float, system_units: str) -> np.ndarray:
    units = str(system_units).upper()
    if units == "SI":
        factor = 1.0
    elif units == "US":
        factor = 0.028316846592
    else:
        raise ValueError(f"unsupported SWMM system units: {system_units!r}")
    return np.asarray(values, dtype=float) * factor


def rainfall_rate_to_mmhr(values: np.ndarray | float, system_units: str) -> np.ndarray:
    units = str(system_units).upper()
    if units == "SI":
        factor = 1.0
    elif units == "US":
        factor = 25.4
    else:
        raise ValueError(f"unsupported SWMM system units: {system_units!r}")
    return np.asarray(values, dtype=float) * factor


def integrate_piecewise_constant_rate(
    elapsed_seconds: np.ndarray,
    rate: np.ndarray,
    *,
    flow_units: str,
) -> np.ndarray:
    """Integrate sampled rates using the preceding sample over each interval."""

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
