from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutableDecision:
    requested: np.ndarray
    source: str
    surrogate_admissible: bool
    projected: bool


@dataclass(frozen=True)
class ReadbackResult:
    passed: bool
    max_target_error: float
    max_current_error: float
    failed_indices: tuple[int, ...]


def choose_first_move(
    *,
    optimized_sequence: np.ndarray,
    surrogate_admissible: bool,
    fallback_first_move: np.ndarray,
    current_settings: np.ndarray,
    min_settings: np.ndarray | float = 0.0,
    max_settings: np.ndarray | float = 1.0,
    max_delta_per_update: np.ndarray | float | None = None,
) -> ExecutableDecision:
    """Fail closed: an inadmissible/invalid MPC result executes fallback, never the candidate."""

    candidate = np.asarray(optimized_sequence, dtype=float)
    fallback = np.asarray(fallback_first_move, dtype=float).reshape(-1)
    current = np.asarray(current_settings, dtype=float).reshape(-1)
    if candidate.ndim != 2 or candidate.shape[1] != current.size:
        raise ValueError("optimized_sequence must be [horizon, actuator]")
    if fallback.shape != current.shape:
        raise ValueError("fallback/current shape mismatch")

    use_candidate = bool(surrogate_admissible and np.isfinite(candidate[0]).all())
    requested = candidate[0].copy() if use_candidate else fallback.copy()
    source = "MPC" if use_candidate else "FALLBACK"

    lo = np.broadcast_to(np.asarray(min_settings, dtype=float), current.shape)
    hi = np.broadcast_to(np.asarray(max_settings, dtype=float), current.shape)
    if np.any(lo > hi):
        raise ValueError("min setting exceeds max setting")
    before = requested.copy()
    requested = np.clip(requested, lo, hi)
    if max_delta_per_update is not None:
        delta = np.broadcast_to(np.asarray(max_delta_per_update, dtype=float), current.shape)
        if np.any(delta < 0):
            raise ValueError("max_delta_per_update must be non-negative")
        requested = np.clip(requested, current - delta, current + delta)
        requested = np.clip(requested, lo, hi)
    projected = not np.allclose(before, requested, rtol=0.0, atol=1e-12)
    return ExecutableDecision(
        requested=requested,
        source=source,
        surrogate_admissible=bool(surrogate_admissible),
        projected=projected,
    )


def verify_setting_readback(
    requested: np.ndarray,
    target_setting: np.ndarray,
    current_setting: np.ndarray,
    *,
    target_tolerance: float = 1e-6,
    current_tolerance: float = 0.05,
) -> ReadbackResult:
    """Verify the write path separately from actuator hydraulic response.

    Target-setting mismatch indicates the command was not accepted. Current-setting
    mismatch can legitimately lag for a modulated device, so its tolerance is wider and
    should be frozen from Phase-0 readback experiments.
    """

    req = np.asarray(requested, dtype=float).reshape(-1)
    target = np.asarray(target_setting, dtype=float).reshape(-1)
    current = np.asarray(current_setting, dtype=float).reshape(-1)
    if req.shape != target.shape or req.shape != current.shape:
        raise ValueError("readback arrays must have identical shape")
    target_error = np.abs(target - req)
    current_error = np.abs(current - req)
    failed = np.where((target_error > target_tolerance) | (current_error > current_tolerance))[0]
    return ReadbackResult(
        passed=bool(failed.size == 0),
        max_target_error=float(target_error.max(initial=0.0)),
        max_current_error=float(current_error.max(initial=0.0)),
        failed_indices=tuple(int(i) for i in failed),
    )
