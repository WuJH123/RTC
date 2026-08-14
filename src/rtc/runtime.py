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


@dataclass(frozen=True)
class CommandContinuityResult:
    passed: bool
    max_delta_from_current: float
    max_delta_from_previous_command: float
    failed_current_indices: tuple[int, ...]
    failed_previous_indices: tuple[int, ...]


def command_continuity(
    requested: np.ndarray,
    current_settings: np.ndarray,
    *,
    previous_requested_settings: np.ndarray | None,
    max_delta_per_update: np.ndarray | float | None,
    enforce_current_delta: bool = True,
    tolerance: float = 1e-9,
) -> CommandContinuityResult:
    """Audit one supervisory command while separating command slew from tracking lag.

    ``previous_requested_settings`` is the supervisory target latch and is the natural
    anchor for a per-command slew/rate contract. ``current_settings`` is the realised SWMM
    device state; for pumps/regulators it can legitimately lag the target.  Historical
    callers retain ``enforce_current_delta=True``.  V127 and fair rule baselines set it to
    ``False`` so physical tracking lag is reported but does not create an artificial second
    command constraint.
    """
    requested = np.asarray(requested, dtype=float).reshape(-1)
    current = np.asarray(current_settings, dtype=float).reshape(-1)
    if requested.shape != current.shape:
        raise ValueError("requested/current command shapes differ")
    if not np.isfinite(requested).all() or not np.isfinite(current).all():
        raise ValueError("requested/current settings must be finite")

    previous: np.ndarray | None = None
    if previous_requested_settings is not None:
        previous = np.asarray(previous_requested_settings, dtype=float).reshape(-1)
        if previous.shape != current.shape or not np.isfinite(previous).all():
            raise ValueError("previous requested settings are invalid")

    current_error = np.abs(requested - current)
    previous_error = (
        np.zeros_like(current_error) if previous is None else np.abs(requested - previous)
    )
    if max_delta_per_update is None:
        return CommandContinuityResult(
            passed=True,
            max_delta_from_current=float(current_error.max(initial=0.0)),
            max_delta_from_previous_command=float(previous_error.max(initial=0.0)),
            failed_current_indices=(),
            failed_previous_indices=(),
        )

    delta = np.broadcast_to(np.asarray(max_delta_per_update, dtype=float), current.shape)
    if np.any(delta < 0) or not np.isfinite(delta).all():
        raise ValueError("max_delta_per_update must be finite and non-negative")
    failed_current = (
        np.flatnonzero(current_error > delta + tolerance)
        if enforce_current_delta
        else np.asarray([], dtype=int)
    )
    failed_previous = (
        np.asarray([], dtype=int)
        if previous is None
        else np.flatnonzero(previous_error > delta + tolerance)
    )
    return CommandContinuityResult(
        passed=bool(failed_current.size == 0 and failed_previous.size == 0),
        max_delta_from_current=float(current_error.max(initial=0.0)),
        max_delta_from_previous_command=float(previous_error.max(initial=0.0)),
        failed_current_indices=tuple(int(i) for i in failed_current),
        failed_previous_indices=tuple(int(i) for i in failed_previous),
    )


def choose_first_move(
    *,
    optimized_sequence: np.ndarray,
    surrogate_admissible: bool,
    fallback_first_move: np.ndarray,
    current_settings: np.ndarray,
    previous_requested_settings: np.ndarray | None = None,
    min_settings: np.ndarray | float = 0.0,
    max_settings: np.ndarray | float = 1.0,
    max_delta_per_update: np.ndarray | float | None = None,
    enforce_current_delta: bool = True,
) -> ExecutableDecision:
    """Choose/project the first move under the requested command-continuity semantics."""
    candidate = np.asarray(optimized_sequence, dtype=float)
    fallback = np.asarray(fallback_first_move, dtype=float).reshape(-1)
    current = np.asarray(current_settings, dtype=float).reshape(-1)
    previous = (
        None
        if previous_requested_settings is None
        else np.asarray(previous_requested_settings, dtype=float).reshape(-1)
    )
    if candidate.ndim != 2 or candidate.shape[1] != current.size:
        raise ValueError("optimized_sequence must be [horizon, actuator]")
    if fallback.shape != current.shape:
        raise ValueError("fallback/current shape mismatch")
    if previous is not None and previous.shape != current.shape:
        raise ValueError("previous requested/current shape mismatch")

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
        if np.any(delta < 0) or not np.isfinite(delta).all():
            raise ValueError("max_delta_per_update must be finite and non-negative")
        lower, upper = lo.copy(), hi.copy()
        if enforce_current_delta:
            lower = np.maximum(lower, current - delta)
            upper = np.minimum(upper, current + delta)
        if previous is not None:
            lower = np.maximum(lower, previous - delta)
            upper = np.minimum(upper, previous + delta)
        if np.any(lower > upper + 1e-12):
            raise ValueError("command anchors leave no feasible continuous setting interval")
        requested = np.minimum(np.maximum(requested, lower), upper)
    projected = not np.allclose(before, requested, rtol=0.0, atol=1e-12)
    continuity = command_continuity(
        requested,
        current,
        previous_requested_settings=previous,
        max_delta_per_update=max_delta_per_update,
        enforce_current_delta=enforce_current_delta,
    )
    if not continuity.passed:
        raise RuntimeError("first-move continuity projection failed")
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
    """Verify target-write acceptance and report realised device tracking separately.

    The returned historical ``passed`` field retains both tolerances for compatibility.
    V127 runtime uses exact target-latch readback as write authority and treats current
    setting error as hydraulic tracking diagnostics, not as proof that the command failed.
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
