"""Fail-closed integrity primitives for Direct-TFV same-prefix SWMM replay.

The authoritative closed loop reasserts the previously issued supervisory target after every SWMM
routing callback and before the next causal observation. A counterfactual replay that omits that
step can observe SWMM's internal pump-state target instead of the Python supervisory latch and create
a false prefix/HOLD mismatch. Replays must call this module rather than reimplement target timing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .closed_loop import _reassert_target_latch


DIRECT_TFV_REPLAY_TARGET_LATCH_CONTRACT = (
    "PROJECT7_DIRECT_TFV_REPLAY_USES_AUTHORITATIVE_TARGET_LATCH_V1"
)


def reassert_authoritative_target_latch(
    link_obj: Mapping[str, Any], held_settings: Mapping[str, float] | None
) -> float:
    """Apply the exact target-latch operation used by ``run_authoritative_closed_loop``."""

    return float(_reassert_target_latch(link_obj, held_settings))


def maximum_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(tuple(left), dtype=np.float64).reshape(-1)
    b = np.asarray(tuple(right), dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"counterfactual prefix shapes differ: {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("counterfactual prefix contains non-finite values")
    return float(np.abs(a - b).max(initial=0.0))


def require_identical_prefix(
    *,
    candidate_state: Sequence[float],
    hold_state: Sequence[float],
    candidate_target: Sequence[float],
    hold_target: Sequence[float],
    candidate_current: Sequence[float],
    hold_current: Sequence[float],
    candidate_statistics: Sequence[float],
    hold_statistics: Sequence[float],
    hold_reference_target: Sequence[float],
    tolerance: float = 1.0e-9,
) -> dict[str, float | str]:
    """Require branch equality and HOLD-reference equality at decision time before H360 replay."""

    if tolerance < 0.0:
        raise ValueError("counterfactual prefix tolerance must be non-negative")
    diagnostics = {
        "prefix_state_max_abs_difference": maximum_abs_difference(candidate_state, hold_state),
        "prefix_target_max_abs_difference": maximum_abs_difference(candidate_target, hold_target),
        "prefix_current_max_abs_difference": maximum_abs_difference(candidate_current, hold_current),
        "prefix_statistics_max_abs_difference": maximum_abs_difference(
            candidate_statistics, hold_statistics
        ),
        "hold_reference_target_max_abs_difference": maximum_abs_difference(
            hold_target, hold_reference_target
        ),
    }
    bad = {key: value for key, value in diagnostics.items() if float(value) > float(tolerance)}
    if bad:
        raise RuntimeError(
            "COUNTERFACTUAL_REPLAY_P0: same-prefix or HOLD-reference mismatch after authoritative "
            f"target-latch reassertion: {bad}"
        )
    return {
        "contract": DIRECT_TFV_REPLAY_TARGET_LATCH_CONTRACT,
        **diagnostics,
    }


__all__ = [
    "DIRECT_TFV_REPLAY_TARGET_LATCH_CONTRACT",
    "maximum_abs_difference",
    "reassert_authoritative_target_latch",
    "require_identical_prefix",
]
