"""Pure helpers for the V12.2 passive target-latch contract.

The supervisory target is the command state that is held when no new command is
accepted.  The realised current setting is intentionally not used as a fallback
command anchor because a physical actuator may lag a successfully written target.
"""
from __future__ import annotations

import numpy as np

V122_PASSIVE_REFERENCE_CONTRACT = (
    "PROJECT7_V122_PASSIVE_TARGET_LATCH_REFERENCE_V1"
)


def passive_target_latch_v122(
    target_setting: np.ndarray, horizon_steps: int
) -> np.ndarray:
    """Repeat a validated supervisory target over the prediction horizon."""

    target = np.asarray(target_setting, dtype=np.float64).reshape(-1)
    if target.size == 0 or int(horizon_steps) <= 0:
        raise ValueError("V122 passive target must be non-empty with positive horizon")
    if not np.isfinite(target).all() or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("V122 passive target is outside [0, 1] or non-finite")
    return np.repeat(target[None, :], int(horizon_steps), axis=0)


def assert_passive_command_sequence_v122(
    commanded_setting: np.ndarray,
    active_target_setting: np.ndarray,
    *,
    atol: float = 1.0e-9,
) -> None:
    """Fail closed unless an old HOLD sequence really held its active target."""

    commands = np.asarray(commanded_setting, dtype=np.float64)
    target = np.asarray(active_target_setting, dtype=np.float64).reshape(-1)
    if commands.ndim != 2 or commands.shape[0] == 0 or commands.shape[1] != target.size:
        raise ValueError("V122 passive command sequence has incompatible shape")
    if not np.isfinite(commands).all() or not np.isfinite(target).all():
        raise ValueError("V122 passive command sequence contains non-finite values")
    if not np.allclose(commands, target[None, :], rtol=0.0, atol=float(atol)):
        raise ValueError("V122 HOLD commands do not equal the active target latch")


__all__ = [
    "V122_PASSIVE_REFERENCE_CONTRACT",
    "assert_passive_command_sequence_v122",
    "passive_target_latch_v122",
]
