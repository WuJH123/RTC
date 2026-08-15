from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .runtime import choose_first_move, command_continuity


CONTINUITY_GUARD_CONTRACT = (
    "RTC_SUPERVISORY_TEMPORAL_CONTINUITY_V4_PER_ACTUATOR_COMMAND_ENVELOPE"
)


def _delta_vector(value: float | np.ndarray, actuator_count: int) -> np.ndarray:
    raw = np.asarray(value, dtype=float)
    try:
        delta = np.broadcast_to(raw, (int(actuator_count),)).astype(float, copy=True)
    except ValueError as exc:
        raise ValueError("max_delta_per_update is not scalar or actuator-aligned") from exc
    if not np.isfinite(delta).all() or np.any(delta < 0.0):
        raise ValueError("max_delta_per_update must be finite and non-negative")
    return delta


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class ContinuityGuardController:
    """Wrap a Python policy with explicit per-actuator command continuity semantics.

    The SWMM ``target_setting`` is the cross-decision supervisory command latch. Realised
    ``current_setting`` can lag that latch for physically modulated devices. Historical
    policies can retain the stricter two-anchor rule; V127/V128 and fair Python comparators
    set ``enforce_current_delta=False`` so rate limits apply to consecutive target commands.

    ``max_delta_per_update`` may be a scalar or one value per actuator. This removes the
    previous implicit assumption that every pump/orifice/weir/outlet has the same command
    travel envelope while remaining backward compatible with the frozen 0.5 scalar study.
    """

    def __init__(
        self,
        controller,
        *,
        max_delta_per_update: float | np.ndarray,
        allow_projection: bool,
        enforce_current_delta: bool = True,
    ) -> None:
        raw = np.asarray(max_delta_per_update, dtype=float)
        if raw.ndim > 1 or not np.isfinite(raw).all() or np.any(raw < 0.0):
            raise ValueError("max_delta_per_update must be finite, non-negative scalar/vector")
        self.controller = controller
        self.max_delta_per_update = (
            float(raw) if raw.ndim == 0 else raw.reshape(-1).astype(float, copy=True)
        )
        self.allow_projection = bool(allow_projection)
        self.enforce_current_delta = bool(enforce_current_delta)
        self.previous_requested: np.ndarray | None = None

    def observe(self, obs: CausalObservation) -> None:
        if hasattr(self.controller, "observe"):
            self.controller.observe(obs)

    def _call_inner(
        self, obs: CausalObservation, *, observation_already_recorded: bool
    ) -> ControllerAction:
        if hasattr(self.controller, "decide"):
            raw = self.controller.decide(
                obs, observation_already_recorded=observation_already_recorded
            )
        else:
            raw = self.controller(obs)
        if isinstance(raw, ControllerAction):
            return raw
        if isinstance(raw, Mapping):
            return ControllerAction(settings=raw)
        raise TypeError("controller must return ControllerAction or actuator-setting mapping")

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = self._call_inner(
            obs, observation_already_recorded=observation_already_recorded
        )
        expected = tuple(obs.actuator_ids)
        if set(action.settings) != set(expected):
            raise ValueError("continuity guard requires a complete actuator setting vector")
        raw = np.asarray([float(action.settings[aid]) for aid in expected], dtype=float)
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        active_target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
        if active_target.shape != current.shape:
            raise ValueError("target/current readback shapes differ")
        delta = _delta_vector(self.max_delta_per_update, len(expected))

        previous_write_mismatch = (
            0.0
            if self.previous_requested is None
            else float(np.abs(active_target - self.previous_requested).max(initial=0.0))
        )
        decision = choose_first_move(
            optimized_sequence=raw[None, :],
            surrogate_admissible=True,
            fallback_first_move=active_target,
            current_settings=current,
            previous_requested_settings=active_target,
            min_settings=0.0,
            max_settings=1.0,
            max_delta_per_update=delta,
            enforce_current_delta=self.enforce_current_delta,
        )
        raw_projection = float(np.abs(decision.requested - raw).max(initial=0.0))
        if raw_projection > 1e-9 and not self.allow_projection:
            raise RuntimeError(
                "policy emitted a command outside its declared temporal-continuity contract: "
                f"projection={raw_projection:.12g}, "
                f"current_tracking_delta={np.abs(raw-current).max(initial=0.0):.12g}, "
                f"target_command_delta={np.abs(raw-active_target).max(initial=0.0):.12g}"
            )
        continuity = command_continuity(
            decision.requested,
            current,
            previous_requested_settings=active_target,
            max_delta_per_update=delta,
            enforce_current_delta=self.enforce_current_delta,
        )
        if not continuity.passed:
            raise RuntimeError("supervisory continuity guard failed after projection")

        self.previous_requested = decision.requested.copy()
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "continuity_guard_contract": CONTINUITY_GUARD_CONTRACT,
                "continuity_guard_passed": True,
                "continuity_projection_applied": bool(raw_projection > 1e-9),
                "continuity_raw_to_projected_max": raw_projection,
                "previous_write_target_readback_mismatch_max": previous_write_mismatch,
                "command_delta_from_current_tracking_max": continuity.max_delta_from_current,
                "command_delta_from_previous_target_max": continuity.max_delta_from_previous_command,
                "current_delta_is_hard_constraint": self.enforce_current_delta,
                "max_setting_delta_per_update_min": float(delta.min(initial=0.0)),
                "max_setting_delta_per_update_max": float(delta.max(initial=0.0)),
                "max_setting_delta_is_per_actuator": bool(np.ptp(delta) > 1e-12),
                "max_setting_delta_vector_sha256": _array_sha256(delta),
            }
        )
        return ControllerAction(
            settings=dict(zip(expected, decision.requested, strict=True)),
            source=action.source,
            diagnostics=diagnostics,
        )

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        return self.decide(obs)
