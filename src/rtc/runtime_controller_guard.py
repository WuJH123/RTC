from __future__ import annotations

from typing import Mapping

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .runtime import choose_first_move, command_continuity


CONTINUITY_GUARD_CONTRACT = "RTC_SUPERVISORY_TEMPORAL_CONTINUITY_V3_COMMAND_VS_TRACKING"


class ContinuityGuardController:
    """Wrap a Python policy with explicit command and tracking continuity semantics.

    The SWMM ``target_setting`` is the cross-decision supervisory command latch.  Realised
    ``current_setting`` can lag that latch for physically modulated devices.  Historical
    policies can retain the stricter two-anchor rule; V127 and its fixed Python comparators
    set ``enforce_current_delta=False`` so the per-update slew is applied to consecutive
    target commands while current-setting lag is recorded as a hydraulic diagnostic.
    """

    def __init__(
        self,
        controller,
        *,
        max_delta_per_update: float,
        allow_projection: bool,
        enforce_current_delta: bool = True,
    ) -> None:
        if max_delta_per_update < 0:
            raise ValueError("max_delta_per_update must be non-negative")
        self.controller = controller
        self.max_delta_per_update = float(max_delta_per_update)
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
            max_delta_per_update=self.max_delta_per_update,
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
            max_delta_per_update=self.max_delta_per_update,
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
                "max_setting_delta_per_update": self.max_delta_per_update,
            }
        )
        return ControllerAction(
            settings=dict(zip(expected, decision.requested, strict=True)),
            source=action.source,
            diagnostics=diagnostics,
        )

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        return self.decide(obs)
