from __future__ import annotations

from typing import Mapping

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .runtime import choose_first_move, command_continuity


CONTINUITY_GUARD_CONTRACT = "RTC_SUPERVISORY_TEMPORAL_CONTINUITY_V1"


class ContinuityGuardController:
    """Wrap a Python policy with cross-decision setting continuity.

    For Proposed, ``allow_projection`` should be False because the MPC controller itself is
    required to emit an already executable first move; any mismatch is a software error. For
    deterministic rule/diagnostic baselines, projection may be enabled so that an idealized
    target (for example All-open) is approached through the same physical per-update movement
    envelope instead of jumping instantaneously.
    """

    def __init__(
        self,
        controller,
        *,
        max_delta_per_update: float,
        allow_projection: bool,
    ) -> None:
        if max_delta_per_update < 0:
            raise ValueError("max_delta_per_update must be non-negative")
        self.controller = controller
        self.max_delta_per_update = float(max_delta_per_update)
        self.allow_projection = bool(allow_projection)
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
        decision = choose_first_move(
            optimized_sequence=raw[None, :],
            surrogate_admissible=True,
            fallback_first_move=current,
            current_settings=current,
            previous_requested_settings=self.previous_requested,
            min_settings=0.0,
            max_settings=1.0,
            max_delta_per_update=self.max_delta_per_update,
        )
        raw_projection = float(np.abs(decision.requested - raw).max(initial=0.0))
        if raw_projection > 1e-9 and not self.allow_projection:
            raise RuntimeError(
                "Proposed emitted a command that violates the frozen temporal-continuity contract"
            )
        continuity = command_continuity(
            decision.requested,
            current,
            previous_requested_settings=self.previous_requested,
            max_delta_per_update=self.max_delta_per_update,
        )
        if not continuity.passed:
            raise RuntimeError("supervisory continuity guard failed after projection")

        previous = self.previous_requested
        self.previous_requested = decision.requested.copy()
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "continuity_guard_contract": CONTINUITY_GUARD_CONTRACT,
                "continuity_guard_passed": True,
                "continuity_projection_applied": bool(raw_projection > 1e-9),
                "continuity_raw_to_projected_max": raw_projection,
                "command_delta_from_current_max": continuity.max_delta_from_current,
                "command_delta_from_previous_target_max": (
                    0.0
                    if previous is None
                    else continuity.max_delta_from_previous_command
                ),
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
