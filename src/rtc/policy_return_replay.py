"""Offline controller wrapper for paired receding-policy return SWMM branches."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .closed_loop import CausalObservation, ControllerAction


POLICY_RETURN_REPLAY_CONTRACT = "PROJECT7_RECEDING_POLICY_RETURN_EXACT_PREFIX_REPLAY_V1"


def _sync_supervisory_latch(controller: object, target: np.ndarray) -> None:
    """Synchronize known controller wrappers after an externally replayed target command.

    This is used only by offline authoritative replay.  It does not alter online policy logic; it
    makes the continuation controller's readback bookkeeping match the target actually written by
    the replay wrapper before the continuation policy takes control.
    """
    value = np.asarray(target, dtype=float).copy()
    if hasattr(controller, "last_requested"):
        setattr(controller, "last_requested", value.copy())
    if hasattr(controller, "previous_requested"):
        setattr(controller, "previous_requested", value.copy())
    inner = getattr(controller, "controller", None)
    if inner is not None and inner is not controller:
        _sync_supervisory_latch(inner, value)


class ExactPrefixThenFrozenPolicyController:
    """Replay an exact recorded prefix, inject one first action, then resume a frozen policy."""

    def __init__(
        self,
        *,
        delegate: object,
        actuator_ids: tuple[str, ...],
        prefix_actions: Mapping[int, Mapping[str, float]],
        branch_elapsed_seconds: int,
        branch_target: Mapping[str, float],
        branch_kind: str,
    ) -> None:
        if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
            raise ValueError("policy-return replay requires 109 unique actuator IDs")
        self.delegate = delegate
        self.actuator_ids = tuple(actuator_ids)
        self.prefix_actions = {
            int(elapsed): {aid: float(settings[aid]) for aid in self.actuator_ids}
            for elapsed, settings in prefix_actions.items()
        }
        self.branch_elapsed_seconds = int(branch_elapsed_seconds)
        if self.branch_elapsed_seconds < 0:
            raise ValueError("branch elapsed time must be non-negative")
        self.branch_target = {aid: float(branch_target[aid]) for aid in self.actuator_ids}
        if branch_kind not in {"CANDIDATE", "HOLD"}:
            raise ValueError("branch_kind must be CANDIDATE or HOLD")
        self.branch_kind = branch_kind
        self.continuation_started = False

    def observe(self, obs: CausalObservation) -> None:
        if hasattr(self.delegate, "observe"):
            self.delegate.observe(obs)

    def _external_action(self, target: Mapping[str, float], source: str) -> ControllerAction:
        vector = np.asarray([float(target[aid]) for aid in self.actuator_ids], dtype=float)
        if not np.isfinite(vector).all() or np.any((vector < 0.0) | (vector > 1.0)):
            raise ValueError("policy-return replay target is outside [0,1]")
        _sync_supervisory_latch(self.delegate, vector)
        return ControllerAction(
            settings={aid: float(target[aid]) for aid in self.actuator_ids},
            source=source,
            diagnostics={
                "policy_return_replay_contract": POLICY_RETURN_REPLAY_CONTRACT,
                "policy_return_branch_kind": self.branch_kind,
                "policy_return_external_replay": True,
                "policy_return_continuation_started": False,
            },
        )

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        elapsed = int(obs.elapsed_seconds)
        if elapsed < self.branch_elapsed_seconds:
            if elapsed not in self.prefix_actions:
                raise RuntimeError(f"policy-return replay lacks recorded prefix action at {elapsed}s")
            return self._external_action(
                self.prefix_actions[elapsed], "POLICY_RETURN_EXACT_PREFIX_REPLAY"
            )
        if elapsed == self.branch_elapsed_seconds:
            return self._external_action(
                self.branch_target,
                f"POLICY_RETURN_{self.branch_kind}_FIRST_ACTION",
            )
        self.continuation_started = True
        if not hasattr(self.delegate, "decide"):
            raise TypeError("policy-return continuation delegate lacks decide()")
        action = self.delegate.decide(
            obs, observation_already_recorded=observation_already_recorded
        )
        if not isinstance(action, ControllerAction):
            raise TypeError("policy-return continuation delegate must return ControllerAction")
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "policy_return_replay_contract": POLICY_RETURN_REPLAY_CONTRACT,
                "policy_return_branch_kind": self.branch_kind,
                "policy_return_external_replay": False,
                "policy_return_continuation_started": True,
            }
        )
        return ControllerAction(settings=action.settings, source=action.source, diagnostics=diagnostics)


__all__ = [
    "POLICY_RETURN_REPLAY_CONTRACT",
    "ExactPrefixThenFrozenPolicyController",
]
