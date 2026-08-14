"""V12.3 evidence adapter on top of the validated V12.2 closed-loop controller.

V12.2 already owns the difficult execution semantics: sparse causal observation,
target-latch readback, score==execute and first-move verification.  V12.3 changes the
scientific policy result schema, not those hydraulic execution rules.  This adapter
therefore only relabels successful V123 MPC actions and exposes the V123 controller
contract while preserving all V12.2 execution diagnostics.
"""
from __future__ import annotations

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step2_policy_v123 import V123_POLICY_CONTRACT

V123_CONTROLLER_CONTRACT = "PROJECT7_V123_TFV_PRIMARY_SOFT_PFV_FINITE_RTC_CONTROLLER_V1"


class V123TorchMPCController(V122TorchMPCController):
    """Reuse V122 execution semantics while emitting truthful V123 evidence labels."""

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(
            obs, observation_already_recorded=observation_already_recorded
        )
        diagnostics = dict(action.diagnostics or {})
        diagnostics["v123_controller_contract"] = V123_CONTROLLER_CONTRACT
        diagnostics["v123_policy_contract"] = V123_POLICY_CONTRACT

        # Successful V123 finite shooting must not be mislabeled as V122 in evidence.
        # Fallback/PASSIVE sources retain their precise V122 execution reason because
        # those semantics are inherited unchanged.
        source = "MPC_V123" if action.source == "MPC_V122" else action.source
        return ControllerAction(
            settings=action.settings,
            source=source,
            diagnostics=diagnostics,
        )


__all__ = ["V123_CONTROLLER_CONTRACT", "V123TorchMPCController"]
