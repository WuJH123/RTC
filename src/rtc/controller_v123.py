"""V12.3 evidence adapter on top of the validated V12.2 closed-loop controller.

V12.2 remains authoritative for execution semantics: sparse causal observation,
target-latch readback, score==execute and first-move verification.  V12.3 only adds a
scientific result schema.  A transparent MPC proxy captures the returned V123 policy
result so PFV/combined-objective diagnostics can be written without changing the action
that V12.2 validated and executed.
"""
from __future__ import annotations

from typing import Any

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step2_policy_v123 import V123_POLICY_CONTRACT

V123_CONTROLLER_CONTRACT = "PROJECT7_V123_TFV_PRIMARY_SOFT_PFV_FINITE_RTC_CONTROLLER_V2"


class _V123ResultCaptureProxy:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_result: Any | None = None

    def optimize(self, *args: Any, **kwargs: Any) -> Any:
        result = self.inner.optimize(*args, **kwargs)
        self.last_result = result
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class V123TorchMPCController(V122TorchMPCController):
    """Reuse V122 execution semantics while emitting truthful V123 evidence labels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mpc = _V123ResultCaptureProxy(self.mpc)

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(
            obs, observation_already_recorded=observation_already_recorded
        )
        diagnostics = dict(action.diagnostics or {})
        diagnostics["v123_controller_contract"] = V123_CONTROLLER_CONTRACT
        diagnostics["v123_policy_contract"] = V123_POLICY_CONTRACT

        result = getattr(self.mpc, "last_result", None)
        if result is not None:
            for name in (
                "predicted_delta_pfv_m3",
                "tfv_risk_m3",
                "pfv_risk_m3",
                "pfv_soft_excess_m3",
                "pfv_penalty_m3_equivalent",
                "objective_score_m3_equivalent",
                "false_benefit_margin_m3",
                "scoring_projection_max",
            ):
                if hasattr(result, name):
                    diagnostics[name] = float(getattr(result, name))
            if hasattr(result, "raw_candidate_count"):
                diagnostics["candidate_count"] = int(result.raw_candidate_count)
            if hasattr(result, "selected_group_score_m3"):
                diagnostics["selected_group_score_m3"] = float(
                    result.selected_group_score_m3
                )

        # Successful V123 finite shooting must not be mislabeled as V122 in evidence.
        # Fallback/PASSIVE sources retain their precise inherited execution reason.
        source = "MPC_V123" if action.source == "MPC_V122" else action.source
        return ControllerAction(
            settings=action.settings,
            source=source,
            diagnostics=diagnostics,
        )


__all__ = ["V123_CONTROLLER_CONTRACT", "V123TorchMPCController"]
