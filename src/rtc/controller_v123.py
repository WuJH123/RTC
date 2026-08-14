"""V123 evidence adapter over the validated V122 execution controller.

V122 remains authoritative for sparse causal observation, target-latch readback,
continuity, and first-move execution.  This adapter only captures the V123
TFV/PFV policy result and emits a truthful V123 runtime evidence schema.
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
    """Reuse V122 execution semantics while emitting V123 diagnostics."""

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
        for key in (
            "v122_controller_contract",
            "v122_step3_contract",
            "v122_readback_contract",
        ):
            diagnostics.pop(key, None)
        diagnostics["v123_controller_contract"] = V123_CONTROLLER_CONTRACT
        diagnostics["v123_policy_contract"] = V123_POLICY_CONTRACT

        result = getattr(self.mpc, "last_result", None)
        if result is not None:
            for name in (
                "predicted_delta_tfv_m3",
                "predicted_delta_pfv_m3",
                "tfv_risk_m3",
                "pfv_risk_m3",
                "pfv_soft_excess_m3",
                "pfv_penalty_m3_equivalent",
                "objective_score_m3_equivalent",
                "selected_group_score_m3",
                "false_benefit_margin_m3",
                "scoring_projection_max",
            ):
                if hasattr(result, name):
                    diagnostics[name] = float(getattr(result, name))
            if hasattr(result, "raw_candidate_count"):
                diagnostics["candidate_count"] = int(result.raw_candidate_count)
            if hasattr(result, "first_move_group_count"):
                diagnostics["first_move_group_count"] = int(result.first_move_group_count)
            if hasattr(result, "tail_only_noop_candidate_count"):
                diagnostics["tail_only_noop_candidate_count"] = int(
                    result.tail_only_noop_candidate_count
                )

        source = str(action.source)
        if source == "MPC_V122":
            source = "MPC_V123"
        elif source.endswith("_V122"):
            source = source[:-5] + "_V123"
        return ControllerAction(
            settings=action.settings,
            source=source,
            diagnostics=diagnostics,
        )


__all__ = ["V123_CONTROLLER_CONTRACT", "V123TorchMPCController"]
