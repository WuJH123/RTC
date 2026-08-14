"""V123 evidence adapter over the validated V122 execution controller.

V122 remains authoritative for sparse causal observation, target-latch readback,
continuity, and first-move execution.  This adapter captures the V123 TFV/PFV policy
result, including whether the low-sensor engineering anchor or a learned override was
executed, without changing the scored action.
"""
from __future__ import annotations

from typing import Any

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step2_policy_v123 import V123_POLICY_CONTRACT


V123_CONTROLLER_CONTRACT = "PROJECT7_V123_KNOWLEDGE_ANCHORED_FINITE_RTC_CONTROLLER_V3"


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
    """Reuse V122 execution semantics while emitting truthful V123 diagnostics."""

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
        policy_mode = getattr(self.mpc, "policy_mode", None)
        policy_mode_contract = getattr(self.mpc, "policy_mode_contract", None)
        if policy_mode is not None:
            diagnostics["v123_policy_mode"] = str(policy_mode)
        if policy_mode_contract is not None:
            diagnostics["v123_policy_mode_contract"] = str(policy_mode_contract)

        result = getattr(self.mpc, "last_result", None)
        if result is not None:
            if hasattr(result, "policy_mode"):
                diagnostics["v123_policy_mode"] = str(getattr(result, "policy_mode"))
            if hasattr(result, "policy_mode_contract"):
                diagnostics["v123_policy_mode_contract"] = str(
                    getattr(result, "policy_mode_contract")
                )
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
                "knowledge_anchor_first_move_delta_max",
            ):
                if hasattr(result, name):
                    diagnostics[name] = float(getattr(result, name))
            for name in (
                "raw_candidate_count",
                "first_move_group_count",
                "tail_only_noop_candidate_count",
                "knowledge_anchor_candidate_index",
            ):
                if hasattr(result, name):
                    key = "candidate_count" if name == "raw_candidate_count" else name
                    diagnostics[key] = int(getattr(result, name))
            for name in (
                "knowledge_anchor_selected",
                "knowledge_anchor_fallback_used",
            ):
                if hasattr(result, name):
                    diagnostics[name] = bool(getattr(result, name))
            anchor_enabled = getattr(self.mpc, "use_sparse_rbc_anchor", None)
            if anchor_enabled is None and hasattr(result, "policy_mode"):
                anchor_enabled = str(getattr(result, "policy_mode")) != "learned_only"
            diagnostics["knowledge_data_fusion"] = bool(anchor_enabled)

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
