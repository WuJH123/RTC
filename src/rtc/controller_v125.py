"""V125 evidence adapter over the validated V122/V123 rolling execution shell."""
from __future__ import annotations

from .closed_loop import CausalObservation, ControllerAction
from .controller_v123 import V123TorchMPCController
from .step2_policy_v125 import V125_POLICY_CONTRACT

V125_CONTROLLER_CONTRACT = "PROJECT7_V125_10MIN_DIRECT_ANCHOR_ADVANTAGE_CONTROLLER_V2"


class V125TorchMPCController(V123TorchMPCController):
    """Keep V122 score==execute/readback semantics and expose V125 decision evidence."""

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        # V122 keeps the optimizer result local. Capture exactly the same call so V125
        # can record anchor/override evidence without duplicating or changing execution.
        captured: dict[str, object] = {}
        original_optimize = self.mpc.optimize

        def recording_optimize(*args, **kwargs):
            result = original_optimize(*args, **kwargs)
            captured["result"] = result
            return result

        self.mpc.optimize = recording_optimize
        try:
            action = super().decide(
                obs, observation_already_recorded=observation_already_recorded
            )
        finally:
            self.mpc.optimize = original_optimize

        diagnostics = dict(action.diagnostics or {})
        diagnostics.pop("v123_controller_contract", None)
        diagnostics.pop("v123_policy_contract", None)
        diagnostics.pop("v123_policy_mode", None)
        diagnostics.pop("v123_policy_mode_contract", None)
        diagnostics["v125_controller_contract"] = V125_CONTROLLER_CONTRACT
        diagnostics["v125_policy_contract"] = V125_POLICY_CONTRACT
        diagnostics["v125_policy_mode"] = "direct_anchor_relative"
        diagnostics["knowledge_data_fusion"] = True
        diagnostics["value_reference"] = "causal_sparse_rbc_anchor"

        result = captured.get("result")
        if result is not None:
            for name in (
                "anchor_tfv_risk_m3",
                "anchor_pfv_risk_m3",
                "anchor_objective_score_m3",
                "learned_tfv_risk_m3",
                "learned_pfv_risk_m3",
                "learned_objective_score_m3",
                "predicted_override_advantage_tfv_m3",
                "anchor_override_margin_m3",
                "knowledge_anchor_first_move_delta_max",
            ):
                if hasattr(result, name):
                    diagnostics[name] = float(getattr(result, name))
            if hasattr(result, "learned_override_admitted"):
                diagnostics["learned_override_admitted"] = bool(
                    getattr(result, "learned_override_admitted")
                )
            if hasattr(result, "selected_source"):
                diagnostics["v125_selected_source"] = str(
                    getattr(result, "selected_source")
                )
            if hasattr(result, "selected_candidate_family"):
                diagnostics["v125_selected_candidate_family"] = str(
                    getattr(result, "selected_candidate_family")
                )
            diagnostics["v125_anchor_selected"] = bool(
                getattr(result, "knowledge_anchor_selected", False)
            )

        source = str(action.source)
        if source in {"MPC_V122", "MPC_V123"}:
            source = "MPC_V125"
        elif source.endswith("_V123"):
            source = source[:-5] + "_V125"
        elif source.endswith("_V122"):
            source = source[:-5] + "_V125"
        return ControllerAction(
            settings=action.settings, source=source, diagnostics=diagnostics
        )


__all__ = ["V125_CONTROLLER_CONTRACT", "V125TorchMPCController"]
