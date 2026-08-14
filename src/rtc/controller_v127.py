"""V127 controller adapter over the validated V122 target-latch execution shell."""
from __future__ import annotations

from typing import Any

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step3_mpc_v127 import V127_STEP3_CONTRACT

V127_CONTROLLER_CONTRACT = "PROJECT7_V127_CONTINUOUS_MPC_TARGET_LATCH_CONTROLLER_V1"


class _ResultCaptureV127:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_result: Any | None = None

    def optimize(self, *args: Any, **kwargs: Any) -> Any:
        result = self.inner.optimize(*args, **kwargs)
        self.last_result = result
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class V127TorchMPCController(V122TorchMPCController):
    """Execute the exact V127 scored first target and report optimizer/fallback truth."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mpc = _ResultCaptureV127(self.mpc)

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        diagnostics = dict(action.diagnostics or {})
        diagnostics["v127_controller_contract"] = V127_CONTROLLER_CONTRACT
        diagnostics["v127_step3_contract"] = V127_STEP3_CONTRACT
        diagnostics["continuous_mpc_scientific_target"] = True
        diagnostics["rbc_role"] = "warm_start_and_safety_fallback_only"
        diagnostics["rbc_is_value_reference"] = False
        diagnostics["rbc_is_action_space_ceiling"] = False
        result = self.mpc.last_result
        if result is not None:
            for name in (
                "continuous_objective_m3",
                "continuous_tfv_risk_m3",
                "continuous_pfv_risk_m3",
                "rbc_objective_m3",
                "rbc_tfv_risk_m3",
                "rbc_pfv_risk_m3",
                "gradient_norm",
                "predicted_delta_tfv_m3",
            ):
                if hasattr(result, name):
                    diagnostics[name] = float(getattr(result, name))
            for name in ("optimisation_steps", "variable_count"):
                if hasattr(result, name):
                    diagnostics[name] = int(getattr(result, name))
            diagnostics["continuous_optimizer_success"] = bool(
                getattr(result, "continuous_optimizer_success", False)
            )
            diagnostics["v127_selected_source"] = str(
                getattr(result, "selected_source", "unknown")
            )
            diagnostics["scipy_message"] = str(getattr(result, "scipy_message", ""))[:2000]

        source = str(action.source)
        if source == "MPC_V122" and result is not None:
            selected = str(getattr(result, "selected_source", ""))
            source = (
                "MPC_V127_CONTINUOUS"
                if selected == "continuous_lbfgsb"
                else "RBC_SAFETY_V127"
                if selected == "rbc_safety_fallback"
                else "MPC_V127"
            )
        elif source.endswith("_V122"):
            source = source[:-5] + "_V127"
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = ["V127_CONTROLLER_CONTRACT", "V127TorchMPCController"]
