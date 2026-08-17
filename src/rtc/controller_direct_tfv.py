"""Authoritative-runtime adapter for the current core Direct-TFV receding MPC.

The mature Project7 target-latch controller provides causal Step1 history, rainfall forecast,
600-s decision timing, target-write verification and score-equals-execute semantics. This module
uses that execution shell while the policy itself is the current all-109 Direct-TFV receding MPC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step3_tfv_value_mpc_v3 import DIRECT_TFV_STEP3_CONTRACT, DirectTFVRecedingMPC


DIRECT_TFV_CONTROLLER_CONTRACT = "PROJECT7_DIRECT_TFV_TARGET_LATCH_CONTROLLER_V2"


@dataclass(frozen=True)
class _RuntimeMPCResult:
    settings: torch.Tensor
    predicted_delta_tfv_m3: float
    candidate_valid: bool
    selected_source: str
    optimizer_success: bool
    optimizer_steps: int
    optimizer_starts: int
    gradient_norm: float
    solver_elapsed_seconds: float
    screened_facility_count: int
    predicted_beneficial_facility_count: int
    active_facility_count: int
    first_move_changed_facility_count: int
    maximum_support_ratio: float
    scipy_message: str


class DirectTFVRuntimeMPCAdapter:
    """Match the validated target-latch controller's historical MPC call signature."""

    def __init__(self, inner: DirectTFVRecedingMPC) -> None:
        self.inner = inner
        self.last_result: _RuntimeMPCResult | None = None

    @property
    def model(self) -> torch.nn.Module:
        """Expose the inner value model required by ``TorchMPCController`` initialisation.

        The shared target-latch controller moves ``mpc.model`` to its runtime device and switches it
        to eval mode during construction. Keep this compatibility surface explicit rather than
        forwarding arbitrary historical MPC attributes.
        """

        return self.inner.model

    def optimize(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        current_settings: torch.Tensor,
        previous_requested_settings: torch.Tensor,
        fallback_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        max_delta_per_update: float | None,
    ) -> _RuntimeMPCResult:
        # A failed inner solve is converted to a runtime fallback by the target-latch controller.
        # Clear the previous solve first so that such a fallback can never inherit stale Step3
        # diagnostics from an earlier successful decision.
        self.last_result = None
        del fallback_settings
        if max_delta_per_update is not None and abs(
            float(max_delta_per_update) - float(self.inner.design.max_setting_delta_per_update)
        ) > 1.0e-9:
            raise ValueError("runtime controller max-delta differs from Direct-TFV Step3 design")
        result = self.inner.optimize(
            current_state=initial_state,
            rainfall=rainfall_scenarios,
            previous_actuator_flow=previous_actuator_flow,
            current_settings=current_settings,
            active_target=previous_requested_settings,
        )
        valid = result.selected_source == "DIRECT_TFV_RECEDING_LBFGSB"
        wrapped = _RuntimeMPCResult(
            settings=result.settings,
            predicted_delta_tfv_m3=float(result.predicted_delta_tfv_m3),
            candidate_valid=bool(valid),
            selected_source=str(result.selected_source),
            optimizer_success=bool(result.optimizer_success),
            optimizer_steps=int(result.optimizer_steps),
            optimizer_starts=int(result.optimizer_starts),
            gradient_norm=float(result.gradient_norm),
            solver_elapsed_seconds=float(result.elapsed_seconds),
            screened_facility_count=int(result.screened_facility_count),
            predicted_beneficial_facility_count=int(result.predicted_beneficial_facility_count),
            active_facility_count=int(result.active_facility_count),
            first_move_changed_facility_count=int(result.first_move_changed_facility_count),
            maximum_support_ratio=float(result.maximum_support_ratio),
            scipy_message=str(result.scipy_message),
        )
        self.last_result = wrapped
        return wrapped


class DirectTFVAuthoritativeController(V122TorchMPCController):
    """Execute the exact first target scored by the current Direct-TFV Step3."""

    def __init__(self, *args: Any, mpc: DirectTFVRecedingMPC, **kwargs: Any) -> None:
        adapter = DirectTFVRuntimeMPCAdapter(mpc)
        super().__init__(*args, mpc=adapter, **kwargs)
        self._direct_mpc_adapter = adapter

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        diagnostics = dict(action.diagnostics or {})
        diagnostics["direct_tfv_controller_contract"] = DIRECT_TFV_CONTROLLER_CONTRACT
        diagnostics["direct_tfv_step3_contract"] = DIRECT_TFV_STEP3_CONTRACT
        diagnostics["all_109_facilities_screened_contract"] = True
        diagnostics["separate_selection_threshold_used"] = False
        result = self._direct_mpc_adapter.last_result
        if result is not None:
            diagnostics.update(
                {
                    "direct_tfv_selected_source": result.selected_source,
                    "optimizer_success": result.optimizer_success,
                    "optimizer_steps": result.optimizer_steps,
                    "optimizer_starts": result.optimizer_starts,
                    "gradient_norm": result.gradient_norm,
                    "solver_elapsed_seconds": result.solver_elapsed_seconds,
                    "screened_facility_count": result.screened_facility_count,
                    "predicted_beneficial_facility_count": result.predicted_beneficial_facility_count,
                    "active_facility_count": result.active_facility_count,
                    "first_move_changed_facility_count": result.first_move_changed_facility_count,
                    "maximum_support_ratio": result.maximum_support_ratio,
                    "scipy_message": result.scipy_message[:2000],
                }
            )
        source = str(action.source)
        if source == "MPC_V122":
            source = "MPC_DIRECT_TFV_RECEDING"
        elif source == "PASSIVE_MPC_NO_PREDICTED_BENEFIT":
            source = "HOLD_DIRECT_TFV_NO_PREDICTED_BENEFIT"
        elif source.endswith("_V122"):
            source = source[:-5] + "_DIRECT_TFV"
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = [
    "DIRECT_TFV_CONTROLLER_CONTRACT",
    "DirectTFVAuthoritativeController",
    "DirectTFVRuntimeMPCAdapter",
]
