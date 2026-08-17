"""Authoritative-runtime adapter for optimizer-consistent Project7 Direct-TFV MPC."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step3_tfv_value_mpc_v6 import DIRECT_TFV_STEP3_CONTRACT, DirectTFVRecedingMPCV6

DIRECT_TFV_CONTROLLER_CONTRACT = "PROJECT7_DIRECT_TFV_TARGET_LATCH_CONTROLLER_V5"
DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT = "PROJECT7_DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_V2"

@dataclass(frozen=True)
class _RuntimeMPCResult:
    settings: torch.Tensor
    predicted_delta_tfv_m3: float
    raw_optimized_predicted_delta_tfv_m3: float
    admission_margin_m3: float
    admission_upper_bound_m3: float
    admission_margin_kind: str
    admission_passed: bool
    calibrated_admission_contract: str
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
    active_facility_ids: tuple[str, ...]
    active_facility_screening_scores_m3: tuple[float, ...]
    first_move_changed_facility_count: int
    maximum_support_ratio: float
    joint_sequence_support_quantile: str
    joint_sequence_first_block_l1: float
    joint_sequence_h120_l1: float
    joint_sequence_h120_total_variation_l1: float
    joint_sequence_support_max_ratio: float
    joint_sequence_support_binding: bool
    best_screening_predicted_delta_tfv_m3: float
    optimizer_gain_beyond_best_screening_m3: float
    active_set_ceiling_binding: bool
    active_support_quantile_requested: str
    active_support_quantile_effective: str
    active_support_ceiling: int
    counterfactual_actuator_ids: tuple[str, ...]
    optimized_free_control_blocks: tuple[tuple[float, ...], ...]
    hold_reference_settings: tuple[float, ...]
    scipy_message: str

class DirectTFVRuntimeMPCAdapter:
    def __init__(self, inner: DirectTFVRecedingMPCV6) -> None:
        self.inner = inner
        self.last_result: _RuntimeMPCResult | None = None

    @property
    def model(self) -> torch.nn.Module:
        return self.inner.model

    def optimize(self, *, initial_state: torch.Tensor, rainfall_scenarios: torch.Tensor,
                 current_settings: torch.Tensor, previous_requested_settings: torch.Tensor,
                 fallback_settings: torch.Tensor, previous_actuator_flow: torch.Tensor,
                 max_delta_per_update: float | None) -> _RuntimeMPCResult:
        self.last_result = None
        del fallback_settings
        if max_delta_per_update is not None and abs(
            float(max_delta_per_update) - float(self.inner.design.max_setting_delta_per_update)
        ) > 1.0e-9:
            raise ValueError("runtime controller max-delta differs from Direct-TFV Step3 design")
        result = self.inner.optimize(
            current_state=initial_state, rainfall=rainfall_scenarios,
            previous_actuator_flow=previous_actuator_flow, current_settings=current_settings,
            active_target=previous_requested_settings,
        )
        valid = result.selected_source == "DIRECT_TFV_RECEDING_LBFGSB"
        actuator_ids = tuple(str(value) for value in self.inner.graph.actuator_ids)
        if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
            raise RuntimeError("Direct-TFV counterfactual actuator order must contain 109 unique IDs")
        block_steps = int(self.inner.design.control_block_steps)
        free_count = int(self.inner.design.free_control_blocks)
        optimized = getattr(result, "optimized_candidate_settings", None)
        if optimized is None:
            optimized = result.settings
        free_blocks_tensor = optimized[::block_steps][:free_count]
        if tuple(free_blocks_tensor.shape) != (free_count, 109):
            raise RuntimeError("Direct-TFV runtime could not recover exact [12,109] optimized plan")
        free_blocks = tuple(tuple(float(value) for value in row)
                            for row in free_blocks_tensor.detach().cpu().to(torch.float64).tolist())
        hold_reference = tuple(float(value) for value in
                               previous_requested_settings.detach().cpu().to(torch.float64).reshape(-1).tolist())
        active_scores = tuple(float(value) for value in result.active_facility_screening_scores_m3)
        best_screening = min(active_scores) if active_scores else 0.0
        raw_pred = float(getattr(result, "raw_optimized_predicted_delta_tfv_m3",
                                 result.predicted_delta_tfv_m3))
        optimizer_gain = float(best_screening - raw_pred) if math.isfinite(raw_pred) else 0.0
        requested_quantile = str(getattr(self.inner.design, "active_support_quantile", "q90"))
        effective_quantile = str(self.inner.active_support_quantile_effective())
        support_ceiling = int(self.inner.active_support_ceiling())
        if int(getattr(self.inner.design, "active_facility_count", 0)) > 0:
            support_ceiling = min(support_ceiling, int(self.inner.design.active_facility_count))
        ceiling_binding = bool(result.predicted_beneficial_facility_count > result.active_facility_count
                               and result.active_facility_count >= min(109, support_ceiling))
        wrapped = _RuntimeMPCResult(
            settings=result.settings, predicted_delta_tfv_m3=float(result.predicted_delta_tfv_m3),
            raw_optimized_predicted_delta_tfv_m3=raw_pred,
            admission_margin_m3=float(getattr(result, "admission_margin_m3", 0.0)),
            admission_upper_bound_m3=float(getattr(result, "admission_upper_bound_m3", raw_pred)),
            admission_margin_kind=str(getattr(result, "admission_margin_kind", "none")),
            admission_passed=bool(getattr(result, "admission_passed", valid)),
            calibrated_admission_contract=str(getattr(result, "calibrated_admission_contract", "")),
            candidate_valid=bool(valid), selected_source=str(result.selected_source),
            optimizer_success=bool(result.optimizer_success), optimizer_steps=int(result.optimizer_steps),
            optimizer_starts=int(result.optimizer_starts), gradient_norm=float(result.gradient_norm),
            solver_elapsed_seconds=float(result.elapsed_seconds),
            screened_facility_count=int(result.screened_facility_count),
            predicted_beneficial_facility_count=int(result.predicted_beneficial_facility_count),
            active_facility_count=int(result.active_facility_count), active_facility_ids=tuple(result.active_facility_ids),
            active_facility_screening_scores_m3=active_scores,
            first_move_changed_facility_count=int(result.first_move_changed_facility_count),
            maximum_support_ratio=float(result.maximum_support_ratio),
            joint_sequence_support_quantile=str(result.joint_sequence_support_quantile),
            joint_sequence_first_block_l1=float(result.joint_sequence_first_block_l1),
            joint_sequence_h120_l1=float(result.joint_sequence_h120_l1),
            joint_sequence_h120_total_variation_l1=float(result.joint_sequence_h120_total_variation_l1),
            joint_sequence_support_max_ratio=float(result.joint_sequence_support_max_ratio),
            joint_sequence_support_binding=bool(result.joint_sequence_support_binding),
            best_screening_predicted_delta_tfv_m3=float(best_screening), optimizer_gain_beyond_best_screening_m3=optimizer_gain,
            active_set_ceiling_binding=ceiling_binding, active_support_quantile_requested=requested_quantile,
            active_support_quantile_effective=effective_quantile, active_support_ceiling=int(support_ceiling),
            counterfactual_actuator_ids=actuator_ids, optimized_free_control_blocks=free_blocks,
            hold_reference_settings=hold_reference, scipy_message=str(result.scipy_message),
        )
        self.last_result = wrapped
        return wrapped

class DirectTFVAuthoritativeController(V122TorchMPCController):
    def __init__(self, *args: Any, mpc: DirectTFVRecedingMPCV6, **kwargs: Any) -> None:
        adapter = DirectTFVRuntimeMPCAdapter(mpc)
        super().__init__(*args, mpc=adapter, **kwargs)
        self._direct_mpc_adapter = adapter

    def decide(self, obs: CausalObservation, *, observation_already_recorded: bool = False) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update({
            "direct_tfv_controller_contract": DIRECT_TFV_CONTROLLER_CONTRACT,
            "direct_tfv_step3_contract": DIRECT_TFV_STEP3_CONTRACT,
            "all_109_facilities_screened_contract": True,
            "calibrated_one_sided_admission_used": True,
            "joint_sequence_support_used": True,
        })
        result = self._direct_mpc_adapter.last_result
        if result is not None:
            diagnostics.update({
                "direct_tfv_selected_source": result.selected_source,
                "optimizer_success": result.optimizer_success, "optimizer_steps": result.optimizer_steps,
                "optimizer_starts": result.optimizer_starts, "gradient_norm": result.gradient_norm,
                "solver_elapsed_seconds": result.solver_elapsed_seconds,
                "screened_facility_count": result.screened_facility_count,
                "predicted_beneficial_facility_count": result.predicted_beneficial_facility_count,
                "active_facility_count": result.active_facility_count, "active_facility_ids": list(result.active_facility_ids),
                "active_facility_screening_scores_m3": list(result.active_facility_screening_scores_m3),
                "first_move_changed_facility_count": result.first_move_changed_facility_count,
                "maximum_support_ratio": result.maximum_support_ratio,
                "joint_sequence_support_quantile": result.joint_sequence_support_quantile,
                "joint_sequence_first_block_l1": result.joint_sequence_first_block_l1,
                "joint_sequence_h120_l1": result.joint_sequence_h120_l1,
                "joint_sequence_h120_total_variation_l1": result.joint_sequence_h120_total_variation_l1,
                "joint_sequence_support_max_ratio": result.joint_sequence_support_max_ratio,
                "joint_sequence_support_binding": result.joint_sequence_support_binding,
                "best_screening_predicted_delta_tfv_m3": result.best_screening_predicted_delta_tfv_m3,
                "optimizer_gain_beyond_best_screening_m3": result.optimizer_gain_beyond_best_screening_m3,
                "active_set_ceiling_binding": result.active_set_ceiling_binding,
                "active_support_quantile_requested": result.active_support_quantile_requested,
                "active_support_quantile_effective": result.active_support_quantile_effective,
                "active_support_ceiling": result.active_support_ceiling,
                "raw_optimized_predicted_delta_tfv_m3": result.raw_optimized_predicted_delta_tfv_m3,
                "admission_margin_m3": result.admission_margin_m3,
                "admission_upper_bound_m3": result.admission_upper_bound_m3,
                "admission_margin_kind": result.admission_margin_kind, "admission_passed": result.admission_passed,
                "calibrated_admission_contract": result.calibrated_admission_contract,
                "counterfactual_plan_telemetry_contract": DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT,
                "counterfactual_reference_semantics": "HOLD_ACTIVE_TARGET_H360",
                "counterfactual_candidate_semantics": "SUPPORT_CONTRACTED_H120_FREE_BLOCKS_THEN_TERMINAL_HOLD_H360",
                "counterfactual_actuator_ids": list(result.counterfactual_actuator_ids),
                "optimized_free_control_blocks": [list(row) for row in result.optimized_free_control_blocks],
                "hold_reference_settings": list(result.hold_reference_settings), "scipy_message": result.scipy_message[:2000],
            })
        source = str(action.source)
        if source == "MPC_V122" and result is not None:
            if result.selected_source == "DIRECT_TFV_RECEDING_LBFGSB":
                source = "MPC_DIRECT_TFV_RECEDING"
            elif result.selected_source == "HOLD_CALIBRATED_TFV_UPPER_BOUND_NONNEGATIVE":
                source = "HOLD_DIRECT_TFV_CALIBRATED_UPPER_BOUND"
            elif result.selected_source == "HOLD_NO_EXECUTABLE_FIRST_MOVE":
                source = "HOLD_DIRECT_TFV_NO_EXECUTABLE_FIRST_MOVE"
            else:
                source = "HOLD_DIRECT_TFV_NO_PREDICTED_BENEFIT"
        elif source == "MPC_V122":
            source = "MPC_DIRECT_TFV_RECEDING"
        elif source == "PASSIVE_MPC_NO_PREDICTED_BENEFIT":
            source = "HOLD_DIRECT_TFV_CALIBRATED_OR_NO_BENEFIT"
        elif source.endswith("_V122"):
            source = source[:-5] + "_DIRECT_TFV"
        diagnostics["calibrated_runtime_action_class"] = "ACTION" if source == "MPC_DIRECT_TFV_RECEDING" else "HOLD"
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)

__all__ = ["DIRECT_TFV_CONTROLLER_CONTRACT", "DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT",
           "DirectTFVAuthoritativeController", "DirectTFVRuntimeMPCAdapter"]
