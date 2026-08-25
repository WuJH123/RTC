"""Authoritative-runtime adapter for Project7 Direct-TFV control.

The adapter is intentionally optimizer-agnostic. Historical implementations treated the literal
``DIRECT_TFV_RECEDING_LBFGSB`` source token as the definition of a valid action, which prevented a
scientifically valid policy-return portfolio from reaching the authoritative target write. Validity
now comes from the Step3 score/admission contract, while legacy optimizer diagnostics remain
available when present.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .closed_loop import CausalObservation, ControllerAction
from .controller_v122 import V122TorchMPCController
from .step3_tfv_value_mpc_v6 import DIRECT_TFV_STEP3_CONTRACT

DIRECT_TFV_CONTROLLER_CONTRACT = "PROJECT7_DIRECT_TFV_TARGET_LATCH_CONTROLLER_V7_OPTIMIZER_AGNOSTIC"
DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT = "PROJECT7_DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_V4_H10_AWARE"


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
    counterfactual_candidate_semantics: str
    scipy_message: str
    receding_prefix_predicted_delta_tfv_m3: float
    receding_prefix_margin_m3: float
    receding_prefix_upper_bound_m3: float
    receding_prefix_admission_passed: bool
    receding_prefix_admission_contract: str
    receding_prefix_candidate_semantics: str
    full_plan_policy_admission_passed: bool
    full_plan_policy_upper_bound_m3: float
    refined_first_move_predicted_delta_tfv_m3: float
    refined_first_move_margin_m3: float
    refined_first_move_upper_bound_m3: float
    refined_first_move_admission_passed: bool
    refined_first_move_admission_contract: str
    refined_first_move_semantics: str
    refined_first_move_changed_facility_count: int
    refined_first_move_changed_facility_ids: tuple[str, ...]
    first_move_refinement_gain_m3: float
    first_move_refiner_elapsed_seconds: float
    first_move_refiner_steps: int
    policy_return_portfolio_contract: str = ""
    policy_return_portfolio_candidate_count: int = 0
    policy_return_portfolio_selected_source: str = "HOLD"
    policy_return_portfolio_sources: tuple[str, ...] = ()
    policy_return_portfolio_scores_m3: tuple[float, ...] = ()
    policy_return_portfolio_upper_bounds_m3: tuple[float, ...] = ()
    policy_return_portfolio_base_step2_scores_m3: tuple[float, ...] = ()
    h10_probe_generator_contract: str = ""
    h10_probe_count: int = 0


class DirectTFVRuntimeMPCAdapter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_result: _RuntimeMPCResult | None = None

    @property
    def model(self) -> torch.nn.Module:
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
        valid = bool(getattr(result, "candidate_valid", getattr(result, "admission_passed", False)))
        actuator_ids = tuple(str(value) for value in self.inner.graph.actuator_ids)
        if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
            raise RuntimeError("Direct-TFV counterfactual actuator order must contain 109 unique IDs")
        block_steps = int(self.inner.design.control_block_steps)
        free_count = int(self.inner.design.free_control_blocks)
        optimized = getattr(result, "optimized_candidate_settings", None)
        if not isinstance(optimized, torch.Tensor):
            optimized = result.settings
        free_blocks_tensor = optimized[::block_steps][:free_count]
        if tuple(free_blocks_tensor.shape) != (free_count, 109):
            raise RuntimeError("Direct-TFV runtime could not recover exact [12,109] counterfactual blocks")
        free_blocks = tuple(
            tuple(float(value) for value in row)
            for row in free_blocks_tensor.detach().cpu().to(torch.float64).tolist()
        )
        hold_reference = tuple(
            float(value)
            for value in previous_requested_settings.detach().cpu().to(torch.float64).reshape(-1).tolist()
        )
        raw_scores = tuple(float(value) for value in getattr(result, "active_facility_screening_scores_m3", ()))
        active_scores = tuple(value if math.isfinite(value) else 0.0 for value in raw_scores)
        finite_scores = [value for value in raw_scores if math.isfinite(value)]
        best_screening = min(finite_scores) if finite_scores else 0.0
        raw_pred = float(getattr(result, "raw_optimized_predicted_delta_tfv_m3", result.predicted_delta_tfv_m3))
        optimizer_gain = float(getattr(result, "optimizer_gain_beyond_best_screening_m3", 0.0))
        if not math.isfinite(optimizer_gain):
            optimizer_gain = 0.0
        requested_quantile = str(getattr(self.inner.design, "active_support_quantile", "q90"))
        effective_quantile = str(self.inner.active_support_quantile_effective()) if hasattr(self.inner, "active_support_quantile_effective") else requested_quantile
        support_ceiling = int(self.inner.active_support_ceiling()) if hasattr(self.inner, "active_support_ceiling") else 109
        active_count = int(getattr(result, "active_facility_count", 0))
        beneficial_count = int(getattr(result, "predicted_beneficial_facility_count", 0))
        ceiling_binding = bool(getattr(result, "active_set_ceiling_binding", beneficial_count > active_count and active_count >= min(109, support_ceiling)))
        refined_semantics = str(getattr(result, "refined_first_move_semantics", ""))
        candidate_semantics = refined_semantics or "SUPPORT_BOUNDED_COUNTERFACTUAL_SEQUENCE"
        admission_margin = float(getattr(result, "admission_margin_m3", 0.0))
        admission_upper = float(getattr(result, "admission_upper_bound_m3", raw_pred))
        admission_passed = bool(getattr(result, "admission_passed", valid))
        first_changed = int(getattr(result, "first_move_changed_facility_count", 0))

        wrapped = _RuntimeMPCResult(
            settings=result.settings,
            predicted_delta_tfv_m3=float(result.predicted_delta_tfv_m3),
            raw_optimized_predicted_delta_tfv_m3=raw_pred,
            admission_margin_m3=admission_margin,
            admission_upper_bound_m3=admission_upper,
            admission_margin_kind=str(getattr(result, "admission_margin_kind", "none")),
            admission_passed=admission_passed,
            calibrated_admission_contract=str(getattr(result, "calibrated_admission_contract", "")),
            candidate_valid=valid,
            selected_source=str(getattr(result, "selected_source", "HOLD")),
            optimizer_success=bool(getattr(result, "optimizer_success", True)),
            optimizer_steps=int(getattr(result, "optimizer_steps", 0)),
            optimizer_starts=int(getattr(result, "optimizer_starts", 0)),
            gradient_norm=float(getattr(result, "gradient_norm", 0.0)),
            solver_elapsed_seconds=float(getattr(result, "elapsed_seconds", 0.0)),
            screened_facility_count=int(getattr(result, "screened_facility_count", 109)),
            predicted_beneficial_facility_count=beneficial_count,
            active_facility_count=active_count,
            active_facility_ids=tuple(getattr(result, "active_facility_ids", ())),
            active_facility_screening_scores_m3=active_scores,
            first_move_changed_facility_count=first_changed,
            maximum_support_ratio=float(getattr(result, "maximum_support_ratio", 0.0)),
            joint_sequence_support_quantile=str(getattr(result, "joint_sequence_support_quantile", effective_quantile)),
            joint_sequence_first_block_l1=float(getattr(result, "joint_sequence_first_block_l1", 0.0)),
            joint_sequence_h120_l1=float(getattr(result, "joint_sequence_h120_l1", 0.0)),
            joint_sequence_h120_total_variation_l1=float(getattr(result, "joint_sequence_h120_total_variation_l1", 0.0)),
            joint_sequence_support_max_ratio=float(getattr(result, "joint_sequence_support_max_ratio", 0.0)),
            joint_sequence_support_binding=bool(getattr(result, "joint_sequence_support_binding", False)),
            best_screening_predicted_delta_tfv_m3=float(best_screening),
            optimizer_gain_beyond_best_screening_m3=optimizer_gain,
            active_set_ceiling_binding=ceiling_binding,
            active_support_quantile_requested=requested_quantile,
            active_support_quantile_effective=effective_quantile,
            active_support_ceiling=support_ceiling,
            counterfactual_actuator_ids=actuator_ids,
            optimized_free_control_blocks=free_blocks,
            hold_reference_settings=hold_reference,
            counterfactual_candidate_semantics=candidate_semantics,
            scipy_message=str(getattr(result, "scipy_message", "NOT_USED")),
            receding_prefix_predicted_delta_tfv_m3=float(getattr(result, "receding_prefix_predicted_delta_tfv_m3", raw_pred)),
            receding_prefix_margin_m3=float(getattr(result, "receding_prefix_margin_m3", admission_margin)),
            receding_prefix_upper_bound_m3=float(getattr(result, "receding_prefix_upper_bound_m3", admission_upper)),
            receding_prefix_admission_passed=bool(getattr(result, "receding_prefix_admission_passed", admission_passed)),
            receding_prefix_admission_contract=str(getattr(result, "receding_prefix_admission_contract", "")),
            receding_prefix_candidate_semantics=str(getattr(result, "receding_prefix_candidate_semantics", candidate_semantics)),
            full_plan_policy_admission_passed=bool(getattr(result, "full_plan_policy_admission_passed", False)),
            full_plan_policy_upper_bound_m3=float(getattr(result, "full_plan_policy_upper_bound_m3", 0.0)),
            refined_first_move_predicted_delta_tfv_m3=float(getattr(result, "refined_first_move_predicted_delta_tfv_m3", raw_pred)),
            refined_first_move_margin_m3=float(getattr(result, "refined_first_move_margin_m3", admission_margin)),
            refined_first_move_upper_bound_m3=float(getattr(result, "refined_first_move_upper_bound_m3", admission_upper)),
            refined_first_move_admission_passed=bool(getattr(result, "refined_first_move_admission_passed", admission_passed)),
            refined_first_move_admission_contract=str(getattr(result, "refined_first_move_admission_contract", getattr(result, "calibrated_admission_contract", ""))),
            refined_first_move_semantics=refined_semantics,
            refined_first_move_changed_facility_count=int(getattr(result, "refined_first_move_changed_facility_count", first_changed)),
            refined_first_move_changed_facility_ids=tuple(getattr(result, "refined_first_move_changed_facility_ids", ())),
            first_move_refinement_gain_m3=float(getattr(result, "first_move_refinement_gain_m3", 0.0)),
            first_move_refiner_elapsed_seconds=float(getattr(result, "first_move_refiner_elapsed_seconds", 0.0)),
            first_move_refiner_steps=int(getattr(result, "first_move_refiner_steps", 0)),
            policy_return_portfolio_contract=str(getattr(result, "policy_return_portfolio_contract", "")),
            policy_return_portfolio_candidate_count=int(getattr(result, "policy_return_portfolio_candidate_count", 0)),
            policy_return_portfolio_selected_source=str(getattr(result, "policy_return_portfolio_selected_source", "HOLD")),
            policy_return_portfolio_sources=tuple(getattr(result, "policy_return_portfolio_sources", ())),
            policy_return_portfolio_scores_m3=tuple(float(x) for x in getattr(result, "policy_return_portfolio_scores_m3", ())),
            policy_return_portfolio_upper_bounds_m3=tuple(float(x) for x in getattr(result, "policy_return_portfolio_upper_bounds_m3", ())),
            policy_return_portfolio_base_step2_scores_m3=tuple(float(x) for x in getattr(result, "policy_return_portfolio_base_step2_scores_m3", ())),
            h10_probe_generator_contract=str(getattr(result, "h10_probe_generator_contract", "")),
            h10_probe_count=int(getattr(result, "h10_probe_count", 0)),
        )
        self.last_result = wrapped
        return wrapped


class DirectTFVAuthoritativeController(V122TorchMPCController):
    def __init__(self, *args: Any, mpc: Any, **kwargs: Any) -> None:
        adapter = DirectTFVRuntimeMPCAdapter(mpc)
        super().__init__(*args, mpc=adapter, **kwargs)
        self._direct_mpc_adapter = adapter

    def decide(self, obs: CausalObservation, *, observation_already_recorded: bool = False) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "direct_tfv_controller_contract": DIRECT_TFV_CONTROLLER_CONTRACT,
                "direct_tfv_step3_contract": str(getattr(self._direct_mpc_adapter.inner, "policy_mode_contract", DIRECT_TFV_STEP3_CONTRACT)),
                "all_109_facilities_screened_contract": True,
                "calibrated_one_sided_admission_used": True,
                "joint_sequence_support_used": True,
                "target_latch_semantics": "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED",
            }
        )
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
                    "active_facility_ids": list(result.active_facility_ids),
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
                    "admission_margin_kind": result.admission_margin_kind,
                    "admission_passed": result.admission_passed,
                    "calibrated_admission_contract": result.calibrated_admission_contract,
                    "refined_first_move_predicted_delta_tfv_m3": result.refined_first_move_predicted_delta_tfv_m3,
                    "refined_first_move_margin_m3": result.refined_first_move_margin_m3,
                    "refined_first_move_upper_bound_m3": result.refined_first_move_upper_bound_m3,
                    "refined_first_move_admission_passed": result.refined_first_move_admission_passed,
                    "refined_first_move_admission_contract": result.refined_first_move_admission_contract,
                    "refined_first_move_semantics": result.refined_first_move_semantics,
                    "refined_first_move_changed_facility_count": result.refined_first_move_changed_facility_count,
                    "refined_first_move_changed_facility_ids": list(result.refined_first_move_changed_facility_ids),
                    "counterfactual_plan_telemetry_contract": DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT,
                    "counterfactual_reference_semantics": "LATCH_PREVIOUS_TARGET_H360",
                    "counterfactual_candidate_semantics": result.counterfactual_candidate_semantics,
                    "counterfactual_actuator_ids": list(result.counterfactual_actuator_ids),
                    "optimized_free_control_blocks": [list(row) for row in result.optimized_free_control_blocks],
                    "hold_reference_settings": list(result.hold_reference_settings),
                    "scipy_message": result.scipy_message[:2000],
                    "policy_return_portfolio_contract": result.policy_return_portfolio_contract,
                    "policy_return_portfolio_candidate_count": result.policy_return_portfolio_candidate_count,
                    "policy_return_portfolio_selected_source": result.policy_return_portfolio_selected_source,
                    "policy_return_portfolio_sources": list(result.policy_return_portfolio_sources),
                    "policy_return_portfolio_scores_m3": list(result.policy_return_portfolio_scores_m3),
                    "policy_return_portfolio_upper_bounds_m3": list(result.policy_return_portfolio_upper_bounds_m3),
                    "policy_return_portfolio_base_step2_scores_m3": list(result.policy_return_portfolio_base_step2_scores_m3),
                    "h10_probe_generator_contract": result.h10_probe_generator_contract,
                    "h10_probe_count": result.h10_probe_count,
                }
            )
            if str(getattr(result, "policy_mode_contract", "")).startswith(
                "PROJECT7_OPERATIONAL_DEVELOPMENT_V25_"
            ):
                diagnostics.update(
                    {
                        "tfv_value_available": bool(getattr(result, "tfv_value_available", False)),
                        "tfv_value_prediction_m3": getattr(result, "tfv_value_prediction_m3", None),
                        "tfv_value_upper_bound_m3": getattr(result, "tfv_value_upper_bound_m3", None),
                        "tfv_value_admission_passed": bool(
                            getattr(result, "tfv_value_admission_passed", False)
                        ),
                        "engineering_feasible": bool(getattr(result, "engineering_feasible", False)),
                        "passive_channels_unchanged": bool(
                            getattr(result, "passive_channels_unchanged", False)
                        ),
                        "sequence_support_valid": bool(getattr(result, "sequence_support_valid", False)),
                        "tfv_value_gate_reason": str(
                            getattr(result, "tfv_value_gate_reason", "UNKNOWN")
                        ),
                    }
                )
        source = str(action.source)
        if source == "MPC_V122" and result is not None:
            source = "MPC_DIRECT_TFV_RECEDING" if result.candidate_valid else "LATCH_PREVIOUS_TARGET_DIRECT_TFV"
        elif source == "MPC_V122":
            source = "MPC_DIRECT_TFV_RECEDING"
        elif source == "PASSIVE_MPC_NO_PREDICTED_BENEFIT":
            source = "LATCH_PREVIOUS_TARGET_DIRECT_TFV"
        elif source.endswith("_V122"):
            source = source[:-5] + "_DIRECT_TFV"
        diagnostics["calibrated_runtime_action_class"] = "ACTION" if source == "MPC_DIRECT_TFV_RECEDING" else "HOLD"
        diagnostics["hold_semantics"] = "LATCH_PREVIOUS_COMMANDED_TARGET"
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = [
    "DIRECT_TFV_CONTROLLER_CONTRACT",
    "DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT",
    "DirectTFVAuthoritativeController",
    "DirectTFVRuntimeMPCAdapter",
]
