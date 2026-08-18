"""Refined-first-move Direct-TFV MPC V9.

V9 retains the V6/V7 all-109 H120/H360 optimizer as a coordinated direction generator, then performs
a small receding-consistent refinement of the only action actually committed before replanning.  The
first 10-minute displacement of each changed facility can only shrink toward HOLD, so the refinement
cannot expand beyond the upstream q95-supported query geometry.  H350 is HOLD_ACTIVE_TARGET and the
same Step2 model evaluates the delayed H360 TFV consequence.

Execution is controlled only by a fresh optimizer-matched first-move conformal margin.  Historical
V9 full-plan and V10 prefix margins remain diagnostic metadata and are not hidden execution floors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS, refine_supported_first_move
from .direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
    first_move_margin_m3,
)
from .direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v7 import DirectTFVMPCResultV7, DirectTFVRecedingMPCV7


DIRECT_TFV_STEP3_CONTRACT = DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT


@dataclass(frozen=True)
class DirectTFVMPCResultV9(DirectTFVMPCResultV7):
    full_plan_raw_predicted_delta_tfv_m3: float = 0.0
    base_prefix_predicted_delta_tfv_m3: float = 0.0
    refined_first_move_predicted_delta_tfv_m3: float = 0.0
    first_move_refinement_gain_m3: float = 0.0
    refined_first_move_changed_facility_count: int = 0
    refined_first_move_changed_facility_ids: tuple[str, ...] = ()
    refined_first_move_margin_m3: float = 0.0
    refined_first_move_upper_bound_m3: float = 0.0
    refined_first_move_admission_passed: bool = False
    refined_first_move_admission_contract: str = DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT
    refined_first_move_semantics: str = DIRECT_TFV_FIRST_MOVE_SEMANTICS
    first_move_refiner_elapsed_seconds: float = 0.0
    first_move_refiner_steps: int = 0


class DirectTFVRecedingMPCV9(DirectTFVRecedingMPCV7):
    """V7 coordinated direction plus optimizer-matched shrink-only H10 refinement."""

    policy_mode = "direct_tfv_all109_receding_mpc_v9"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        policy_admission_calibration: Mapping[str, Any],
        first_move_admission_calibration: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
        first_move_maxiter: int = 12,
        first_move_deadline_seconds: float = 30.0,
    ) -> None:
        if str(policy_admission_calibration.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
            raise ValueError("Direct-TFV V9 requires the accepted V2 policy admission as diagnostic lineage")
        first = dict(first_move_admission_calibration)
        if str(first.get("contract", "")) != DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT:
            raise ValueError("Direct-TFV V9 requires optimizer-matched first-move admission")
        if first.get("development_only") is not True:
            raise ValueError("Direct-TFV V9 first-move admission must be Development-only")
        if str(first.get("execution_estimand", "")) != DIRECT_TFV_FIRST_MOVE_SEMANTICS:
            raise ValueError("Direct-TFV V9 first-move admission has the wrong estimand")
        if str(first.get("query_step3_contract", "")) != DIRECT_TFV_STEP3_CONTRACT:
            raise ValueError("Direct-TFV V9 first-move admission was calibrated on a different query contract")
        if first.get("generic_d3_floor_controls_execution") is not False:
            raise ValueError("generic D3 margin must not control refined first-move execution")
        if first.get("v9_full_plan_margin_controls_execution") is not False:
            raise ValueError("full-plan policy margin must not control refined first-move execution")
        if first.get("v10_prefix_margin_controls_execution") is not False:
            raise ValueError("V10 prefix margin must not control refined first-move execution")
        if int(first.get("calibration_rainfall_group_count", 0)) < int(
            first.get("minimum_calibration_rainfall_groups", 24)
        ):
            raise ValueError("Direct-TFV V9 first-move calibration has insufficient rainfall groups")
        if first_move_maxiter <= 0 or not 0.0 < float(first_move_deadline_seconds) < 600.0:
            raise ValueError("invalid Direct-TFV V9 first-move refinement budget")
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            policy_admission_calibration=policy_admission_calibration,
            sequence_support=sequence_support,
            design=design,
        )
        self.first_move_admission_calibration = first
        self.first_move_maxiter = int(first_move_maxiter)
        self.first_move_deadline_seconds = float(first_move_deadline_seconds)

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV9:
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor):
            raise ValueError("Direct-TFV V9 requires active_target")

        upstream = super().optimize(**kwargs)
        base_candidate = upstream.optimized_candidate_settings
        values = dict(vars(upstream))
        full_plan_score = float(upstream.raw_optimized_predicted_delta_tfv_m3)

        if base_candidate is None or full_plan_score >= 0.0:
            values.update(
                {
                    "policy_mode": self.policy_mode,
                    "policy_mode_contract": self.policy_mode_contract,
                    "full_plan_raw_predicted_delta_tfv_m3": full_plan_score,
                    "refined_first_move_admission_passed": False,
                    "refined_first_move_admission_contract": DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
                    "refined_first_move_semantics": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
                    "calibrated_admission_contract": DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
                }
            )
            return DirectTFVMPCResultV9(**values)

        refined = refine_supported_first_move(
            mpc=self,
            base_candidate=base_candidate,
            current_state=kwargs["current_state"],
            rainfall=kwargs["rainfall"],
            previous_actuator_flow=kwargs["previous_actuator_flow"],
            active_target=active_target,
            maxiter=self.first_move_maxiter,
            deadline_seconds=self.first_move_deadline_seconds,
        )
        margin = first_move_margin_m3(self.first_move_admission_calibration)
        upper = float(refined.predicted_delta_tfv_m3 + margin)
        passed = bool(refined.changed_facility_count > 0 and upper < 0.0)
        hold = self._hold_sequence(active_target)
        executed = refined.sequence if passed else hold
        diagnostics = self.joint_sequence_support_diagnostics(refined.sequence, active_target)

        values.update(
            {
                # Telemetry/replay must bind prediction and sequence to the same refined counterfactual.
                "settings": executed,
                "optimized_candidate_settings": refined.sequence,
                "predicted_delta_tfv_m3": float(refined.predicted_delta_tfv_m3) if passed else 0.0,
                "raw_optimized_predicted_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "selected_source": (
                    "DIRECT_TFV_RECEDING_LBFGSB"
                    if passed
                    else "HOLD_REFINED_FIRST_MOVE_UPPER_BOUND_NONNEGATIVE"
                ),
                "admission_margin_m3": float(margin),
                "admission_upper_bound_m3": float(upper),
                "admission_margin_kind": "refined_first_move_global",
                "admission_passed": passed,
                "calibrated_admission_contract": DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
                "first_move_changed_facility_count": int(refined.changed_facility_count),
                "joint_sequence_support_quantile": str(diagnostics["quantile"]),
                "joint_sequence_first_block_l1": float(diagnostics["first_block_l1"]),
                "joint_sequence_h120_l1": float(diagnostics["h120_l1"]),
                "joint_sequence_h120_total_variation_l1": float(
                    diagnostics["h120_total_variation_l1"]
                ),
                "joint_sequence_support_max_ratio": float(diagnostics["max_ratio"]),
                "joint_sequence_support_binding": bool(diagnostics["binding"]),
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "full_plan_raw_predicted_delta_tfv_m3": full_plan_score,
                "base_prefix_predicted_delta_tfv_m3": float(
                    refined.base_prefix_predicted_delta_tfv_m3
                ),
                "refined_first_move_predicted_delta_tfv_m3": float(
                    refined.predicted_delta_tfv_m3
                ),
                "first_move_refinement_gain_m3": float(refined.gain_vs_base_prefix_m3),
                "refined_first_move_changed_facility_count": int(refined.changed_facility_count),
                "refined_first_move_changed_facility_ids": tuple(refined.changed_facility_ids),
                "refined_first_move_margin_m3": float(margin),
                "refined_first_move_upper_bound_m3": float(upper),
                "refined_first_move_admission_passed": passed,
                "refined_first_move_admission_contract": DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
                "refined_first_move_semantics": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
                "first_move_refiner_elapsed_seconds": float(refined.elapsed_seconds),
                "first_move_refiner_steps": int(refined.optimizer_steps),
            }
        )
        return DirectTFVMPCResultV9(**values)


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCResultV9",
    "DirectTFVRecedingMPCV9",
]
