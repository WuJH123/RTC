"""Receding-consistent Direct-TFV MPC V8.

V8 leaves the V6/V7 raw optimizer and q95 support geometry unchanged. It changes only the benefit
credited to the action that is actually committed before the next observation. The full H120/H360
optimizer plan remains search guidance; execution is admitted from the Step2 value of the optimized
first 10-minute block followed by HOLD for H350, calibrated with a rainfall-group one-sided bound.

This allows V8 to do two things that V9 could not do cleanly: recover a first move that the conservative
full-plan margin rejected when its executable prefix is reliably beneficial, and reject a full plan
whose apparent H360 benefit is carried by future blocks that will be replanned before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from .direct_tfv_receding_prefix import (
    DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_EXECUTION_STEP3_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
    executable_prefix_sequence,
    receding_prefix_margin_m3,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v7 import DirectTFVMPCResultV7, DirectTFVRecedingMPCV7


DIRECT_TFV_STEP3_CONTRACT = DIRECT_TFV_RECEDING_PREFIX_EXECUTION_STEP3_CONTRACT


@dataclass(frozen=True)
class DirectTFVMPCResultV8(DirectTFVMPCResultV7):
    receding_prefix_predicted_delta_tfv_m3: float = 0.0
    receding_prefix_margin_m3: float = 0.0
    receding_prefix_upper_bound_m3: float = 0.0
    receding_prefix_admission_passed: bool = False
    receding_prefix_admission_contract: str = DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT
    receding_prefix_candidate_semantics: str = DIRECT_TFV_RECEDING_PREFIX_SEMANTICS
    receding_prefix_calibration_rainfall_group_count: int = 0
    full_plan_policy_admission_passed: bool = False
    full_plan_policy_upper_bound_m3: float = 0.0


class DirectTFVRecedingMPCV8(DirectTFVRecedingMPCV7):
    """V7 raw optimizer with execution aligned to the H10 receding commitment."""

    policy_mode = "direct_tfv_all109_receding_mpc_v8"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        policy_admission_calibration: Mapping[str, Any],
        receding_prefix_admission_calibration: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
    ) -> None:
        prefix = dict(receding_prefix_admission_calibration)
        if str(policy_admission_calibration.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
            raise ValueError("Direct-TFV V8 requires the accepted V2 full-plan policy admission")
        if str(prefix.get("contract", "")) != DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT:
            raise ValueError("Direct-TFV V8 requires the receding-prefix admission contract")
        if prefix.get("development_only") is not True:
            raise ValueError("Direct-TFV V8 receding-prefix admission must be Development-only")
        if str(prefix.get("execution_estimand", "")) != DIRECT_TFV_RECEDING_PREFIX_SEMANTICS:
            raise ValueError("Direct-TFV V8 receding-prefix calibration has the wrong estimand")
        if str(prefix.get("policy_query_step3_contract", "")) != DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT:
            raise ValueError("Direct-TFV V8 prefix calibration was generated from a different raw optimizer")
        if str(prefix.get("execution_step3_contract", "")) != DIRECT_TFV_STEP3_CONTRACT:
            raise ValueError("Direct-TFV V8 prefix calibration targets a different execution contract")
        if prefix.get("raw_optimizer_query_distribution_unchanged") is not True:
            raise ValueError("Direct-TFV V8 requires frozen V6 raw optimizer queries")
        if prefix.get("full_plan_policy_margin_controls_execution") is not False:
            raise ValueError("V2 full-plan margin must not remain the V8 execution gate")
        if int(prefix.get("receding_prefix_calibration_rainfall_group_count", 0)) < 9:
            raise ValueError("Direct-TFV V8 requires at least nine prefix-calibration rainfall groups")
        if str(prefix.get("density_classification_variable", "")) != "FIRST_MOVE_CHANGED_FACILITY_COUNT":
            raise ValueError("Direct-TFV V8 prefix density must follow the executed first move")
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            policy_admission_calibration=policy_admission_calibration,
            sequence_support=sequence_support,
            design=design,
        )
        self.receding_prefix_admission_calibration = prefix

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV8:
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor):
            raise ValueError("Direct-TFV V8 requires active_target")
        result = super().optimize(**kwargs)
        candidate = result.optimized_candidate_settings
        values = dict(vars(result))
        full_policy_passed = bool(result.admission_passed)
        full_policy_upper = float(result.admission_upper_bound_m3)

        if candidate is None or float(result.raw_optimized_predicted_delta_tfv_m3) >= 0.0:
            values.update(
                {
                    "policy_mode": self.policy_mode,
                    "policy_mode_contract": self.policy_mode_contract,
                    "receding_prefix_admission_passed": False,
                    "receding_prefix_calibration_rainfall_group_count": int(
                        self.receding_prefix_admission_calibration[
                            "receding_prefix_calibration_rainfall_group_count"
                        ]
                    ),
                    "full_plan_policy_admission_passed": full_policy_passed,
                    "full_plan_policy_upper_bound_m3": full_policy_upper,
                    "calibrated_admission_contract": DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT,
                }
            )
            return DirectTFVMPCResultV8(**values)

        prefix = executable_prefix_sequence(
            candidate,
            active_target,
            control_block_steps=int(self.design.control_block_steps),
        )
        prefix_score = self.score_sequence(
            current_state=kwargs["current_state"],
            rainfall=kwargs["rainfall"],
            sequence=prefix,
            previous_actuator_flow=kwargs["previous_actuator_flow"],
            active_target=active_target,
        )
        score_value = float(prefix_score.detach().cpu())
        changed = int(result.first_move_changed_facility_count)
        margin, kind = receding_prefix_margin_m3(
            self.receding_prefix_admission_calibration,
            changed,
        )
        upper = float(score_value + margin)
        hold = self._hold_sequence(active_target)
        passed = bool(changed > 0 and upper < 0.0)

        values.update(
            {
                "settings": candidate if passed else hold,
                "predicted_delta_tfv_m3": score_value if passed else 0.0,
                "selected_source": (
                    "DIRECT_TFV_RECEDING_LBFGSB"
                    if passed
                    else "HOLD_RECEDING_PREFIX_UPPER_BOUND_NONNEGATIVE"
                ),
                "admission_margin_m3": float(margin),
                "admission_upper_bound_m3": float(upper),
                "admission_margin_kind": str(kind),
                "admission_passed": passed,
                "calibrated_admission_contract": DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT,
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "receding_prefix_predicted_delta_tfv_m3": float(score_value),
                "receding_prefix_margin_m3": float(margin),
                "receding_prefix_upper_bound_m3": float(upper),
                "receding_prefix_admission_passed": passed,
                "receding_prefix_admission_contract": DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT,
                "receding_prefix_candidate_semantics": DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
                "receding_prefix_calibration_rainfall_group_count": int(
                    self.receding_prefix_admission_calibration[
                        "receding_prefix_calibration_rainfall_group_count"
                    ]
                ),
                "full_plan_policy_admission_passed": full_policy_passed,
                "full_plan_policy_upper_bound_m3": full_policy_upper,
            }
        )
        return DirectTFVMPCResultV8(**values)


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCResultV8",
    "DirectTFVRecedingMPCV8",
]
