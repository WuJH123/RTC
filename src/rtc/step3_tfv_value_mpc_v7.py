"""Policy-matched Direct-TFV receding MPC V7.

V7 intentionally leaves the V6 raw optimizer unchanged: all 109 facilities are screened, the same
H120 L-BFGS-B problem is solved, and the same q95 D3-HOLD joint temporal contraction is applied.
Only the post-optimization admission margin changes.  This isolates the scientific question raised
by Development evidence: can a margin calibrated on the *current* optimizer query distribution
recover truly beneficial actions that the legacy pre-V6 residual maximum rejects?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
from .direct_tfv_policy_admission import (
    DIRECT_TFV_POLICY_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_EXECUTION_STEP3_CONTRACT,
    DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v6 import DirectTFVMPCResultV6, DirectTFVRecedingMPCV6


DIRECT_TFV_STEP3_CONTRACT = DIRECT_TFV_POLICY_EXECUTION_STEP3_CONTRACT


@dataclass(frozen=True)
class DirectTFVMPCResultV7(DirectTFVMPCResultV6):
    policy_matched_admission: bool = True
    policy_calibration_rainfall_group_count: int = 0
    policy_query_step3_contract: str = DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT


class DirectTFVRecedingMPCV7(DirectTFVRecedingMPCV6):
    """V6 raw query generator plus V2 policy-matched admission."""

    policy_mode = "direct_tfv_all109_receding_mpc_v7"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        policy_admission_calibration: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
    ) -> None:
        policy = dict(policy_admission_calibration)
        if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
            raise ValueError("Direct-TFV V7 requires the policy-matched V2 admission contract")
        if policy.get("development_only") is not True:
            raise ValueError("Direct-TFV V7 policy admission must be Development-only evidence")
        if str(policy.get("reference_semantics", "")) != "HOLD_ACTIVE_TARGET_H360":
            raise ValueError("Direct-TFV V7 policy admission has the wrong reference semantics")
        if str(policy.get("policy_query_step3_contract", "")) != DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT:
            raise ValueError("Direct-TFV V7 policy admission was calibrated on a different optimizer")
        if str(policy.get("execution_step3_contract", "")) != DIRECT_TFV_STEP3_CONTRACT:
            raise ValueError("Direct-TFV V7 policy admission targets a different execution contract")
        if policy.get("raw_optimizer_query_distribution_unchanged_between_v6_and_v7") is not True:
            raise ValueError("Direct-TFV V7 requires explicit V6/V7 raw-query equivalence")
        if int(policy.get("policy_calibration_rainfall_group_count", 0)) < 9:
            raise ValueError("Direct-TFV V7 requires at least nine policy-calibration rainfall groups")
        if policy.get("legacy_optimizer_replay_controls_current_margin") is not False:
            raise ValueError("legacy pre-V6 optimizer extrema must not control the V7 margin")

        # V6/V5 only needs the final scalar margins to perform post-optimizer admission.  Supply a
        # compatibility view with the V1 schema while keeping the V2 artifact as the authoritative
        # lineage.  Because admission is applied *after* V4/V6 optimization, this does not alter the
        # raw optimizer query distribution used to build the policy calibration panel.
        compatibility = {
            "contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
            "development_only": True,
            "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
            "optimizer_replay_count": max(4, int(policy["policy_calibration_plan_count"])),
            "density_floor_changed_facilities": int(policy["density_floor_changed_facilities"]),
            "global_margin_m3": float(policy["global_margin_m3"]),
            "dense_margin_m3": float(policy["dense_margin_m3"]),
        }
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            admission_calibration=compatibility,
            sequence_support=sequence_support,
            design=design,
        )
        self.policy_admission_calibration = policy

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV7:
        result = super().optimize(**kwargs)
        values = dict(vars(result))
        values.update(
            {
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "calibrated_admission_contract": DIRECT_TFV_POLICY_ADMISSION_CONTRACT,
                "admission_margin_kind": (
                    "policy_dense"
                    if str(result.admission_margin_kind) == "dense"
                    else "policy_global"
                    if str(result.admission_margin_kind) == "global"
                    else str(result.admission_margin_kind)
                ),
                "policy_matched_admission": True,
                "policy_calibration_rainfall_group_count": int(
                    self.policy_admission_calibration["policy_calibration_rainfall_group_count"]
                ),
                "policy_query_step3_contract": DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
            }
        )
        return DirectTFVMPCResultV7(**values)


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCResultV7",
    "DirectTFVRecedingMPCV7",
]
