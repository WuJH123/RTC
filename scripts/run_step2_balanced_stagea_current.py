"""Current Project7 Step2 runner with balanced Stage-A direct supervision.

This wrapper keeps the existing counterfactual-first V128 model, B0/objective code, data
boundaries and fail-closed production gates. It replaces only Stage A with the balanced FIT-only
direct-pair curriculum and uses the same source-lineage helper as the Development audits.
"""
from __future__ import annotations

import run_step2_action_identifiable_current as current
import rtc.step2_stagea_balanced_v128 as balanced
from rtc.step2_current_dev_context_v128 import action_identifiable_source_sha256

CURRENT_BALANCED_RUN_CONTRACT = (
    "PROJECT7_V128_CURRENT_COUNTERFACTUAL_FIRST_BALANCED_STAGE_A_DEV_V1"
)


def main() -> None:
    original_source = current._enhanced_source_sha256
    original_stage_a = current.train_counterfactual_first_stage_a_v5
    original_run_contract = current.CURRENT_ACTION_IDENTIFIABLE_RUN_CONTRACT
    original_stage_contract = current.COUNTERFACTUAL_STAGE_A_V5_CONTRACT
    original_a0_contract = current.DIRECT_FLOW_A0_V5_CONTRACT
    original_a1_contract = current.ORACLE_HYDRAULIC_A1_V5_CONTRACT
    original_a2_contract = current.JOINT_DIRECT_A2_V5_CONTRACT
    try:
        current._enhanced_source_sha256 = action_identifiable_source_sha256
        current.train_counterfactual_first_stage_a_v5 = (
            balanced.train_counterfactual_first_stage_a_balanced_v128
        )
        current.CURRENT_ACTION_IDENTIFIABLE_RUN_CONTRACT = CURRENT_BALANCED_RUN_CONTRACT
        current.COUNTERFACTUAL_STAGE_A_V5_CONTRACT = balanced.BALANCED_STAGE_A_CONTRACT
        current.DIRECT_FLOW_A0_V5_CONTRACT = balanced.BALANCED_A0_CONTRACT
        current.ORACLE_HYDRAULIC_A1_V5_CONTRACT = balanced.BALANCED_A1_CONTRACT
        current.JOINT_DIRECT_A2_V5_CONTRACT = balanced.BALANCED_A2_CONTRACT
        current.main()
    finally:
        current._enhanced_source_sha256 = original_source
        current.train_counterfactual_first_stage_a_v5 = original_stage_a
        current.CURRENT_ACTION_IDENTIFIABLE_RUN_CONTRACT = original_run_contract
        current.COUNTERFACTUAL_STAGE_A_V5_CONTRACT = original_stage_contract
        current.DIRECT_FLOW_A0_V5_CONTRACT = original_a0_contract
        current.ORACLE_HYDRAULIC_A1_V5_CONTRACT = original_a1_contract
        current.JOINT_DIRECT_A2_V5_CONTRACT = original_a2_contract


if __name__ == "__main__":
    main()
