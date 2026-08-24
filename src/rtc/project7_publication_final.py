"""Publication-facing Project7 controller contract and evidence checks.

This module does not change control actions.  It freezes the scientific interpretation of the
already-developed V23 controller after the bounded V6/V7 Step2 follow-ups failed.  The publication
policy is therefore the exact V23 fixed policy, evaluated end to end in authoritative SWMM.
"""
from __future__ import annotations

from typing import Any, Mapping


PUBLICATION_FINAL_CONTRACT = "PROJECT7_PUBLICATION_FINAL_CONTROLLER_V1"
EXPECTED_STEP2_SHA256 = "3a05704812a07a914d0ce9d8d026f6c84a4dbed646743f95d27726b29c3a544a"
EXPECTED_POLICY_LOCK_CONTRACT = "PROJECT7_V23_POLICY_LOCK_V1"
EXPECTED_VALIDATION_CONTRACT = "PROJECT7_V23_FORMAL_DEVELOPMENT_VALIDATION_EVIDENCE_V1"
EXPECTED_FINAL_RUN_CONTRACT = "PROJECT7_V23_POLICY_LOCKED_FINAL_PROPOSED_RUN_V1"
EXPECTED_FORMAL_MODE = "FIXED_POLICY_NO_RETRAIN"
EXPECTED_PFV_CONTRACT = "PFV_PROPOSED <= 100_M3 + 1.05 * PFV_NO_CONTROL"
EXPECTED_CANDIDATES = (
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
)
EXPECTED_DISTRIBUTION_SHIFT_DISPOSITION = (
    "RETAIN_FIXED_POLICY_ONLY_BECAUSE_END_TO_END_VALIDATION_PRECEDED_POLICY_LOCK"
)


def _lower(value: object) -> str:
    return str(value or "").strip().lower()


def validate_publication_controller_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("contract") != PUBLICATION_FINAL_CONTRACT:
        raise ValueError("wrong Project7 publication-final controller contract")
    goal = payload.get("research_goal")
    if not isinstance(goal, Mapping):
        raise ValueError("publication contract lacks research_goal")
    if goal.get("primary_outcome") != "SYSTEM_WIDE_TOTAL_FLOOD_VOLUME_M3":
        raise RuntimeError("publication primary objective drifted from system-wide TFV")
    if goal.get("priority8_safety_contract") != EXPECTED_PFV_CONTRACT:
        raise RuntimeError("Priority8 PFV non-inferiority contract changed")
    if goal.get("global_peak_role") != "REPORT_ONLY":
        raise RuntimeError("Global Peak was reintroduced as an optimization/gating target")

    step1 = payload.get("step1")
    if not isinstance(step1, Mapping) or step1.get("training_status") != "FROZEN":
        raise RuntimeError("publication Step1 must remain frozen")
    if float(step1.get("minimum", float("nan"))) != 0.70:
        raise RuntimeError("Step1 acceptance threshold changed")

    step2 = payload.get("step2")
    if not isinstance(step2, Mapping):
        raise ValueError("publication contract lacks step2")
    if _lower(step2.get("checkpoint_sha256")) != EXPECTED_STEP2_SHA256:
        raise RuntimeError("publication Step2 is not the frozen Direct-TFV V5 checkpoint")
    if step2.get("training_status") != "FROZEN_NO_RETRAIN":
        raise RuntimeError("publication Step2 retraining is forbidden")
    if step2.get("standalone_surrogate_superiority_claim_allowed") is not False:
        raise RuntimeError("failed Step2 standalone surrogate was relabelled as accepted")
    followups = step2.get("development_followups")
    if not isinstance(followups, Mapping):
        raise ValueError("publication contract lacks Step2 development disposition")
    if followups.get("v6_main_preservation") != "REJECTED_EXISTING_DATA_GATE":
        raise RuntimeError("V6 rejection is not preserved")
    if followups.get("v7_action_identity_interaction") != "REJECTED_EXISTING_DATA_GATE":
        raise RuntimeError("V7 rejection is not preserved")
    if followups.get("v8_allowed") is not False:
        raise RuntimeError("publication final contract reopened Step2 architecture search")

    step3 = payload.get("step3")
    if not isinstance(step3, Mapping):
        raise ValueError("publication contract lacks step3")
    if step3.get("continuous_gradient_mpc") is not False:
        raise RuntimeError("publication controller cannot claim continuous gradient MPC")
    if int(step3.get("decision_interval_seconds", -1)) != 600:
        raise RuntimeError("publication control interval changed")
    if int(step3.get("execute_first_move_minutes", -1)) != 10:
        raise RuntimeError("publication first-move execution changed")
    if tuple(step3.get("candidate_families", ())) != EXPECTED_CANDIDATES:
        raise RuntimeError("publication candidate family contract changed")
    if int(step3.get("production_candidate_family_count_max", -1)) != 3:
        raise RuntimeError("publication candidate cardinality changed")
    if step3.get("v21_boundary_candidate_distribution_exact_match") is not False:
        raise RuntimeError("publication contract must retain the V21/V23 candidate-distribution mismatch")
    if step3.get("distribution_shift_disposition") != EXPECTED_DISTRIBUTION_SHIFT_DISPOSITION:
        raise RuntimeError("publication Step3 distribution-shift disposition changed")
    if step3.get("distribution_shift_must_be_reported_as_component_limitation") is not True:
        raise RuntimeError("publication contract hides the V23 boundary distribution-shift limitation")

    engineering = payload.get("engineering")
    if not isinstance(engineering, Mapping):
        raise ValueError("publication contract lacks engineering controls")
    if (
        engineering.get("model_action_channels"),
        engineering.get("supervisory_controls"),
        engineering.get("passive_reference_channels"),
    ) != (109, 82, 27):
        raise RuntimeError("109/82/27 action-control contract changed")
    if float(engineering.get("max_setting_delta_per_update", float("nan"))) != 0.5:
        raise RuntimeError("command slew contract changed")
    if engineering.get("support_quantile") != "q95":
        raise RuntimeError("q95 engineering support was changed")
    if engineering.get("online_swmm_candidate_search") is not False:
        raise RuntimeError("online SWMM candidate search is forbidden")

    formal = payload.get("formal")
    if not isinstance(formal, Mapping) or formal.get("mode") != EXPECTED_FORMAL_MODE:
        raise RuntimeError("publication Formal mode must remain FIXED_POLICY_NO_RETRAIN")
    if formal.get("final_can_tune_policy") is not False:
        raise RuntimeError("Final tuning firewall was weakened")
    if formal.get("final_results_used_for_training") is not False:
        raise RuntimeError("Final results cannot train the publication policy")
    if formal.get("final_results_used_for_tuning") is not False:
        raise RuntimeError("Final results cannot tune the publication policy")

    boundaries = payload.get("claim_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("publication contract lacks claim boundaries")
    if boundaries.get("must_report_v23_boundary_candidate_distribution_shift") is not True:
        raise RuntimeError("publication claims hide the V23/V21 candidate-distribution shift")


def validate_publication_policy_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("contract") != EXPECTED_POLICY_LOCK_CONTRACT or lock.get("locked") is not True:
        raise RuntimeError("publication package requires an immutable V23 Policy Lock")
    if lock.get("formal_mode") != EXPECTED_FORMAL_MODE:
        raise RuntimeError("Policy Lock uses another Formal mode")
    if _lower(lock.get("step2_checkpoint_sha256")) != EXPECTED_STEP2_SHA256:
        raise RuntimeError("Policy Lock uses another Step2 checkpoint")
    if lock.get("step2_runtime_lineage_accepted") is not True:
        raise RuntimeError("Policy Lock lacks Step2 runtime-lineage acceptance")
    if lock.get("policy_mutation_after_lock_forbidden") is not True:
        raise RuntimeError("Policy Lock mutation firewall was weakened")
    if lock.get("final_can_tune_policy") is not False:
        raise RuntimeError("Policy Lock permits Final tuning")
    if lock.get("pfv_safety_contract") != EXPECTED_PFV_CONTRACT:
        raise RuntimeError("Policy Lock PFV safety contract changed")


def validate_publication_validation(validation: Mapping[str, Any]) -> None:
    if validation.get("contract") != EXPECTED_VALIDATION_CONTRACT:
        raise ValueError("wrong Formal validation evidence contract")
    if validation.get("pfv_safety_all_events_pass") is not True:
        raise RuntimeError("Formal validation failed Priority8 PFV safety")
    if validation.get("engineering_all_events_pass") is not True:
        raise RuntimeError("Formal validation failed engineering execution")
    if validation.get("policy_changed_after_validation_started") is not False:
        raise RuntimeError("policy changed during Formal validation")
    if validation.get("final_truth_opened") is not False:
        raise RuntimeError("Formal validation accessed Final truth")


__all__ = [
    "EXPECTED_DISTRIBUTION_SHIFT_DISPOSITION",
    "EXPECTED_FINAL_RUN_CONTRACT",
    "EXPECTED_PFV_CONTRACT",
    "EXPECTED_POLICY_LOCK_CONTRACT",
    "EXPECTED_STEP2_SHA256",
    "EXPECTED_VALIDATION_CONTRACT",
    "PUBLICATION_FINAL_CONTRACT",
    "validate_publication_controller_contract",
    "validate_publication_policy_lock",
    "validate_publication_validation",
]
