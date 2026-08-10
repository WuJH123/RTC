from __future__ import annotations

import hashlib
import json


IMPLEMENTATION_CONTRACT = {
    "release": "WUHAN_RTC_V068_SIMULATION_ASSET_LINEAGE_IMPLEMENTATION_V1",
    "causal_observation": "T0_INCLUDED_ALL_TRAJECTORIES_V2",
    "event_preparation": "EXPLICIT_SOURCE_ADDITIONAL_EFFECTIVE_WARMUP_V2",
    "hydraulic_initialization": "WARMUP_DISTINCT_FROM_HISTORY_WITH_DEVELOPMENT_SENSITIVITY_V2",
    "pretraining_readiness": "EVENT_HISTORY_RAIN_SENSOR_ACTUATION_PROVENANCE_FAIL_CLOSED_V1",
    "step1_sampling": "TRAJECTORY_BALANCED_DRY_WET_FLOOD_HIGH_STRATIFIED_TRAIN_ONLY_V1",
    "step1_acceptance": "ALL_WINDOWS_GROUP_BALANCED_PLUS_HYDRAULIC_STRATA_V1",
    "no_control": "NO_SUPERVISORY_RTC_V2",
    "internal_rtc": "EVENT_FORCING_DWF_PLUS_FROZEN_NATIVE_CONTROLS_PAYLOAD_V1",
    "baseline_information_budget": "STRONG_COMPARATOR_INFORMATION_ADVANTAGES_DISCLOSED_V1",
    "hydraulic_truth": "SWMM_CUMULATIVE_NODE_FLOOD_VOLUME_V1",
    "predicted_volume": "TRAPEZOID_CURRENT_PLUS_FUTURE_FLOOD_RATE_V1",
    "step2_time": "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2",
    "swmm_engine_lineage": "ONE_ENGINE_PER_TRAIN_ACCEPT_FINAL_V1",
    "counterfactual_prefix": "EXACT_NO_CONTROL_PREFIX_REPLAY_V1",
    "event_forcing_lineage": "SCIENTIFIC_EVENT_INP_EXCEPT_CONTROLS_THREADS_V1",
    "native_rule_lineage": "CANONICAL_CONTROLS_PAYLOAD_SHA256_V1",
    "final_event_completeness": "ALL_LOCKED_FINAL_EVENTS_COMPLETE_SEVEN_STRATEGY_V1",
    "formal_rule_baselines": "AUTO_RBC_CAUSAL_PLUS_STORAGE_EFD_V1",
    "d3_feasibility": "SEQUENTIAL_SETTING_RATE_MATCHES_CONTROLLER_V1",
    "mpc_action_space": "ALL_SWMM_WRITABLE_ACTUATORS_CONTINUOUS_SIMULATION_V2",
    "field_actuation_claim": "REQUIRES_ENGINEERING_CAPABILITY_MAP_OTHERWISE_SIMULATION_ONLY_V1",
    "mpc_objective": "TFV_PRIMARY_PRIORITY_SOFT_SECONDARY_V1",
    "mpc_robust_selection": "BEST_SO_FAR_AND_HOLD_DOMINANCE_V1",
    "action_effect_validation": "D2_LOCAL_PLUS_D3_JOINT_SEQUENCE_V1",
    "efficient_d2_sampling": "ROTATING_ALL_ACTUATOR_COVERAGE_BUDGET_V1",
    "pretraining_viability": "EXACT_SWMM_CONTROL_LEVERAGE_PILOT_V1",
    "phase0_group_selection": "FORCING_ONLY_DIVERSE_DEVELOPMENT_GROUPS_V1",
    "phase0_timing_freeze": "NON_CENSORED_EVIDENCE_BOUND_TIMING_WITH_LONG_TRAJECTORY_VIEWS_V2",
    "formal_aggregation": "EQUAL_WEIGHT_PER_RAINFALL_GROUP_V1",
    "resume_semantics": "GLOBAL_STATE_ACTION_ENGINE_SIMULATION_IDENTITY_V2",
    "simulation_asset_registry": "LOCAL_LARGE_DATA_SQLITE_INDEX_HASH_VERIFIED_V1",
    "endpoint_preflight": "CHECKPOINT_PLUS_HORIZON_BEFORE_SWMM_LAUNCH_V1",
    "horizon_reuse": "LONG_TRAJECTORY_PREFIX_FOR_TIMING_NOT_CUMULATIVE_TRUTH_V1",
}
CODE_CONTRACT = "RTC_SCIENTIFIC_IMPLEMENTATION_CONTRACT_V2"


def rtc_implementation_contract_sha256() -> str:
    canonical = json.dumps(
        {"contract": CODE_CONTRACT, "implementation": IMPLEMENTATION_CONTRACT},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rtc_source_tree_sha256() -> str:
    return rtc_implementation_contract_sha256()
