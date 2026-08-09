from __future__ import annotations

import hashlib
import json


# This is intentionally a semantic/release contract, not a byte-for-byte hash of every
# ``src/rtc`` file. Scientific inputs/config/model/artifact files remain independently hashed.
IMPLEMENTATION_CONTRACT = {
    "release": "WUHAN_RTC_V063_INDEPENDENT_VIABILITY_IMPLEMENTATION_V1",
    "causal_observation": "T0_INCLUDED_ALL_TRAJECTORIES_V2",
    "no_control": "NO_SUPERVISORY_RTC_V2",
    "hydraulic_truth": "SWMM_CUMULATIVE_NODE_FLOOD_VOLUME_V1",
    "predicted_volume": "TRAPEZOID_CURRENT_PLUS_FUTURE_FLOOD_RATE_V1",
    "step2_time": "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2",
    "swmm_engine_lineage": "ONE_ENGINE_PER_TRAIN_ACCEPT_FINAL_V1",
    "counterfactual_prefix": "EXACT_NO_CONTROL_PREFIX_REPLAY_V1",
    "event_forcing_lineage": "SCIENTIFIC_EVENT_INP_EXCEPT_CONTROLS_THREADS_V1",
    "final_event_completeness": "ALL_LOCKED_FINAL_EVENTS_COMPLETE_FIVE_STRATEGY_V1",
    "d3_feasibility": "SEQUENTIAL_SETTING_RATE_MATCHES_CONTROLLER_V1",
    "mpc_action_space": "ALL_ACTUATORS_DIRECT_CONTINUOUS_PROJECTED_V1",
    "mpc_objective": "TFV_PRIMARY_PRIORITY_SOFT_SECONDARY_V1",
    "mpc_robust_selection": "BEST_SO_FAR_AND_HOLD_DOMINANCE_V1",
    "action_effect_validation": "D2_LOCAL_PLUS_D3_JOINT_SEQUENCE_V1",
    "efficient_d2_sampling": "ROTATING_ALL_ACTUATOR_COVERAGE_BUDGET_V1",
    "pretraining_viability": "EXACT_SWMM_CONTROL_LEVERAGE_PILOT_V1",
    "formal_aggregation": "EQUAL_WEIGHT_PER_RAINFALL_GROUP_V1",
    "resume_semantics": "INPUT_CONFIG_ARTIFACT_HASH_BOUND_V1",
}
CODE_CONTRACT = "RTC_SCIENTIFIC_IMPLEMENTATION_CONTRACT_V2"


def rtc_implementation_contract_sha256() -> str:
    """Return a stable fingerprint of scientific implementation semantics."""

    canonical = json.dumps(
        {"contract": CODE_CONTRACT, "implementation": IMPLEMENTATION_CONTRACT},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rtc_source_tree_sha256() -> str:
    """Backward-compatible alias for the semantic scientific implementation fingerprint."""

    return rtc_implementation_contract_sha256()
