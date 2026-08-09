from __future__ import annotations

import hashlib
import json


# This is intentionally a semantic/release contract, not a byte-for-byte hash of every
# ``src/rtc`` file. Hashing the whole source tree made otherwise reusable SWMM data/models
# fail after unrelated edits (reporting, docs-facing helpers, error messages, etc.).
#
# Bump one of these IDs only when the corresponding scientific semantics change enough that
# previously generated evidence is no longer valid. Scientific inputs/config/model/artifact
# files are still independently hashed at the generation/training/Policy-Lock layers.
IMPLEMENTATION_CONTRACT = {
    "release": "WUHAN_RTC_V06_FINAL_IMPLEMENTATION_V1",
    "causal_observation": "T0_INCLUDED_CAUSAL_HISTORY_V1",
    "no_control": "NO_SUPERVISORY_RTC_V2",
    "hydraulic_truth": "SWMM_CUMULATIVE_NODE_FLOOD_VOLUME_V1",
    "predicted_volume": "TRAPEZOID_CURRENT_PLUS_FUTURE_FLOOD_RATE_V1",
    "step2_time": "STEP2_FIXED_DISCRETE_TIME_V1",
    "mpc_action_space": "ALL_ACTUATORS_DIRECT_CONTINUOUS_PROJECTED_V1",
    "mpc_objective": "TFV_PRIMARY_PRIORITY_SOFT_SECONDARY_V1",
    "action_effect_validation": "D2_LOCAL_PLUS_D3_JOINT_SEQUENCE_V1",
    "formal_aggregation": "EQUAL_WEIGHT_PER_RAINFALL_GROUP_V1",
    "resume_semantics": "INPUT_CONFIG_ARTIFACT_HASH_BOUND_V1",
}
CODE_CONTRACT = "RTC_SCIENTIFIC_IMPLEMENTATION_CONTRACT_V2"


def rtc_implementation_contract_sha256() -> str:
    """Return a stable fingerprint of scientific implementation semantics.

    The fingerprint is deliberately insensitive to unrelated source-file edits. Reuse safety
    comes from this semantic contract *plus* exact hashes of the inputs that can change the
    numerical result: event/runtime INP, timing/configuration, action/sequence manifests,
    graph/model checkpoints, training manifests and generated artefacts.
    """

    canonical = json.dumps(
        {"contract": CODE_CONTRACT, "implementation": IMPLEMENTATION_CONTRACT},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rtc_source_tree_sha256() -> str:
    """Backward-compatible alias for the semantic implementation fingerprint.

    Older v0.6 metadata uses the field name ``rtc_source_tree_sha256``. Keeping the function
    name avoids breaking those interfaces, but the value is now a *scientific implementation
    contract hash*, not a hash of every Python source file.
    """

    return rtc_implementation_contract_sha256()
