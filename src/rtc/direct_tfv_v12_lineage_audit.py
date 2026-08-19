"""Auditable recovery rules for V12 admission/source lineage drift.

This module never declares two different behavioral fingerprints equivalent.  A stale
admission remains fail-closed unless its aggregate behavioral fingerprint and all
runtime artifact lineages match the current V12 implementation.  When provenance is
insufficient, the recovery action is a matched V12 calibration refresh using the
existing role-pure V12 calibration rainfall groups, not new rainfall or generic D3.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_direct_tfv import (
    direct_tfv_first_move_behavioral_source_sha256,
    direct_tfv_first_move_source_sha256,
)
from .direct_tfv_v12_lineage import (
    V12_RAINFALL_AGGREGATION,
    V12_RAINFALL_DECAY_PER_STEP,
    V12_RAINFALL_HISTORY_STEPS,
    V12_RAINFALL_MULTIPLIERS,
    direct_tfv_v12_behavioral_sha256,
)
from .step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)


V12_BEHAVIORAL_MANIFEST_CONTRACT = "PROJECT7_DIRECT_TFV_V12_BEHAVIORAL_MANIFEST_V1"
V12_LINEAGE_AUDIT_CONTRACT = "PROJECT7_DIRECT_TFV_V12_ADMISSION_LINEAGE_AUDIT_V1"
V12_REFRESH_RECOMMENDATION = "RECALIBRATE_CURRENT_MAIN_WITH_EXISTING_V12_CALIBRATION_GROUPS"
V12_REUSE_RECOMMENDATION = "REUSE_CURRENT_MATCHED_V12_ADMISSION"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_tfv_v12_behavioral_manifest() -> dict[str, Any]:
    """Return an interpretable decomposition of the current V12 behavior contract."""
    root = Path(__file__).resolve().parent
    return {
        "contract": V12_BEHAVIORAL_MANIFEST_CONTRACT,
        "v12_behavioral_source_sha256": direct_tfv_v12_behavioral_sha256(),
        "first_move_behavioral_source_sha256": direct_tfv_first_move_behavioral_source_sha256(),
        "first_move_full_source_sha256_provenance_only": direct_tfv_first_move_source_sha256(),
        "scenario_mean_step3_source_sha256": _sha256(root / "step3_tfv_value_mpc_v10.py"),
        "step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "rainfall_scenario_contract": DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
        "rainfall_multipliers": list(V12_RAINFALL_MULTIPLIERS),
        "rainfall_history_steps": int(V12_RAINFALL_HISTORY_STEPS),
        "rainfall_decay_per_step": float(V12_RAINFALL_DECAY_PER_STEP),
        "rainfall_aggregation": V12_RAINFALL_AGGREGATION,
    }


def _artifact_behavior(admission: Mapping[str, Any]) -> str:
    lineage = admission.get("lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}
    return str(
        admission.get(
            "v12_behavioral_source_sha256",
            lineage.get("v12_behavioral_source_sha256", ""),
        )
    ).lower()


def audit_v12_admission_lineage(
    admission: Mapping[str, Any],
    *,
    step2_checkpoint_sha256: str,
    sequence_support_sha256: str,
) -> dict[str, Any]:
    """Classify whether a V12 admission can be reused by current source.

    A behavioral mismatch is deliberately not resolved by a compatibility whitelist.
    The audit instead prescribes a role-pure exact-query calibration refresh.  Existing
    V12 calibration rainfall groups may be reused only while they remain calibration
    only and have not been used to tune the current implementation.
    """
    current = direct_tfv_v12_behavioral_manifest()
    lineage = admission.get("lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}
    calibrated_behavior = _artifact_behavior(admission)
    current_behavior = str(current["v12_behavioral_source_sha256"]).lower()

    checks = {
        "behavioral_fingerprint_match": bool(calibrated_behavior)
        and calibrated_behavior == current_behavior,
        "step2_checkpoint_match": str(lineage.get("step2_checkpoint_sha256", "")).lower()
        == str(step2_checkpoint_sha256).lower(),
        "sequence_support_match": str(lineage.get("sequence_support_sha256", "")).lower()
        == str(sequence_support_sha256).lower(),
        "query_step3_contract_match": str(admission.get("query_step3_contract", ""))
        == DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "rainfall_scenario_contract_match": str(admission.get("rainfall_scenario_contract", ""))
        == DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    }
    safe = all(checks.values())
    stored_manifest = admission.get("v12_behavioral_manifest")
    manifest_present = isinstance(stored_manifest, Mapping)
    component_equivalence_proven = bool(
        safe
        or (
            manifest_present
            and str(stored_manifest.get("contract", "")) == V12_BEHAVIORAL_MANIFEST_CONTRACT
            and str(stored_manifest.get("v12_behavioral_source_sha256", "")).lower()
            == current_behavior
        )
    )

    reasons = [name for name, passed in checks.items() if not passed]
    if safe:
        recommendation = V12_REUSE_RECOMMENDATION
    else:
        recommendation = V12_REFRESH_RECOMMENDATION
        if not manifest_present and not checks["behavioral_fingerprint_match"]:
            reasons.append("historical_admission_lacks_component_behavior_manifest")

    return {
        "contract": V12_LINEAGE_AUDIT_CONTRACT,
        "safe_to_reuse_admission": safe,
        "recommended_action": recommendation,
        "checks": checks,
        "mismatch_reasons": reasons,
        "calibrated_v12_behavioral_source_sha256": calibrated_behavior,
        "current_v12_behavioral_source_sha256": current_behavior,
        "current_behavioral_manifest": current,
        "calibrated_component_manifest_present": manifest_present,
        "component_behavioral_equivalence_proven": component_equivalence_proven,
        "full_byte_hash_can_establish_behavioral_compatibility": False,
        "compatibility_whitelist_allowed": False,
        "existing_v12_calibration_rainfall_groups_may_be_reused_if_role_pure": True,
        "new_rainfall_required_for_lineage_refresh": False,
        "generic_d3_required_for_lineage_refresh": False,
        "minimum_role_pure_calibration_rainfall_groups": 24,
        "authoritative_refresh_branch_count": 48,
        "authoritative_refresh_hold_branches": 24,
        "authoritative_refresh_candidate_branches": 24,
        "historical_swmm_truth_reuse": (
            "NOT_AUTOMATIC_REQUIRE_EXACT_PREFIX_SEQUENCE_INP_AND_BRANCH_IDENTITY_PROOF"
        ),
        "refresh_estimand": (
            "CURRENT_MAIN_V12_SCENARIO_MEAN_REFINED_TARGET_LATCH_FIRST_MOVE_VS_PREVIOUS_TARGET_HOLD"
        ),
    }


__all__ = [
    "V12_BEHAVIORAL_MANIFEST_CONTRACT",
    "V12_LINEAGE_AUDIT_CONTRACT",
    "V12_REFRESH_RECOMMENDATION",
    "V12_REUSE_RECOMMENDATION",
    "audit_v12_admission_lineage",
    "direct_tfv_v12_behavioral_manifest",
]
