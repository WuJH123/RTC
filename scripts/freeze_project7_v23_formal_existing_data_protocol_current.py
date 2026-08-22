"""Freeze the Project7 V23 publication protocol after the existing-truth audit.

The protocol is valid in either of two preregistered scientific modes:
1) EXACT_MATCH_RETRAIN_ALLOWED, only when the complete frozen 18 Train + 6 Validation roles are
   distribution-matched to V23; or
2) FIXED_POLICY_NO_RETRAIN, which freezes the development-selected V23+V15+V21 controller without
   reusing mismatched labels.

The old calibration role stays removed. Final remains sealed until a separate Policy Lock is created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_v23_formal_reuse import (
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    V23_FORMAL_PROTOCOL_CONTRACT,
    V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
    validate_frozen_split,
)


MODEL_ACCEPTANCE_CONTRACT = "MODEL_ACCEPTANCE_CONTRACT_V4_DIMENSIONLESS_PREREGISTERED"
ALLOWED_FORMAL_MODES = ("EXACT_MATCH_RETRAIN_ALLOWED", "FIXED_POLICY_NO_RETRAIN")


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-audit", required=True)
    parser.add_argument("--formal-split-contract", required=True)
    parser.add_argument("--model-acceptance-contract", required=True)
    parser.add_argument("--development-comparison", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    audit_path = Path(args.reuse_audit).resolve()
    split_path = Path(args.formal_split_contract).resolve()
    acceptance_path = Path(args.model_acceptance_contract).resolve()
    development_path = Path(args.development_comparison).resolve()
    audit = _json(audit_path)
    split = _json(split_path)
    acceptance = _json(acceptance_path)
    development = _json(development_path)

    if audit.get("contract") != V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT:
        raise ValueError("wrong V23 existing-truth audit contract")
    if audit.get("final_truth_contamination") is not False:
        raise RuntimeError("Final truth contamination blocks a clean V23 Formal protocol")
    if audit.get("fixed_policy_formal_evaluation_allowed") is not True:
        raise RuntimeError("existing-truth audit did not authorize any clean Formal evaluation path")
    mode = str(audit.get("recommended_formal_mode", ""))
    if mode not in ALLOWED_FORMAL_MODES:
        raise ValueError(f"unsupported V23 Formal mode: {mode}")
    if mode == "EXACT_MATCH_RETRAIN_ALLOWED" and audit.get("formal_retraining_allowed") is not True:
        raise RuntimeError("audit mode/retraining permission mismatch")

    roles = validate_frozen_split(split)
    if acceptance.get("contract") != MODEL_ACCEPTANCE_CONTRACT:
        raise ValueError("V23 Formal protocol requires frozen MODEL_ACCEPTANCE_CONTRACT_V4")
    instruction = str(acceptance.get("instruction", ""))
    if "never" not in instruction.lower() or "final" not in instruction.lower():
        raise ValueError("model acceptance contract lost its no-threshold-relaxation/Final firewall")
    if development.get("development_only") is not True:
        raise ValueError("V23 source comparison must remain Development-only evidence")

    role_manifest = {
        "contract": V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
        "scientific_role_authority": "configs/project7_v069_split_contract.json",
        "development_train_event_ids": list(roles["development_train"]),
        "development_validation_event_ids": list(roles["development_validation"]),
        "final_event_ids": list(roles["final"]),
        "development_train_count": len(roles["development_train"]),
        "development_validation_count": len(roles["development_validation"]),
        "final_count": len(roles["final"]),
        "calibration_role_removed": True,
        "safety_audit_role_removed": True,
        "historical_policy_return_calibration_records": "ARCHIVAL_ONLY_NOT_FORMAL_LEARNING",
        "rainfall_groups_cross_scientific_splits": False,
        "validation_used_for_training": False,
        "final_untouched_before_policy_lock": True,
        "final_used_for_tuning": False,
        "source_split_contract_sha256": sha256_file(split_path),
    }

    protocol = {
        "contract": V23_FORMAL_PROTOCOL_CONTRACT,
        "formal_mode": mode,
        "formal_mode_reason": (
            "complete existing V23-exact Train/Validation truth permits optional minimal retraining"
            if mode == "EXACT_MATCH_RETRAIN_ALLOWED"
            else "existing truth is not fully V23-distribution-matched; freeze the development-selected controller rather than fabricate/relabel training truth"
        ),
        "existing_data_only_for_any_retraining": True,
        "new_rainfall_for_training_allowed": False,
        "new_policy_return_truth_allowed": False,
        "new_training_examples_allowed": False,
        "nearest_neighbor_or_interpolated_labels_allowed": False,
        "partial_matched_subset_training_allowed": False,
        "calibration_role_removed": True,
        "candidate_generator": "FROZEN_V23_STRONG_STORM_HYDRAULIC_CANDIDATE_V1",
        "candidate_generator_can_change_after_protocol": False,
        "development_benchmark_can_tune_after_protocol": False,
        "validation_can_update_model_parameters": False,
        "final_can_tune_any_model_threshold_or_candidate": False,
        "model_acceptance_contract": MODEL_ACCEPTANCE_CONTRACT,
        "hard_acceptance_thresholds_may_be_lowered": False,
        "step1_policy": "FREEZE_IF_ACCEPTED; RETRAIN_ONLY_FROM_PREEXISTING_DEVELOPMENT_TRAIN_IF_REQUIRED_AND_FORMAL_MODE_ALLOWS",
        "step2_policy": "FREEZE_IF_ACCEPTED; RETRAIN_ONLY_FROM_PREEXISTING_DEVELOPMENT_TRAIN_IF_REQUIRED_AND_FORMAL_MODE_ALLOWS",
        "step3_policy": (
            "MINIMAL_RETRAIN_FROM_FULL_EXACT_MATCHED_DEVELOPMENT_TRAIN_ONLY"
            if mode == "EXACT_MATCH_RETRAIN_ALLOWED"
            else "FREEZE_CURRENT_V15_RANK_AND_V21_BOUNDARY_NO_RETRAIN"
        ),
        "primary_control_metric": "SYSTEM_WIDE_TFV_M3_AUTHORITATIVE_SWMM",
        "secondary_safety_metric": "PRIORITY8_PFV_M3_AUTHORITATIVE_SWMM",
        "pfv_safety_contract": "PFV_PROPOSED <= 100_M3 + 1.05 * PFV_NO_CONTROL",
        "global_peak_role": "REPORT_ONLY",
        "competitive_comparators": ["no_control", "internal_rtc", "auto_rbc", "efd"],
        "diagnostic_only_strategies": ["all_open", "all_closed", "hold"],
        "statistical_unit": "RAINFALL_EVENT",
        "decision_rows_are_independent_replicates": False,
        "report_event_balanced_mean": True,
        "report_median": True,
        "report_aggregate_volume": True,
        "report_paired_event_differences": True,
        "engineering_gates": [
            "target_write_readback",
            "routing_continuity",
            "actuator_bounds_0_1",
            "target_slew_le_0p5",
            "q95_joint_sequence_support",
            "passive_27_channels_unchanged",
            "no_online_swmm_candidate_search",
            "no_projected_gradient",
            "no_lbfgsb",
        ],
        "policy_lock_required_before_final": True,
        "final_opened": False,
        "formal_protocol_frozen": True,
        "ready_for_policy_lock": False,
        "source_reuse_audit_sha256": sha256_file(audit_path),
        "source_split_contract_sha256": sha256_file(split_path),
        "source_model_acceptance_contract_sha256": sha256_file(acceptance_path),
        "source_v23_development_comparison_sha256": sha256_file(development_path),
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roles_path = out_dir / "V23_PUBLICATION_ROLE_MANIFEST.json"
    roles_path.write_text(json.dumps(role_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol["publication_role_manifest_sha256"] = sha256_file(roles_path)
    protocol_path = out_dir / "V23_FORMAL_PROTOCOL.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "formal_mode": mode,
                "formal_protocol_path": str(protocol_path),
                "formal_protocol_sha256": sha256_file(protocol_path),
                "publication_role_manifest_path": str(roles_path),
                "publication_role_manifest_sha256": sha256_file(roles_path),
                "development_train_count": len(roles["development_train"]),
                "development_validation_count": len(roles["development_validation"]),
                "final_count": len(roles["final"]),
                "calibration_role_removed": True,
                "new_empirical_training_data_generated": False,
                "ready_for_policy_lock": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
