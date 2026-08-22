"""Freeze the Project7 V23 Formal protocol from existing evidence only.

The learning records keep their original rainfall-group-disjoint train/validation/calibration roles.
Closed-loop Final events come only from the already preregistered Project7 v0.6.9 forcing-only split.
No model is trained and no SWMM/rainfall/truth is generated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_v23_formal_reuse import (
    LEARNING_ROLES,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    V23_FORMAL_PROTOCOL_CONTRACT,
    V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
    assert_excluded_events_absent_from_learning,
    learning_groups_by_role,
    validate_frozen_final_split,
)


MODEL_ACCEPTANCE_CONTRACT = "MODEL_ACCEPTANCE_CONTRACT_V4_DIMENSIONLESS_PREREGISTERED"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("matched existing-truth JSONL contains no valid rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-audit", required=True)
    parser.add_argument("--matched-records-jsonl", required=True)
    parser.add_argument("--formal-split-contract", required=True)
    parser.add_argument("--model-acceptance-contract", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    audit_path = Path(args.reuse_audit).resolve()
    matched_path = Path(args.matched_records_jsonl).resolve()
    split_path = Path(args.formal_split_contract).resolve()
    acceptance_path = Path(args.model_acceptance_contract).resolve()
    audit = _load_json(audit_path)
    if audit.get("contract") != V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT:
        raise ValueError("wrong V23 existing-truth reuse audit contract")
    if audit.get("formal_training_allowed") is not True:
        raise RuntimeError(
            "V23 existing truth is not fully exact-matched across preserved learning roles; "
            "do not fabricate labels or selectively train on easy matched queries"
        )
    required_audit_flags = {
        "no_new_empirical_data": True,
        "nearest_neighbor_matching_used": False,
        "interpolation_used": False,
        "pseudo_labels_used": False,
        "partial_match_cherry_picking_allowed": False,
        "original_learning_roles_preserved": True,
    }
    for key, expected in required_audit_flags.items():
        if audit.get(key) is not expected:
            raise RuntimeError(f"V23 reuse audit violates Formal firewall: {key}")

    records = _load_jsonl(matched_path)
    roles = learning_groups_by_role(records)
    assert_excluded_events_absent_from_learning(
        records,
        audit.get("excluded_development_event_ids", ()),
    )
    split = _load_json(split_path)
    final_events = validate_frozen_final_split(split)
    acceptance = _load_json(acceptance_path)
    if acceptance.get("contract") != MODEL_ACCEPTANCE_CONTRACT:
        raise ValueError("V23 Formal protocol requires frozen MODEL_ACCEPTANCE_CONTRACT_V4")

    role_summary: dict[str, Any] = {}
    for role in LEARNING_ROLES:
        group_set = set(roles[role])
        rows = [row for row in records if str(row["rainfall_group"]) in group_set]
        role_summary[role] = {
            "rainfall_groups": list(roles[role]),
            "rainfall_group_count": len(roles[role]),
            "query_set_count": len({str(row["query_set_id"]) for row in rows}),
            "existing_record_count": len(rows),
            "candidate_source_counts": dict(
                sorted(Counter(str(row["candidate_source"]) for row in rows).items())
            ),
        }

    role_manifest = {
        "contract": V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
        "existing_data_only": True,
        "learning_split_unit": "RAINFALL_GROUP",
        "learning_roles_preserved_from_authoritative_existing_records": True,
        "rainfall_groups_cross_learning_roles": False,
        "development_steering_events_excluded_from_learning": True,
        "learning_roles": role_summary,
        "closed_loop_final_source": "FROZEN_PROJECT7_V069_FORCING_ONLY_SPLIT",
        "closed_loop_final_event_ids": list(final_events),
        "closed_loop_final_event_count": len(final_events),
        "closed_loop_final_untouched_before_policy_lock": True,
        "source_reuse_audit_sha256": sha256_file(audit_path),
        "source_matched_records_sha256": sha256_file(matched_path),
        "formal_split_contract_sha256": sha256_file(split_path),
        "model_acceptance_contract_sha256": sha256_file(acceptance_path),
    }

    protocol = {
        "contract": V23_FORMAL_PROTOCOL_CONTRACT,
        "existing_data_only_for_training": True,
        "new_rainfall_for_training_allowed": False,
        "new_policy_return_truth_allowed": False,
        "new_training_examples_allowed": False,
        "development_benchmark_can_tune_after_freeze": False,
        "final_can_tune_any_model_or_threshold": False,
        "candidate_generator": "FROZEN_V23_STRONG_STORM_HYDRAULIC_CANDIDATE",
        "step1": "KEEP_FROZEN_IF_MODEL_ACCEPTANCE_V4_PASSES; OTHERWISE_RETRAIN_ONLY_FROM_PREEXISTING_DEVELOPMENT_TRAIN_DATA",
        "step2": "KEEP_FROZEN_IF_MODEL_ACCEPTANCE_V4_PASSES; OTHERWISE_RETRAIN_ONLY_FROM_PREEXISTING_DEVELOPMENT_TRAIN_DATA",
        "step3": (
            "AUDIT_EXISTING_V15_RANK_FIRST; RETRAIN RANK/SELECTED_VS_HOLD ONLY FROM THE PRESERVED "
            "EXACT_MATCHED EXISTING ROLES WHEN VALIDATION EVIDENCE REQUIRES IT"
        ),
        "model_acceptance_contract": MODEL_ACCEPTANCE_CONTRACT,
        "hard_thresholds_may_be_lowered_after_results": False,
        "primary_control_metric": "TFV_M3_AUTHORITATIVE_SWMM",
        "safety_metric": "PFV_M3_AUTHORITATIVE_SWMM",
        "pfv_safety_contract": "PFV_PROPOSED <= 100_M3 + 1.05 * PFV_NO_CONTROL",
        "global_peak_role": "REPORT_ONLY",
        "competitive_comparators": ["no_control", "internal_rtc", "auto_rbc", "efd"],
        "statistical_unit": "RAINFALL_EVENT",
        "decision_rows_are_independent_replicates": False,
        "report_each_final_event": True,
        "report_event_balanced_mean_and_median": True,
        "report_aggregate_volume": True,
        "report_paired_event_differences": True,
        "engineering_gates": [
            "target_write_readback",
            "routing_continuity",
            "actuator_bounds",
            "target_slew_0p5",
            "q95_joint_sequence_support",
            "passive_27_channels_unchanged",
            "no_online_swmm_candidate_search",
            "no_projected_gradient",
            "no_lbfgsb",
        ],
        "policy_lock_required_before_closed_loop_final": True,
        "ready_for_policy_lock": False,
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roles_path = out_dir / "V23_PUBLICATION_ROLE_MANIFEST.json"
    protocol_path = out_dir / "V23_FORMAL_REUSE_ONLY_PROTOCOL.json"
    roles_path.write_text(json.dumps(role_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol["publication_role_manifest_sha256"] = sha256_file(roles_path)
    protocol["frozen_final_event_ids"] = list(final_events)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "formal_protocol_path": str(protocol_path),
                "formal_protocol_sha256": sha256_file(protocol_path),
                "publication_role_manifest_path": str(roles_path),
                "publication_role_manifest_sha256": sha256_file(roles_path),
                "learning_role_group_counts": {
                    role: value["rainfall_group_count"] for role, value in role_summary.items()
                },
                "frozen_final_event_ids": list(final_events),
                "new_empirical_training_data_generated": False,
                "ready_for_policy_lock": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
