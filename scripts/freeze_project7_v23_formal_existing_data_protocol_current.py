"""Freeze the Project7 V23 publication protocol using existing exact-matched truth only.

This command is intentionally downstream of ``audit_project7_v23_existing_truth_reuse_current.py``.
It creates only role/protocol manifests.  It does not train a model, run SWMM, create rainfall, or
change any authoritative label.  Final-test rainfall groups are sealed here and must not influence
model fitting, threshold selection, candidate design, or further policy steering.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_v23_formal_reuse import (
    DEFAULT_ROLE_FRACTIONS,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    V23_FORMAL_PROTOCOL_CONTRACT,
    V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
    assert_excluded_events_absent,
    deterministic_publication_split,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("matched existing-truth file contains no valid records")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-audit", required=True)
    parser.add_argument("--matched-records-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audit_path = Path(args.reuse_audit).resolve()
    matched_path = Path(args.matched_records_jsonl).resolve()
    audit = _read_json(audit_path)
    if audit.get("contract") != V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT:
        raise ValueError("wrong V23 existing-truth reuse audit contract")
    if audit.get("formal_training_allowed") is not True:
        raise RuntimeError(
            "V23 existing-truth exact-match coverage is insufficient; publication training must "
            "fail closed rather than create new labels or reuse mismatched truth"
        )
    for key in (
        "no_new_empirical_data",
        "nearest_neighbor_matching_used",
        "interpolation_used",
        "pseudo_labels_used",
    ):
        expected = key == "no_new_empirical_data"
        if audit.get(key) is not expected:
            raise RuntimeError(f"V23 reuse audit violates publication reuse firewall: {key}")

    records = _read_records(matched_path)
    eligible_queries = {
        str(row["query_set_id"])
        for row in audit.get("query_reports", [])
        if row.get("publication_role_eligible") is True
    }
    if not eligible_queries:
        raise RuntimeError("reuse audit contains no publication-eligible exact query")
    records = [row for row in records if str(row.get("query_set_id")) in eligible_queries]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_ids_by_group: dict[str, set[str]] = defaultdict(set)
    for row in records:
        group = str(row["rainfall_group"])
        by_group[group].append(row)
        event_ids_by_group[group].add(str(row["event_id"]))
    groups_from_audit = {
        str(row["rainfall_group"])
        for row in audit.get("query_reports", [])
        if row.get("publication_role_eligible") is True
    }
    if set(by_group) != groups_from_audit:
        raise RuntimeError("matched-record file and reuse audit disagree on eligible rainfall groups")

    roles = deterministic_publication_split(
        sorted(by_group),
        seed=int(args.seed),
        fractions=DEFAULT_ROLE_FRACTIONS,
    )
    excluded_events = tuple(str(value) for value in audit.get("excluded_development_event_ids", []))
    assert_excluded_events_absent(event_ids_by_group, roles, excluded_events)

    role_summary: dict[str, Any] = {}
    for role, groups in roles.items():
        group_set = set(groups)
        role_rows = [row for row in records if str(row["rainfall_group"]) in group_set]
        role_summary[role] = {
            "rainfall_groups": list(groups),
            "rainfall_group_count": len(groups),
            "query_set_count": len({str(row["query_set_id"]) for row in role_rows}),
            "existing_record_count": len(role_rows),
            "event_ids": sorted({str(row["event_id"]) for row in role_rows}),
            "candidate_sources": sorted({str(row["candidate_source"]) for row in role_rows}),
        }

    role_manifest = {
        "contract": V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
        "existing_data_only": True,
        "split_unit": "RAINFALL_GROUP",
        "split_seed": int(args.seed),
        "target_role_fractions": list(DEFAULT_ROLE_FRACTIONS),
        "rainfall_group_overlap_allowed": False,
        "development_steering_events_excluded_from_all_publication_roles": True,
        "excluded_development_event_ids": list(excluded_events),
        "source_reuse_audit_path": str(audit_path),
        "source_reuse_audit_sha256": sha256_file(audit_path),
        "source_matched_records_path": str(matched_path),
        "source_matched_records_sha256": sha256_file(matched_path),
        "roles": role_summary,
    }

    protocol = {
        "contract": V23_FORMAL_PROTOCOL_CONTRACT,
        "publication_roles_contract": V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
        "existing_data_only": True,
        "new_training_truth_allowed": False,
        "new_rainfall_allowed": False,
        "new_policy_return_truth_allowed": False,
        "development_benchmark_can_tune_formal_policy": False,
        "final_test_can_tune_any_model_or_threshold": False,
        "candidate_generator_role": "FROZEN_V23_AFTER_OPERATIONAL_DEVELOPMENT",
        "step1_role": "FROZEN_EXISTING_ACCEPTED_ASSET_SUBJECT_TO_ACCEPTANCE_AUDIT",
        "step2_role": "FROZEN_EXISTING_ACCEPTED_ASSET_SUBJECT_TO_ACCEPTANCE_AUDIT",
        "step3_training_scope": (
            "ONLY_EXISTING_V23_EXACT_MATCHED_QUERY_TRUTH; audit rank first; retrain selected-vs-HOLD "
            "boundary and rank only when the preregistered validation role demonstrates need"
        ),
        "primary_control_metric": "TFV_M3_AUTHORITATIVE_SWMM",
        "safety_metric": "PFV_M3_AUTHORITATIVE_SWMM",
        "pfv_safety_contract": "PFV_PROPOSED <= 100_M3 + 1.05 * PFV_NO_CONTROL",
        "global_peak_role": "REPORT_ONLY",
        "competitive_comparators": ["no_control", "internal_rtc", "auto_rbc", "efd"],
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
        "statistical_unit": "RAINFALL_EVENT_GROUP",
        "decision_rows_are_independent_replicates": False,
        "report_event_balanced_results": True,
        "report_paired_event_differences": True,
        "report_uncertainty": "PAIRED_EVENT_BOOTSTRAP_CI_WHEN_EVENT_COUNT_SUPPORTS_IT",
        "policy_lock_required_before_final_test": True,
        "ready_for_policy_lock": False,
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roles_path = out_dir / "V23_PUBLICATION_ROLE_MANIFEST.json"
    protocol_path = out_dir / "V23_FORMAL_REUSE_ONLY_PROTOCOL.json"
    roles_path.write_text(json.dumps(role_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol["publication_role_manifest_path"] = str(roles_path)
    protocol["publication_role_manifest_sha256"] = sha256_file(roles_path)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "formal_protocol_path": str(protocol_path),
                "formal_protocol_sha256": sha256_file(protocol_path),
                "publication_role_manifest_path": str(roles_path),
                "publication_role_manifest_sha256": sha256_file(roles_path),
                "role_group_counts": {
                    role: value["rainfall_group_count"] for role, value in role_summary.items()
                },
                "new_empirical_data_generated": False,
                "ready_for_policy_lock": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
