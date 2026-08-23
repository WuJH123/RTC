"""Freeze a contamination-remediated, fixed-policy Project7 V23 scientific split.

The original v0.6.9 split is never edited. Pre-lock-exposed source-Final events are quarantined, and a
replacement Final6 is selected from already-prepared, exposure-free events using forcing descriptors
only. The resulting protocol is permanently FIXED_POLICY_NO_RETRAIN.

The superseded source split used fixed 60/120/180/240/300/360-minute Final cells. The remediated split
does not require those contaminated cells to be recreated. Instead it spans six distinct untouched
duration strata from the available prepared-event support, selected deterministically without opening
hydraulic or controller outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rtc.project7_v23_final_remediation import (
    REMEDIATED_SPLIT_CONTRACT,
    select_reblind_final,
    validate_exposure_ledger,
    validate_remediated_split,
)
from rtc.project7_v23_formal_reuse import (
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    validate_frozen_split,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-split-contract", required=True)
    parser.add_argument("--contamination-audit", required=True)
    parser.add_argument("--exposure-ledger", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    original_path = Path(args.original_split_contract).resolve()
    audit_path = Path(args.contamination_audit).resolve()
    ledger_path = Path(args.exposure_ledger).resolve()
    original = _json(original_path)
    audit = _json(audit_path)
    ledger = _json(ledger_path)
    original_roles = validate_frozen_split(original)
    candidates = validate_exposure_ledger(ledger)

    if audit.get("contract") != V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT:
        raise ValueError("contamination remediation requires the current V23 existing-truth audit")
    if audit.get("final_truth_contamination") is not True:
        raise ValueError("source split has no documented Final contamination to remediate")
    candidate_by_id = {row.event_id: row for row in candidates}
    missing_source_final = [
        event_id for event_id in original_roles["final"] if event_id not in candidate_by_id
    ]
    if missing_source_final:
        raise ValueError(
            "Final exposure ledger does not cover every source-Final event: "
            f"{missing_source_final}"
        )
    contaminated = tuple(
        sorted(
            event_id
            for event_id in original_roles["final"]
            if candidate_by_id[event_id].exposed_prelock
        )
    )
    if not contaminated:
        raise ValueError("exposure ledger does not reproduce the documented source-Final contamination")
    audit_count = int(audit.get("final_truth_record_count_detected_prelock", 0))
    if audit_count <= 0:
        raise ValueError("contamination audit lacks a positive pre-lock Final record count")
    if ledger.get("evidence_categories_complete") is not True:
        raise RuntimeError("Final exposure ledger is incomplete")
    if str(ledger.get("source_original_split_sha256", "")).lower() != _sha(original_path).lower():
        raise RuntimeError("Final exposure ledger was built against another source split")

    selected = select_reblind_final(
        candidates,
        protected_validation_events=original_roles["development_validation"],
        original_final_events=original_roles["final"],
    )
    selected_ids = tuple(row.event_id for row in selected)
    if set(selected_ids) & set(contaminated):
        raise RuntimeError("reblind Final selector attempted to reuse a contaminated source-Final event")
    selected_exposed = [row.event_id for row in selected if row.exposed_prelock]
    if selected_exposed:
        raise RuntimeError(f"reblind Final contains pre-lock exposure: {selected_exposed}")

    development_train = tuple(
        value for value in original_roles["development_train"] if value not in set(selected_ids)
    )
    validation = tuple(original_roles["development_validation"])
    clean_original_final = tuple(
        value for value in original_roles["final"] if value not in set(contaminated)
    )
    superseded_unassigned = tuple(
        value for value in clean_original_final if value not in set(selected_ids)
    )
    eligible_clean = [
        row
        for row in candidates
        if not row.exposed_prelock and row.event_id not in set(validation)
    ]
    eligible_durations = sorted({row.duration_minutes for row in eligible_clean})
    selected_durations = [row.duration_minutes for row in selected]

    payload: dict[str, Any] = {
        "contract": REMEDIATED_SPLIT_CONTRACT,
        "source_split_contract": str(original.get("contract")),
        "source_split_contract_path": str(original_path),
        "source_split_contract_sha256": _sha(original_path),
        "source_contamination_audit_path": str(audit_path),
        "source_contamination_audit_sha256": _sha(audit_path),
        "source_exposure_ledger_path": str(ledger_path),
        "source_exposure_ledger_sha256": _sha(ledger_path),
        "source_final_contamination_detected": True,
        "formal_mode": "FIXED_POLICY_NO_RETRAIN",
        "formal_retraining_allowed": False,
        "contaminated_records_deleted_or_relabeled": False,
        "remediation_reason": (
            "Source-Final events had pre-lock historical policy/model exposure. The source split is "
            "retained for provenance but is not publication-Final authority."
        ),
        "selection_basis": {
            "hydraulic_outcomes_used": False,
            "controller_performance_used": False,
            "forcing_descriptors_used": ["return_period_year", "duration_minutes"],
            "prelock_exposure_status_used": True,
            "source_duration_cells_required": False,
            "duration_design": (
                "six distinct exposure-free duration strata from the eligible prepared-event support; "
                "if more than six strata exist, use deterministic equally spaced duration order-statistic positions"
            ),
            "tie_break": (
                "within selected duration prefer still-clean source-Final; then new return-period "
                "coverage; then least-used return period; then event_id"
            ),
        },
        "eligible_clean_event_count": len(eligible_clean),
        "eligible_clean_duration_minutes": eligible_durations,
        "selected_final_duration_minutes": selected_durations,
        "development_train": list(development_train),
        "development_validation": list(validation),
        "final": list(selected_ids),
        "quarantined_prelock_exposed_final": list(contaminated),
        "retained_clean_source_final": [
            value for value in clean_original_final if value in set(selected_ids)
        ],
        "removed_from_source_development_train_for_reblind_final": [
            value for value in original_roles["development_train"] if value in set(selected_ids)
        ],
        "added_from_outside_source_split_for_reblind_final": [
            value
            for value in selected_ids
            if value
            not in set(
                original_roles["development_train"]
                + original_roles["development_validation"]
                + original_roles["final"]
            )
        ],
        "superseded_clean_source_final_not_selected": list(superseded_unassigned),
        "counts": {
            "development_train": len(development_train),
            "development_validation": len(validation),
            "final": len(selected_ids),
            "quarantined_final": len(contaminated),
        },
        "final_exposure_count_prelock": 0,
        "final_forcing_descriptors": [
            {
                "event_id": row.event_id,
                "return_period_year": row.return_period_year,
                "duration_minutes": row.duration_minutes,
                "prepared_inp_sha256": row.prepared_inp_sha256,
            }
            for row in selected
        ],
        "invariants": {
            "calibration_role_removed": True,
            "safety_audit_role_removed": True,
            "rainfall_groups_cross_scientific_splits": False,
            "development_groups_cross_train_validation": False,
            "final_untouched_before_policy_lock": True,
            "final_used_for_tuning": False,
            "validation_used_for_training": False,
            "contaminated_source_final_excluded_from_publication_final": True,
        },
        "new_rainfall_generated": False,
        "new_training_data_generated": False,
        "new_policy_return_truth_generated": False,
        "hydraulic_outcomes_opened_during_selection": False,
    }
    validate_remediated_split(payload)
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "remediated_split_path": str(destination),
                "remediated_split_sha256": _sha(destination),
                "quarantined_final": list(contaminated),
                "eligible_clean_duration_minutes": eligible_durations,
                "selected_final_duration_minutes": selected_durations,
                "reblind_final": list(selected_ids),
                "formal_mode": "FIXED_POLICY_NO_RETRAIN",
                "hydraulic_outcomes_opened_during_selection": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
