"""Canonicalize reusable Project7 full exact-return truth into Train/Validation/Test.

Historical Train/Validation/Calibration roles and prior Step3 visibility are provenance only. The
builder reconstructs causal state/action rows from JSONL, JSON and NPZ, recovers missing causal
contexts from same-query peers, removes re-serialized copies of the same SWMM observation, quarantines
only genuinely ambiguous independent truth conflicts, and then makes one deterministic leakage-group
Train/Validation/Test split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from rtc.project7_v26_historical_supervision import (
    CanonicalCandidateRecord,
    ContextResolver,
    EXACT_TRUTH_FIELD,
    HISTORICAL_SUPERVISION_CONTRACT,
    HistoricalCandidateRecord,
    adjudicate_canonical_duplicates,
    canonicalize_record,
    deterministic_split,
    leakage_components,
    read_candidate_records,
    recover_missing_contexts,
    sha256_file,
)


V26_DATASET_CONTRACT = "PROJECT7_STEP3_V26_EXACT_RETURN_TRAIN_VALIDATION_TEST_V3"


def _source_paths(args: argparse.Namespace) -> list[Path]:
    selected: list[Path] = []
    for value in list(args.asset) + list(args.records_jsonl):
        selected.append(Path(value).resolve())
    if args.inventory:
        inventory_path = Path(args.inventory).resolve()
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        files = payload.get("reusable_files", [])
        if not isinstance(files, list):
            raise ValueError("inventory reusable_files must be a list")
        selected.extend(Path(str(value)).resolve() for value in files)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(path)
        deduped.append(path)
    if not deduped:
        raise ValueError("no historical candidate assets were supplied")
    return deduped


def _materialize_context(path: Path, context: dict[str, np.ndarray]) -> str:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            current_state=np.asarray(context["current_state"]),
            rainfall_scenarios=np.asarray(context["rainfall_scenarios"]),
            active_target=np.asarray(context["active_target"]),
            previous_actuator_flow=np.asarray(context["previous_actuator_flow"]),
        )
    return sha256_file(path)


def _leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "rainfall_group",
        "event_id",
        "causal_context_fingerprint_sha256",
        "leakage_group_id",
    )
    overlaps: dict[str, dict[str, list[str]]] = {}
    for field in fields:
        memberships: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row.get(field, "")).strip().lower()
            if value:
                memberships[value].add(str(row["split"]))
        bad = {key: sorted(value) for key, value in memberships.items() if len(value) > 1}
        if bad:
            overlaps[field] = bad
    return {
        "passed": not overlaps,
        "checked_identity_fields": list(fields),
        "cross_split_overlaps": overlaps,
    }


def _read_context_reference_records(path: Path) -> list[HistoricalCandidateRecord]:
    """Read provenance-only context rows; exact-return labels are not required."""
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else [payload]
    else:
        return []
    return [
        HistoricalCandidateRecord(dict(value), path, index)
        for index, value in enumerate(values)
        if isinstance(value, dict)
    ]


def _adjudication_public_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"resolved_derived_disagreements", "unresolved_conflicts"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", default=[], help="Historical JSONL/JSON/NPZ asset")
    parser.add_argument(
        "--records-jsonl",
        action="append",
        default=[],
        help="Backward-compatible alias for --asset; old role/version is provenance only",
    )
    parser.add_argument(
        "--inventory",
        help="Inventory JSON produced by audit_project7_v26_exact_return_inventory.py",
    )
    parser.add_argument("--study-root", help="Root used to repair stale historical context paths")
    parser.add_argument(
        "--context-records",
        action="append",
        default=[],
        help="Optional historical JSON/JSONL rows used only to recover causal context references",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source_paths = _source_paths(args)
    resolver = ContextResolver(study_root=args.study_root)
    raw_record_count = 0
    records_by_source: Counter[str] = Counter()
    all_historical: list[HistoricalCandidateRecord] = []
    for path in source_paths:
        historical = read_candidate_records(path)
        raw_record_count += len(historical)
        records_by_source[str(path)] += len(historical)
        all_historical.extend(historical)

    context_reference_paths = [Path(value).resolve() for value in args.context_records]
    reference_records: list[HistoricalCandidateRecord] = []
    for path in context_reference_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        reference_records.extend(_read_context_reference_records(path))

    context_recovery = recover_missing_contexts(
        all_historical,
        resolver=resolver,
        references=reference_records,
    )

    canonical: list[CanonicalCandidateRecord] = []
    rejected: Counter[str] = Counter()
    for record in all_historical:
        converted, reason = canonicalize_record(record, resolver=resolver)
        if converted is None:
            rejected[reason] += 1
        else:
            canonical.append(converted)
    if not canonical:
        raise ValueError("no reusable full exact-return records survived canonicalization")

    adjudication = adjudicate_canonical_duplicates(canonical)
    records = adjudication.records
    if not records:
        raise ValueError("no exact-return records remain after provenance adjudication")

    components = leakage_components(records)
    split_by_component = deterministic_split(
        components.values(),
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
    )

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V26 dataset output is not empty: {out_dir}")
    context_dir = out_dir / "contexts"
    context_dir.mkdir(parents=True, exist_ok=True)

    adjudication_path = out_dir / "V26_EXACT_RETURN_ADJUDICATION.json"
    adjudication_path.write_text(
        json.dumps(adjudication.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = dict(record.row)
        component = components[index]
        row["leakage_group_id"] = component
        row["split"] = split_by_component[component]
        context_path = context_dir / f"{row['causal_context_fingerprint_sha256']}.npz"
        row["context_npz"] = str(context_path)
        row["context_npz_sha256"] = _materialize_context(context_path, record.context)
        rows.append(row)

    audit = _leakage_audit(rows)
    if not audit["passed"]:
        raise RuntimeError(f"Train/Validation/Test leakage detected: {audit['cross_split_overlaps']}")

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["split"]),
            str(row["leakage_group_id"]),
            str(row["causal_context_fingerprint_sha256"]),
            str(row["candidate_first_target_sha256"]),
            str(row["query_set_id"]),
        ),
    )
    records_path = out_dir / "V26_EXACT_RETURN_RECORDS.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered_rows),
        encoding="utf-8",
    )

    split_counts: Counter[str] = Counter()
    split_groups: defaultdict[str, set[str]] = defaultdict(set)
    split_contexts: defaultdict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    independent_counts: Counter[str] = Counter()
    for row in ordered_rows:
        split = str(row["split"])
        split_counts[split] += 1
        split_groups[split].add(str(row["leakage_group_id"]))
        split_contexts[split].add(str(row["causal_context_fingerprint_sha256"]))
        source_counts[str(row["candidate_source"])] += 1
        format_counts[str(row["historical_source_format"])] += 1
        role_counts[str(row.get("historical_original_data_role", ""))] += 1
        independent_counts[
            "independent"
            if row.get("historical_truth_is_independent_observation") is True
            else "derived_recovery_only"
        ] += 1

    manifest = {
        "contract": V26_DATASET_CONTRACT,
        "historical_supervision_contract": HISTORICAL_SUPERVISION_CONTRACT,
        "records_jsonl": str(records_path),
        "records_sha256": sha256_file(records_path),
        "context_dir": str(context_dir),
        "adjudication_report": str(adjudication_path),
        "adjudication_report_sha256": sha256_file(adjudication_path),
        "seed": int(args.seed),
        "split_strategy": "DETERMINISTIC_CONNECTED_RAINFALL_EVENT_CONTEXT_GROUP_TRAIN_VALIDATION_TEST",
        "split_unit": "CONNECTED_RAINFALL_EVENT_CONTEXT_LEAKAGE_GROUP",
        "train_fraction_requested": float(args.train_fraction),
        "validation_fraction_requested": float(args.validation_fraction),
        "raw_candidate_record_count": raw_record_count,
        "eligible_before_exact_dedup": len(canonical),
        "record_count": len(ordered_rows),
        "independent_leakage_group_count": len(set(components.values())),
        "unique_causal_context_count": len(
            {row["causal_context_fingerprint_sha256"] for row in ordered_rows}
        ),
        "split_record_counts": dict(sorted(split_counts.items())),
        "split_group_counts": {key: len(value) for key, value in sorted(split_groups.items())},
        "split_context_counts": {key: len(value) for key, value in sorted(split_contexts.items())},
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "historical_source_format_counts": dict(sorted(format_counts.items())),
        "historical_original_role_counts": dict(sorted(role_counts.items())),
        "truth_observation_counts": dict(sorted(independent_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "context_recovery": context_recovery,
        "repaired_context_reference_count": int(context_recovery["repaired"]),
        "adjudication": _adjudication_public_summary(adjudication.report),
        "source_assets": [str(path) for path in source_paths],
        "context_reference_assets": [str(path) for path in context_reference_paths],
        "source_asset_sha256": {str(path): sha256_file(path) for path in source_paths},
        "source_raw_record_counts": dict(sorted(records_by_source.items())),
        "leakage_audit": audit,
        "truth_field": EXACT_TRUTH_FIELD,
        "truth_semantics": "AUTHORITATIVE_FULL_CANDIDATE_MINUS_HOLD_POLICY_RETURN_TFV_M3",
        "derived_dataset_copy_counts_as_new_swmm_observation": False,
        "unresolved_independent_truth_conflicts_are_quarantined": True,
        "any_single_conflict_aborts_entire_dataset": False,
        "h120_or_step2_targets_promoted_to_full_return": False,
        "old_roles_control_statistical_use": False,
        "previous_version_visibility_excludes_training": False,
        "step1_step2_prior_exposure_excludes_training": False,
        "only_new_split_membership_controls_step3_statistical_use": True,
        "test_used_for_model_fitting_or_validation_selection": False,
        "new_swmm_truth_generated": False,
        "development_only": True,
        "formal_evidence": False,
    }
    manifest_path = out_dir / "V26_EXACT_RETURN_DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
