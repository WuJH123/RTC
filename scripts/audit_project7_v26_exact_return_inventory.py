"""Inventory every reusable Project7 full exact-return candidate asset under a study tree.

JSONL, JSON and NPZ historical assets are visible. Path names and old data roles do not exclude a
semantically valid record. Statistical use is decided later by the new Train/Validation/Test split.
Damaged unrelated files are reported as parse failures and do not abort discovery of the remaining
historical SWMM truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
import zipfile

from rtc.project7_v26_historical_supervision import (
    HISTORICAL_SUPERVISION_CONTRACT,
    SUPPORTED_SUFFIXES,
    exact_truth,
    read_candidate_records,
    sha256_file,
)


INVENTORY_CONTRACT = "PROJECT7_STEP3_V26_EXACT_RETURN_HISTORY_INVENTORY_V3"


def _report(path: Path) -> dict[str, Any] | None:
    records = read_candidate_records(path)
    if not records:
        return None
    sources: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    groups: set[str] = set()
    events: set[str] = set()
    queries: set[str] = set()
    target_direct = 0
    context_refs = 0
    embedded = 0
    exact = 0
    derived_copies = 0
    for record in records:
        row = record.row
        if exact_truth(row) is not None:
            exact += 1
        source = str(row.get("candidate_source", "") or "HISTORICAL_EXACT_ACTION")
        role = str(row.get("data_role", row.get("source_data_role", "")))
        sources[source] += 1
        roles[role] += 1
        if row.get("rainfall_group"):
            groups.add(str(row["rainfall_group"]))
        if row.get("event_id"):
            events.add(str(row["event_id"]))
        if row.get("query_set_id"):
            queries.add(str(row["query_set_id"]))
        if row.get("candidate_target") is not None or row.get("candidate_first_target") is not None:
            target_direct += 1
        if row.get("context_npz"):
            context_refs += 1
        if record.embedded_context is not None:
            embedded += 1
        if row.get("historical_supervision_contract"):
            derived_copies += 1
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "format": path.suffix.lower().lstrip("."),
        "candidate_exact_return_row_count": exact,
        "rainfall_group_count": len(groups),
        "event_id_count": len(events),
        "query_count": len(queries),
        "direct_target_row_count": target_direct,
        "context_reference_row_count": context_refs,
        "embedded_context_row_count": embedded,
        "prior_canonical_copy_row_count": derived_copies,
        "candidate_source_counts": dict(sorted(sources.items())),
        "original_data_role_counts": dict(sorted(roles.items())),
        "reusable_candidate_learning_source": exact > 0,
        "prior_version_or_role_is_not_an_exclusion": True,
        "path_name_is_not_an_exclusion": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    scanned_by_format: Counter[str] = Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        scanned_by_format[path.suffix.lower().lstrip(".")] += 1
        try:
            report = _report(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, zipfile.BadZipFile) as exc:
            failures.append({"path": str(path), "error": str(exc)})
            continue
        if report is not None:
            reports.append(report)

    reusable = [item for item in reports if item["reusable_candidate_learning_source"]]
    source_totals: Counter[str] = Counter()
    role_totals: Counter[str] = Counter()
    format_totals: Counter[str] = Counter()
    raw_rows = 0
    prior_copy_rows = 0
    for item in reusable:
        raw_rows += int(item["candidate_exact_return_row_count"])
        prior_copy_rows += int(item["prior_canonical_copy_row_count"])
        format_totals[str(item["format"])] += int(item["candidate_exact_return_row_count"])
        source_totals.update(item["candidate_source_counts"])
        role_totals.update(item["original_data_role_counts"])

    payload = {
        "contract": INVENTORY_CONTRACT,
        "historical_supervision_contract": HISTORICAL_SUPERVISION_CONTRACT,
        "root": str(root),
        "supported_formats": sorted(suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES),
        "scanned_file_counts_by_format": dict(sorted(scanned_by_format.items())),
        "candidate_asset_count": len(reports),
        "reusable_candidate_learning_file_count": len(reusable),
        "candidate_exact_return_rows_before_canonicalization_and_dedup": raw_rows,
        "prior_canonical_copy_rows_before_dedup": prior_copy_rows,
        "candidate_exact_return_rows_by_format": dict(sorted(format_totals.items())),
        "candidate_source_totals_before_cross_file_dedup": dict(sorted(source_totals.items())),
        "original_role_totals_before_cross_file_dedup": dict(sorted(role_totals.items())),
        "reusable_files": [item["path"] for item in reusable],
        "files": reports,
        "parse_failures": failures,
        "policy": {
            "old_train_validation_calibration_roles_are_provenance_only": True,
            "prior_step3_version_visibility_is_not_an_exclusion": True,
            "step1_step2_prior_exposure_is_not_an_exclusion": True,
            "formal_final_benchmark_path_names_are_not_exclusion_rules": True,
            "full_exact_return_semantics_are_required": True,
            "prior_canonical_copy_is_inventory_visible_but_not_a_new_swmm_observation": True,
            "damaged_unrelated_asset_does_not_abort_inventory": True,
            "train_validation_test_split_is_applied_after_canonical_dedup": True,
            "inventory_is_read_only": True,
            "new_swmm_truth_generated": False,
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"inventory": str(out), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
