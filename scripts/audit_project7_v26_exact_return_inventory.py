"""Inventory reusable Project7 candidate-vs-HOLD exact-return JSONL artifacts.

This is a read-only discovery tool.  It scans a study tree for JSONL files, identifies records that
look like candidate-level policy-return truth, reports schema/source/rainfall coverage, and separates
obvious closed-loop benchmark/final products from candidate-return learning data.  It never decides
that data are unusable merely because an earlier Step3 version saw them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INVENTORY_CONTRACT = "PROJECT7_STEP3_V26_EXACT_RETURN_HISTORY_INVENTORY_V1"
CANDIDATE_FAMILIES = {
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return [], f"line {line_number} is not a JSON object"
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], str(exc)
    return rows, None


def _classify(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = str(path).lower()
    exact_rows = [row for row in rows if "true_policy_return_delta_tfv_m3" in row]
    candidate_rows = [
        row for row in exact_rows
        if str(row.get("candidate_source", "")) in CANDIDATE_FAMILIES
    ]
    query_rows = [row for row in candidate_rows if row.get("query_set_id") and row.get("rainfall_group")]
    target_rows = [
        row for row in query_rows
        if row.get("candidate_target") is not None or row.get("candidate_first_target") is not None
    ]
    context_rows = [row for row in query_rows if row.get("context_npz")]
    obvious_eval = any(
        token in names
        for token in (
            "benchmark",
            "formal",
            "final",
            "blind",
            "policy_lock",
            "policyl","
            "operational_benchmark",
        )
    )
    sources: dict[str, int] = {}
    roles: dict[str, int] = {}
    groups: set[str] = set()
    queries: set[str] = set()
    for row in candidate_rows:
        source = str(row.get("candidate_source", ""))
        role = str(row.get("data_role", row.get("v26_original_data_role", "")))
        sources[source] = sources.get(source, 0) + 1
        roles[role] = roles.get(role, 0) + 1
        if row.get("rainfall_group"):
            groups.add(str(row["rainfall_group"]))
        if row.get("query_set_id"):
            queries.add(str(row["query_set_id"]))
    reusable_candidate_learning = bool(target_rows and not obvious_eval)
    reason = (
        "candidate_exact_return_truth"
        if reusable_candidate_learning
        else "evaluation_output_path" if obvious_eval and candidate_rows
        else "no_candidate_exact_return_rows"
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "row_count": len(rows),
        "exact_return_row_count": len(exact_rows),
        "candidate_exact_return_row_count": len(candidate_rows),
        "query_identified_row_count": len(query_rows),
        "target_present_row_count": len(target_rows),
        "context_present_row_count": len(context_rows),
        "rainfall_group_count": len(groups),
        "query_count": len(queries),
        "candidate_source_counts": sources,
        "original_data_role_counts": roles,
        "obvious_closed_loop_or_formal_evaluation_path": obvious_eval,
        "reusable_candidate_learning_source": reusable_candidate_learning,
        "classification_reason": reason,
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
    parse_failures: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.jsonl")):
        rows, error = _rows(path)
        if error is not None:
            parse_failures.append({"path": str(path), "error": error})
            continue
        report = _classify(path, rows)
        if report["exact_return_row_count"] > 0 or report["candidate_exact_return_row_count"] > 0:
            reports.append(report)
    reusable = [report for report in reports if report["reusable_candidate_learning_source"]]
    source_totals: dict[str, int] = {}
    role_totals: dict[str, int] = {}
    for report in reusable:
        for source, count in report["candidate_source_counts"].items():
            source_totals[source] = source_totals.get(source, 0) + int(count)
        for role, count in report["original_data_role_counts"].items():
            role_totals[role] = role_totals.get(role, 0) + int(count)
    payload = {
        "contract": INVENTORY_CONTRACT,
        "root": str(root),
        "jsonl_with_exact_return_count": len(reports),
        "reusable_candidate_learning_file_count": len(reusable),
        "reusable_candidate_source_totals_before_cross_file_dedup": source_totals,
        "reusable_original_role_totals_before_cross_file_dedup": role_totals,
        "reusable_files": [report["path"] for report in reusable],
        "files": reports,
        "parse_failures": parse_failures,
        "policy": {
            "prior_version_visibility_is_not_an_exclusion": True,
            "train_validation_test_split_is_applied_later": True,
            "obvious_benchmark_final_formal_paths_are_not_auto_training_sources": True,
            "inventory_is_read_only": True,
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"inventory": str(out), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
