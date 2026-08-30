"""Build a zero-SWMM, policy-consistent view of an existing exact-return JSONL bank.

The historical source is never edited. Missing continuation hashes are recovered only from a unique
explicit continuation hash on the exact same causal-context fingerprint. Rows that cannot be resolved
remain in the full derived audit view but are excluded from the target-continuation training view.

The target-view manifest is deliberately compatible with the frozen V27 trainer.  This permits one
provenance-clean refit with the *same* architecture, feature builder and hyperparameter grids while
removing pointwise exact-return supervision from other continuation policies.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from rtc.project7_policy_consistency import (
    POLICY_CONSISTENCY_CONTRACT,
    RESOLVED_CONTINUATION_FIELD,
    normalize_sha256,
    policy_consistent_decision_unit,
    resolve_continuations,
    select_target_continuation,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("records JSONL must contain at least one object row")
    return [dict(row) for row in rows]


def _leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "rainfall_group",
        "event_id",
        "causal_context_fingerprint_sha256",
        "leakage_group_id",
        "policy_consistent_decision_unit",
    )
    overlaps: dict[str, dict[str, list[str]]] = {}
    for field in fields:
        membership: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row.get(field, "")).strip().lower()
            split = str(row.get("split", "")).strip().lower()
            if value and split:
                membership[value].add(split)
        bad = {key: sorted(values) for key, values in membership.items() if len(values) > 1}
        if bad:
            overlaps[field] = bad
    return {
        "passed": not overlaps,
        "checked_identity_fields": list(fields),
        "cross_split_overlaps": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--target-continuation-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.records).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target_continuation = normalize_sha256(args.target_continuation_sha256)
    if not target_continuation:
        raise ValueError("target continuation must be one 64-hex SHA256")
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolution = resolve_continuations(_rows(source))
    selected, selection_report = select_target_continuation(
        resolution.rows,
        target_continuation_sha256=target_continuation,
    )
    for row in selected:
        row["policy_consistent_decision_unit"] = policy_consistent_decision_unit(row)
        if normalize_sha256(row.get(RESOLVED_CONTINUATION_FIELD)) != target_continuation:
            raise RuntimeError("selected target-continuation row drifted from the frozen estimand")

    split_values = {str(row.get("split", "")).strip().lower() for row in selected}
    if split_values != {"train", "validation", "test"}:
        raise ValueError(
            "policy-consistent target view must preserve the frozen Train/Validation/Test split"
        )
    leakage_audit = _leakage_audit(selected)
    if leakage_audit["passed"] is not True:
        raise RuntimeError(
            f"policy-consistent target view has cross-split leakage: {leakage_audit['cross_split_overlaps']}"
        )

    full_path = out_dir / "POLICY_CONSISTENT_ALL_ROWS.jsonl"
    full_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in resolution.rows),
        encoding="utf-8",
    )
    selected_path = out_dir / "POLICY_CONSISTENT_TARGET_CONTINUATION_ROWS.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )

    split_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    split_groups: defaultdict[str, set[str]] = defaultdict(set)
    split_contexts: defaultdict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, int] = {}
    for row in selected:
        split = str(row.get("split", "")).strip().lower()
        split_counts[split] = split_counts.get(split, 0) + 1
        split_groups[split].add(str(row.get("leakage_group_id", "")))
        split_contexts[split].add(str(row.get("causal_context_fingerprint_sha256", "")))
        provenance = str(row.get("resolved_continuation_provenance", ""))
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
        candidate_source = str(row.get("candidate_source", ""))
        source_counts[candidate_source] = source_counts.get(candidate_source, 0) + 1
        if not str(row.get(RESOLVED_CONTINUATION_FIELD, "")):
            raise RuntimeError("selected target-continuation row lost continuation identity")

    manifest = {
        "contract": POLICY_CONSISTENCY_CONTRACT,
        "source_records": str(source),
        "source_records_sha256": _sha(source),
        "all_rows": str(full_path),
        "all_rows_sha256": _sha(full_path),
        "target_rows": str(selected_path),
        "target_rows_sha256": _sha(selected_path),
        # V27-trainer compatibility.  The trainer still checks the original frozen split and uses
        # the same feature/model code; only the row view is policy-consistent.
        "records_jsonl": str(selected_path),
        "records_sha256": _sha(selected_path),
        "record_count": len(selected),
        "leakage_audit": leakage_audit,
        "split_counts": dict(sorted(split_counts.items())),
        "split_group_counts": {
            split: len(groups) for split, groups in sorted(split_groups.items())
        },
        "split_context_counts": {
            split: len(contexts) for split, contexts in sorted(split_contexts.items())
        },
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "target_continuation_policy_sha256": target_continuation,
        "resolution": resolution.report,
        "selection": selection_report,
        "selected_split_counts": dict(sorted(split_counts.items())),
        "selected_provenance_counts": dict(sorted(provenance_counts.items())),
        "pairwise_decision_unit_includes_continuation": True,
        "pointwise_training_rows_are_target_continuation_only": True,
        "v27_architecture_or_hyperparameter_grid_changed": False,
        "historical_source_mutated": False,
        "new_swmm_truth_generated": False,
        "new_rainfall_generated": False,
        "model_training_performed": False,
        "development_only": True,
        "ready_for_policy_lock": False,
    }
    manifest_path = out_dir / "POLICY_CONSISTENT_VIEW_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
