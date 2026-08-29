"""Build a provenance-safe V28 dataset by appending targeted q95 truth.

The V27 dataset is treated as an immutable base.  Targeted V28 records are
appended only when their causal context already exists in that base dataset;
the base context determines the new statistical split.  Same-context,
different-action records are retained, while an exact repeated observation is
counted once and an exact-key truth conflict is quarantined.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONTRACT = "PROJECT7_STEP3_V28_Q95_MATCHED_RESIDUAL_VALUE_AUGMENTED_DATASET_V1"
TRUTH_FIELD = "true_policy_return_delta_tfv_m3"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL contains a non-object record: {path}")
    return values


def _text(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "") or "").strip().lower()


def _action_sha(row: dict[str, Any]) -> str:
    for key in ("q95_supported_target_sha256", "candidate_first_target_sha256"):
        value = _text(row, key)
        if len(value) == 64:
            return value
    target = row.get("candidate_target")
    if not isinstance(target, list) or not target:
        raise ValueError("record lacks a candidate action identity")
    encoded = json.dumps([float(value) for value in target], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _continuation_sha(row: dict[str, Any]) -> str:
    for key in (
        "continuation_policy_sha256",
        "continuation_policy_hash",
        "dedup_continuation_sha256",
    ):
        value = _text(row, key)
        if value:
            return value
    # The V27 canonical rows do not expose a standalone continuation hash.  A
    # blank value is intentional: context fingerprint already binds the
    # causal prefix and the V27 contract binds its frozen continuation.
    return ""


def exact_observation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return state/action/continuation identity, excluding the truth value."""

    context = _text(row, "causal_context_fingerprint_sha256")
    if len(context) != 64:
        raise ValueError("record lacks causal_context_fingerprint_sha256")
    return context, _action_sha(row), _continuation_sha(row)


def leakage_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(row, "rainfall_group"),
        _text(row, "event_id") or _text(row, "rainfall_group"),
        _text(row, "causal_context_fingerprint_sha256"),
        _text(row, "leakage_group_id"),
    )


def _split_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_text(row, "split") for row in rows).items()))


def _unique_count(rows: Iterable[dict[str, Any]], key: str) -> int:
    return len({_text(row, key) for row in rows if _text(row, key)})


def audit_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "rainfall_group": defaultdict(set),
        "event_id": defaultdict(set),
        "causal_context_fingerprint_sha256": defaultdict(set),
        "leakage_group_id": defaultdict(set),
    }
    for row in rows:
        split = _text(row, "split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split!r}")
        for key in checks:
            value = _text(row, key)
            if key == "event_id" and not value:
                value = _text(row, "rainfall_group")
            if value:
                checks[key][value].add(split)
    overlaps = {
        key: sorted(value for value, splits in mapping.items() if len(splits) > 1)
        for key, mapping in checks.items()
    }
    passed = not any(overlaps.values())
    return {
        "passed": passed,
        "checked_identity_fields": list(checks),
        "cross_split_overlaps": overlaps,
    }


def _adapt_targeted_row(
    template: dict[str, Any],
    targeted: dict[str, Any],
    *,
    targeted_sha256: str,
) -> dict[str, Any]:
    row = copy.deepcopy(template)
    context = _text(targeted, "causal_context_fingerprint_sha256")
    if context != _text(template, "causal_context_fingerprint_sha256"):
        raise ValueError("targeted context does not match its template")
    row.update(
        {
            "contract": CONTRACT,
            "candidate_source": targeted.get("candidate_source", "TYPE_AWARE_HYDRAULIC_PRESSURE"),
            "candidate_target": targeted["candidate_target"],
            "candidate_first_target_sha256": targeted["q95_supported_target_sha256"],
            "q95_supported_target_sha256": targeted["q95_supported_target_sha256"],
            "q95_support_scale": targeted.get("q95_support_scale"),
            "q95_max_support_ratio": targeted.get("q95_max_support_ratio"),
            "q95_binding": targeted.get("q95_binding"),
            "true_policy_return_delta_tfv_m3": targeted[TRUTH_FIELD],
            "event_id": targeted.get("event_id", template.get("event_id", "")),
            "rainfall_group": targeted.get("rainfall_group", template.get("rainfall_group", "")),
            "query_set_id": targeted.get("query_set_id", template.get("query_set_id", "")),
            "context_npz": targeted.get("context_npz", template.get("context_npz")),
            "context_npz_sha256": targeted.get("context_npz_sha256", template.get("context_npz_sha256")),
            "historical_source_path": targeted.get("candidate_metadata_path"),
            "historical_origin_source_path": targeted.get("candidate_metadata_path"),
            "historical_original_context_path": targeted.get("source_context_path"),
            "historical_original_context_sha256": targeted.get("context_npz_sha256"),
            "historical_source_format": "targeted_swmm_jsonl",
            "historical_source_index": targeted.get("decision_index"),
            "historical_original_data_role": targeted.get("data_role", "policy_return_train"),
            "historical_is_derived_canonical_copy": False,
            "historical_truth_is_independent_observation": True,
            "v28_targeted_truth_record_sha256": targeted_sha256,
            "v28_targeted_truth_contract": targeted.get("contract"),
            "new_swmm_truth_generated": True,
            "new_rainfall_generated": False,
            "formal_evidence": False,
            "development_only": True,
            "h120_truth_used_for_training": False,
        }
    )
    return row


def augment_dataset(
    *,
    base_manifest_path: str | Path,
    base_records_path: str | Path,
    targeted_records_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"V28 augmented dataset output is not empty: {out}")
    base_manifest_path = Path(base_manifest_path).resolve()
    base_records_path = Path(base_records_path).resolve()
    targeted_records_path = Path(targeted_records_path).resolve()
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_rows = read_jsonl(base_records_path)
    targeted_rows = read_jsonl(targeted_records_path)
    if sha256_file(base_records_path).lower() != str(base_manifest.get("records_sha256", "")).lower():
        raise ValueError("base V27 records SHA mismatch")
    if len(base_rows) != int(base_manifest.get("record_count", -1)):
        raise ValueError("base V27 record count mismatch")
    if not targeted_rows:
        raise ValueError("targeted V28 truth is empty")
    if any(TRUTH_FIELD not in row for row in targeted_rows):
        raise ValueError("targeted truth lacks exact policy-return target")

    templates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        templates[_text(row, "causal_context_fingerprint_sha256")].append(row)
    if any(not key or not values for key, values in templates.items()):
        raise ValueError("base V27 dataset has an invalid causal context")

    out_rows = list(base_rows)
    existing_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        existing_by_key[exact_observation_key(row)].append(row)
    appended: list[dict[str, Any]] = []
    duplicate_count = 0
    conflict_keys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    repaired_contexts = 0
    for targeted in targeted_rows:
        context = _text(targeted, "causal_context_fingerprint_sha256")
        if context not in templates:
            rejected["targeted_context_not_in_base_dataset"] += 1
            continue
        candidate_truth = float(targeted[TRUTH_FIELD])
        candidate_key = exact_observation_key(
            {
                **targeted,
                "candidate_first_target_sha256": targeted.get("q95_supported_target_sha256"),
            }
        )
        existing = existing_by_key.get(candidate_key, [])
        if existing:
            if all(abs(float(row[TRUTH_FIELD]) - candidate_truth) <= 1.0e-6 for row in existing):
                duplicate_count += 1
                continue
            conflict_keys["|".join(candidate_key)].extend(existing)
            conflict_keys["|".join(candidate_key)].append(targeted)
            rejected["unresolved_independent_truth_conflict"] += 1
            continue
        template = templates[context][0]
        row = _adapt_targeted_row(template, targeted, targeted_sha256=sha256_file(targeted_records_path))
        # The base context, not the historical source role, controls the new
        # split and leakage group.  This is the key protection against role
        # leakage while retaining historical truth.
        row["split"] = template["split"]
        row["leakage_group_id"] = template["leakage_group_id"]
        row["causal_context_fingerprint_sha256"] = template["causal_context_fingerprint_sha256"]
        appended.append(row)
        existing_by_key[candidate_key].append(row)
        repaired_contexts += 1

    out_rows.extend(appended)
    leakage = audit_leakage(out_rows)
    if not leakage["passed"]:
        raise ValueError(f"V28 augmented dataset leakage audit failed: {leakage}")
    split_counts = _split_counts(out_rows)
    if any(split_counts.get(name, 0) == 0 for name in ("train", "validation", "test")):
        raise ValueError("V28 augmented dataset has an empty split")
    split_groups = {
        name: len({_text(row, "leakage_group_id") for row in out_rows if _text(row, "split") == name})
        for name in ("train", "validation", "test")
    }
    split_contexts = {
        name: len(
            {
                _text(row, "causal_context_fingerprint_sha256")
                for row in out_rows
                if _text(row, "split") == name
            }
        )
        for name in ("train", "validation", "test")
    }
    source_counts = dict(sorted(Counter(_text(row, "candidate_source") for row in out_rows).items()))
    format_counts = dict(sorted(Counter(_text(row, "historical_source_format") for row in out_rows).items()))
    role_counts = dict(sorted(Counter(_text(row, "historical_original_data_role") for row in out_rows).items()))
    truth_counts = dict(
        sorted(
            Counter(
                "independent"
                if bool(row.get("historical_truth_is_independent_observation", True))
                else "derived"
                for row in out_rows
            ).items()
        )
    )
    records_path = out / "V28_Q95_MATCHED_AUGMENTED_EXACT_RETURN_RECORDS.jsonl"
    manifest_path = out / "V28_Q95_MATCHED_AUGMENTED_EXACT_RETURN_DATASET_MANIFEST.json"
    out.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "contract": CONTRACT,
        "development_only": True,
        "formal_evidence": False,
        "base_dataset_manifest": str(base_manifest_path),
        "base_dataset_manifest_sha256": sha256_file(base_manifest_path),
        "base_dataset_records": str(base_records_path),
        "base_dataset_records_sha256": sha256_file(base_records_path),
        "targeted_truth_records": str(targeted_records_path),
        "targeted_truth_records_sha256": sha256_file(targeted_records_path),
        "raw_candidate_record_count": int(base_manifest.get("raw_candidate_record_count", len(base_rows)) + len(targeted_rows)),
        "eligible_before_exact_dedup": int(base_manifest.get("eligible_before_exact_dedup", len(base_rows)) + len(targeted_rows)),
        "exact_duplicate_count": int(base_manifest.get("exact_duplicate_count", 0) + duplicate_count),
        "record_count": len(out_rows),
        "records": str(records_path),
        "records_sha256": sha256_file(records_path),
        "unique_causal_context_count": _unique_count(out_rows, "causal_context_fingerprint_sha256"),
        "independent_leakage_group_count": _unique_count(out_rows, "leakage_group_id"),
        "split_record_counts": split_counts,
        "split_group_counts": split_groups,
        "split_context_counts": split_contexts,
        "split_strategy": "INHERIT_IMMUTABLE_V27_CONTEXT_SPLIT_FOR_TARGETED_Q95_ACTIONS_V1",
        "candidate_source_counts": source_counts,
        "historical_source_format_counts": format_counts,
        "historical_original_role_counts": role_counts,
        "truth_observation_counts": truth_counts,
        "rejected_counts": dict(sorted(rejected.items())),
        "context_recovery": {
            "base_already_available": int(base_manifest.get("context_recovery", {}).get("already_available", 0)),
            "targeted_already_available": len(appended),
            "targeted_missing": int(rejected.get("targeted_context_not_in_base_dataset", 0)),
        },
        "repaired_context_reference_count": repaired_contexts,
        "adjudication": {
            "inherited_from_base": True,
            "appended_targeted_record_count": len(appended),
            "derived_copy_dropped_count": int(base_manifest.get("adjudication", {}).get("derived_copy_dropped_count", 0)),
            "resolved_derived_disagreement_key_count": int(
                base_manifest.get("adjudication", {}).get("resolved_derived_disagreement_key_count", 0)
            ),
            "unresolved_conflict_key_count": len(conflict_keys),
            "quarantined_conflict_record_count": sum(len(values) for values in conflict_keys.values()),
            "conflicts": conflict_keys,
        },
        "targeted_truth_reuse": {
            "selected_context_count": len(targeted_rows),
            "included_count": len(appended),
            "duplicate_truth_skipped_count": duplicate_count,
            "new_swmm_truth_generated_count": sum(bool(row.get("new_swmm_truth_generated")) for row in targeted_rows),
        },
        "leakage_audit": leakage,
        "truth_field": TRUTH_FIELD,
        "h120_role": "diagnostic_only",
        "test_used_for_training_or_model_selection": False,
        "ready_for_policy_lock": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--base-records", required=True)
    parser.add_argument("--targeted-records", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            augment_dataset(
                base_manifest_path=args.base_manifest,
                base_records_path=args.base_records,
                targeted_records_path=args.targeted_records,
                out_dir=args.out_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
