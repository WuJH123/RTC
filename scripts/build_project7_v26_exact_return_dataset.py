"""Consolidate existing Project7 exact policy-return truth into a simple Train/Validation/Test bank.

V26 deliberately reuses valid historical Development truth.  A record is eligible when it contains
an actual candidate action, causal context and exact full-branch candidate-vs-HOLD TFV return.
Records are not rejected merely because an earlier Step3 version saw them.  The only statistical
separation enforced here is rainfall-group disjointness across the newly frozen three-way split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rtc.direct_tfv_policy_return_portfolio_admission import CURRENT_THREE_FAMILY_SOURCES


V26_DATASET_CONTRACT = "PROJECT7_STEP3_V26_EXACT_RETURN_TRAIN_VALIDATION_TEST_V1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> bool:
    raw = str(value or "").strip().lower()
    return len(raw) == 64 and all(char in "0123456789abcdef" for char in raw)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL object in {path}")
    return rows


def _target(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("candidate_target")
    if value is None:
        value = row.get("candidate_first_target")
    if value is None:
        return None
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.shape != (109,) or not np.isfinite(out).all():
        return None
    return out


def _action_hash(row: dict[str, Any], target: np.ndarray) -> str:
    explicit = str(row.get("candidate_first_target_sha256", "")).strip().lower()
    if _canonical_sha(explicit):
        return explicit
    return hashlib.sha256(np.ascontiguousarray(target, dtype=np.float64).tobytes(order="C")).hexdigest()


def _context_index(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Collect every historical context reference; do not collapse same-query paths across versions."""
    index: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for row in _read_jsonl(path):
            query = str(row.get("query_set_id", "")).strip()
            context = str(row.get("context_npz", "")).strip()
            if not query or not context:
                continue
            context_sha = str(row.get("context_npz_sha256", "")).strip().lower()
            key = (query, context, context_sha)
            if key in seen:
                continue
            seen.add(key)
            index.setdefault(query, []).append(dict(row))
    return index


def _resolve_context(
    row: dict[str, Any],
    *,
    context_index: dict[str, list[dict[str, Any]]],
    query: str,
) -> tuple[str | None, str, str]:
    """Resolve a usable context without allowing one stale historical path to kill the dataset."""
    direct = str(row.get("context_npz", "")).strip()
    direct_sha = str(row.get("context_npz_sha256", "")).strip().lower()
    if direct:
        direct_path = Path(direct).resolve()
        if direct_path.is_file():
            observed_sha = _sha(direct_path)
            if _canonical_sha(direct_sha) and observed_sha.lower() != direct_sha:
                return None, "", "context_sha_mismatch"
            return str(direct_path), observed_sha, "resolved"

    available: list[tuple[Path, str]] = []
    for candidate in context_index.get(query, []):
        raw = str(candidate.get("context_npz", "")).strip()
        if not raw:
            continue
        path = Path(raw).resolve()
        if not path.is_file():
            continue
        observed_sha = _sha(path)
        candidate_sha = str(candidate.get("context_npz_sha256", "")).strip().lower()
        if _canonical_sha(candidate_sha) and candidate_sha != observed_sha.lower():
            continue
        if _canonical_sha(direct_sha) and direct_sha != observed_sha.lower():
            continue
        identity = (path, observed_sha)
        if identity not in available:
            available.append(identity)
    if len(available) == 1:
        return str(available[0][0]), available[0][1], "resolved"
    if not available:
        return None, "", "missing_causal_context"
    return None, "", "ambiguous_causal_context"


def _eligible_row(
    row: dict[str, Any],
    *,
    context_index: dict[str, list[dict[str, Any]]],
    source_file: Path,
) -> tuple[dict[str, Any] | None, str]:
    source = str(row.get("candidate_source", ""))
    if source not in set(CURRENT_THREE_FAMILY_SOURCES):
        return None, "non_current_candidate_family"
    query = str(row.get("query_set_id", "")).strip()
    group = str(row.get("rainfall_group", "")).strip()
    if not query or not group:
        return None, "missing_query_or_rainfall_group"
    truth_value = row.get("true_policy_return_delta_tfv_m3")
    try:
        truth = float(truth_value)
    except (TypeError, ValueError):
        return None, "missing_exact_return_truth"
    if not math.isfinite(truth):
        return None, "nonfinite_exact_return_truth"
    target = _target(row)
    if target is None:
        return None, "missing_candidate_target"
    for flag in (
        "same_prefix_verified",
        "same_continuation_policy_verified",
        "candidate_target_write_readback_verified",
        "target_write_readback_verified",
    ):
        if flag in row and row.get(flag) is False:
            return None, f"failed_{flag}"
    if "candidate_branch_tfv_m3" in row and "hold_branch_tfv_m3" in row:
        expected = float(row["candidate_branch_tfv_m3"]) - float(row["hold_branch_tfv_m3"])
        if not math.isfinite(expected) or abs(expected - truth) > 1.0e-6:
            return None, "exact_return_arithmetic_mismatch"

    context, context_sha, context_reason = _resolve_context(
        row,
        context_index=context_index,
        query=query,
    )
    if context is None:
        return None, context_reason

    out = dict(row)
    out.update(
        {
            "v26_dataset_contract": V26_DATASET_CONTRACT,
            "query_set_id": query,
            "rainfall_group": group,
            "candidate_source": source,
            "candidate_target": target.tolist(),
            "candidate_first_target_sha256": _action_hash(row, target),
            "true_policy_return_delta_tfv_m3": truth,
            "context_npz": context,
            "context_npz_sha256": context_sha,
            "v26_source_jsonl": str(source_file),
            "v26_original_data_role": str(row.get("data_role", "")),
            "h120_truth_used_for_training": False,
        }
    )
    return out, "eligible"


def _split_groups(groups: list[str], *, seed: int, train_fraction: float, validation_fraction: float) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("V26 requires at least three rainfall groups for Train/Validation/Test")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must lie in (0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train + Validation fractions must leave a non-empty Test fraction")
    ordered = sorted(
        set(groups),
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode("utf-8")).hexdigest(),
    )
    n = len(ordered)
    n_train = min(n - 2, max(1, int(round(n * train_fraction))))
    n_validation = min(n - n_train - 1, max(1, int(round(n * validation_fraction))))
    result: dict[str, str] = {}
    for group in ordered[:n_train]:
        result[group] = "train"
    for group in ordered[n_train : n_train + n_validation]:
        result[group] = "validation"
    for group in ordered[n_train + n_validation :]:
        result[group] = "test"
    if set(result.values()) != {"train", "validation", "test"}:
        raise RuntimeError("V26 three-way split collapsed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", action="append", required=True)
    parser.add_argument("--context-records", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source_paths = [Path(value).resolve() for value in args.records_jsonl]
    context_paths = [Path(value).resolve() for value in args.context_records]
    for path in source_paths + context_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    contexts = _context_index(context_paths + source_paths)

    eligible: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for path in source_paths:
        for row in _read_jsonl(path):
            converted, reason = _eligible_row(row, context_index=contexts, source_file=path)
            if converted is None:
                rejected[reason] = rejected.get(reason, 0) + 1
            else:
                eligible.append(converted)
    if not eligible:
        raise ValueError("V26 found no eligible exact-return records")

    # Collapse only exact state/action duplicates. Distinct contexts and distinct actions remain
    # training evidence even when an older Step3 version assigned them the same query_set_id.
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for row in eligible:
        key = (
            str(row["rainfall_group"]),
            str(row["query_set_id"]),
            str(row["context_npz_sha256"]),
            str(row["candidate_first_target_sha256"]),
        )
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = row
            continue
        if abs(float(previous["true_policy_return_delta_tfv_m3"]) - float(row["true_policy_return_delta_tfv_m3"])) > 1.0e-6:
            raise ValueError(f"duplicate exact state/action has conflicting truth: {key}")
        duplicate_count += 1
    rows = list(deduped.values())
    split_by_group = _split_groups(
        [str(row["rainfall_group"]) for row in rows],
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
    )
    for row in rows:
        row["split"] = split_by_group[str(row["rainfall_group"])]

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V26 dataset output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "V26_EXACT_RETURN_RECORDS.jsonl"
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["split"]),
            str(row["rainfall_group"]),
            str(row["query_set_id"]),
            str(row["context_npz_sha256"]),
            str(row["candidate_source"]),
            str(row["candidate_first_target_sha256"]),
        ),
    )
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered_rows),
        encoding="utf-8",
    )

    split_counts: dict[str, int] = {}
    split_groups: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    source_counts: dict[str, int] = {}
    for row in ordered_rows:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        split_groups[split].add(str(row["rainfall_group"]))
        source = str(row["candidate_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    if split_groups["train"] & split_groups["validation"] or split_groups["train"] & split_groups["test"] or split_groups["validation"] & split_groups["test"]:
        raise RuntimeError("V26 rainfall-group leakage across split")

    manifest = {
        "contract": V26_DATASET_CONTRACT,
        "records_jsonl": str(records_path),
        "records_sha256": _sha(records_path),
        "seed": int(args.seed),
        "split_strategy": "DETERMINISTIC_RAINFALL_GROUP_TRAIN_VALIDATION_TEST",
        "train_fraction_requested": float(args.train_fraction),
        "validation_fraction_requested": float(args.validation_fraction),
        "record_count": len(ordered_rows),
        "rainfall_group_count": len(split_by_group),
        "split_record_counts": split_counts,
        "split_group_counts": {key: len(value) for key, value in split_groups.items()},
        "candidate_source_counts": source_counts,
        "source_jsonls": [str(path) for path in source_paths],
        "source_jsonl_sha256": {str(path): _sha(path) for path in source_paths},
        "context_jsonls": [str(path) for path in context_paths],
        "context_jsonl_sha256": {str(path): _sha(path) for path in context_paths},
        "eligible_before_exact_dedup": len(eligible),
        "exact_duplicate_count": duplicate_count,
        "rejected_counts": rejected,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "h120_role": "diagnostic_only",
        "previous_version_visibility_excludes_training": False,
        "only_split_membership_controls_statistical_use": True,
        "development_only": True,
        "formal_evidence": False,
    }
    manifest_path = out_dir / "V26_EXACT_RETURN_DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
