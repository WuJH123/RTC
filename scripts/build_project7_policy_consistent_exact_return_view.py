"""Build a zero-SWMM, policy-consistent view of an existing exact-return JSONL bank.

The historical source is never edited. Missing continuation hashes are recovered only from a unique
explicit continuation hash on the exact same causal-context fingerprint. Rows that cannot be resolved
remain in the full derived audit view but are excluded from the target-continuation training view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.project7_policy_consistency import (
    POLICY_CONSISTENCY_CONTRACT,
    RESOLVED_CONTINUATION_FIELD,
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


def _rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("records JSONL must contain at least one object row")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--target-continuation-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.records).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolution = resolve_continuations(_rows(source))
    selected, selection_report = select_target_continuation(
        resolution.rows,
        target_continuation_sha256=args.target_continuation_sha256,
    )
    for row in selected:
        row["policy_consistent_decision_unit"] = policy_consistent_decision_unit(row)

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
    for row in selected:
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        provenance = str(row.get("resolved_continuation_provenance", ""))
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
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
        "resolution": resolution.report,
        "selection": selection_report,
        "selected_split_counts": dict(sorted(split_counts.items())),
        "selected_provenance_counts": dict(sorted(provenance_counts.items())),
        "pairwise_decision_unit_includes_continuation": True,
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
