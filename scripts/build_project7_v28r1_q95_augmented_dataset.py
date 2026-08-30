"""Build the V28 augmented dataset while preserving targeted raw-proposal provenance.

The V28 builder correctly appends q95-supported truth but does not copy ``raw_candidate_target``
from the targeted truth record. V28R1 preserves that expensive provenance for future diagnostics.
The supported target remains the statistical action identity and the immutable V27 split is still
inherited. This script does not make raw proposals executable or automatically usable as residual
features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_project7_v28_q95_augmented_dataset import augment_dataset, read_jsonl, sha256_file


CONTRACT = "PROJECT7_STEP3_V28R1_Q95_AUGMENTED_DATASET_RAW_PROVENANCE_V1"


def _target_key(row: dict[str, Any]) -> tuple[str, str]:
    context = str(row.get("causal_context_fingerprint_sha256", "")).strip().lower()
    action = str(
        row.get("q95_supported_target_sha256", row.get("candidate_first_target_sha256", ""))
    ).strip().lower()
    return context, action


def _raw_sha(target: Any) -> str:
    payload = json.dumps([float(value) for value in target], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_v28r1_dataset(
    *,
    base_manifest: str | Path,
    base_records: str | Path,
    targeted_records: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    manifest = augment_dataset(
        base_manifest_path=base_manifest,
        base_records_path=base_records,
        targeted_records_path=targeted_records,
        out_dir=out_dir,
    )
    out = Path(out_dir).resolve()
    records_path = Path(str(manifest["records"])).resolve()
    rows = read_jsonl(records_path)
    targeted = read_jsonl(targeted_records)
    raw_by_key: dict[tuple[str, str], list[float]] = {}
    for row in targeted:
        raw = row.get("raw_candidate_target")
        if not isinstance(raw, list) or not raw:
            continue
        key = _target_key(row)
        if not key[0] or not key[1]:
            continue
        values = [float(value) for value in raw]
        previous = raw_by_key.get(key)
        if previous is not None and previous != values:
            raise ValueError(f"conflicting raw proposal provenance for targeted key {key}")
        raw_by_key[key] = values

    preserved = 0
    for row in rows:
        raw = raw_by_key.get(_target_key(row))
        if raw is None:
            continue
        row["raw_candidate_target"] = raw
        row["raw_candidate_target_sha256_json"] = _raw_sha(raw)
        row["v28r1_raw_proposal_provenance_preserved"] = True
        preserved += 1

    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = out / "V28_Q95_MATCHED_AUGMENTED_EXACT_RETURN_DATASET_MANIFEST.json"
    manifest.update(
        {
            "contract": CONTRACT,
            "records_sha256": sha256_file(records_path),
            "v28r1_targeted_raw_proposal_count": int(len(raw_by_key)),
            "v28r1_targeted_raw_proposal_preserved_count": int(preserved),
            "v28r1_raw_proposal_used_for_model_fit": False,
            "v28r1_raw_proposal_role": "provenance_and_future_diagnostic_only",
        }
    )
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
            build_v28r1_dataset(
                base_manifest=args.base_manifest,
                base_records=args.base_records,
                targeted_records=args.targeted_records,
                out_dir=args.out_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
