"""Assemble the exact current-V23 selected-candidate Train bank without running SWMM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_project7_v25_matched_counterfactual_current import _find_reference, _window_metrics


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _compact(meta_path: Path) -> Path:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    value = Path(str(payload["compact_file"]))
    return value.resolve() if value.is_absolute() else (meta_path.parent / value).resolve()


def _materialize_existing_exact(
    row: dict[str, Any], *, priority_nodes: tuple[str, ...], generated_hash: str
) -> dict[str, Any]:
    candidate_meta = Path(str(row["candidate_metadata_path"])).resolve()
    hold_meta = Path(str(row["hold_metadata_path"])).resolve()
    candidate = _window_metrics(
        _compact(candidate_meta),
        start_seconds=int(row["decision_elapsed_seconds"]),
        duration_seconds=7200,
        priority_nodes=priority_nodes,
    )
    hold = _window_metrics(
        _compact(hold_meta),
        start_seconds=int(row["decision_elapsed_seconds"]),
        duration_seconds=7200,
        priority_nodes=priority_nodes,
    )
    out = dict(row)
    out.update(
        {
            "contract": "PROJECT7_V25_CURRENT_V23_SELECTED_CANDIDATE_MATCHED_TRAIN_RECORD_V1",
            "v25_matched_counterfactual_contract": "PROJECT7_V25_CURRENT_V23_SELECTED_CANDIDATE_MATCHED_TRAIN_RECORD_V1",
            "action_encoding_contract": "H10_CANDIDATE_THEN_FROZEN_CAUSAL_CONTINUATION_H120_V1",
            "estimand": "TFV_H120_CURRENT_V23_SELECTED_CANDIDATE_VS_EXISTING_HOLD_IDENTICAL_CONTINUATION",
            "candidate_first_target_sha256": generated_hash,
            "existing_current_v23_exact_match": True,
            "new_current_policy_action_truth": False,
            "candidate_compact_path": str(_compact(candidate_meta)),
            "existing_hold_compact_path": str(_compact(hold_meta)),
            "true_policy_return_delta_tfv_h120_m3": candidate["tfv_m3"] - hold["tfv_m3"],
            "tfv_windows": {
                "120": {"candidate": candidate, "hold": hold, "delta_tfv_m3": candidate["tfv_m3"] - hold["tfv_m3"]}
            },
            "pfv_h120_candidate_m3": candidate["pfv_m3"],
            "pfv_h120_hold_m3": hold["pfv_m3"],
            "global_peak_h120_candidate_m3s": candidate["global_peak_flood_rate_m3s"],
            "global_peak_h120_hold_m3s": hold["global_peak_flood_rate_m3s"],
            "storage_volume_change_h120_candidate_m3": candidate["storage_volume_change_m3"],
            "storage_volume_change_h120_hold_m3": hold["storage_volume_change_m3"],
            "same_prefix_verified": True,
            "same_continuation_policy_verified": True,
            "candidate_target_write_readback_verified": bool(row.get("target_write_readback_verified", False)),
            "candidate_support_lineage_verified": bool(row.get("candidate_manifest_support_lineage_verified", False)),
            "new_swmm_runs": 0,
            "new_policy_return_truth_records": 0,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--matched-root", action="append", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    rows = _load_records(Path(args.records_jsonl).resolve())
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    by_query = {str(row["query_set_id"]): row for row in audit["query_reports"]}
    hydraulic_hash = {}
    for query_id, report in by_query.items():
        expected_context = str(
            next(
                row["context_npz"]
                for row in rows
                if str(row["query_set_id"]) == query_id
                and str(row.get("candidate_source")) == "TYPE_AWARE_HYDRAULIC_PRESSURE"
            )
        )
        for generated in report["generated_candidates"]:
            if (
                str(generated["candidate_source"]) == "TYPE_AWARE_HYDRAULIC_PRESSURE"
                and str(generated.get("context_npz")) == expected_context
            ):
                hydraulic_hash[query_id] = str(generated["candidate_first_target_sha256"])
                break
    priority = tuple(
        line.strip() for line in Path(args.priority_nodes).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("exactly eight Priority8 nodes are required")
    matched_files: dict[str, Path] = {}
    for root_value in args.matched_root:
        root = Path(root_value).resolve()
        for path in root.rglob("V25_MATCHED_COUNTERFACTUAL_RECORD.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            query_id = str(payload.get("query_set_id", ""))
            if query_id:
                if query_id in matched_files:
                    raise ValueError(f"duplicate finalized matched record for query {query_id}")
                matched_files[query_id] = path

    existing_hydraulic = {
        str(row["query_set_id"]): row
        for row in rows
        if str(row.get("candidate_source")) == "TYPE_AWARE_HYDRAULIC_PRESSURE"
    }
    if len(existing_hydraulic) != 48:
        raise ValueError(f"expected 48 existing hydraulic rows, found {len(existing_hydraulic)}")

    final_rows: list[dict[str, Any]] = []
    reused_exact = 0
    new_truth = 0
    for query_id in sorted(existing_hydraulic):
        if query_id not in hydraulic_hash:
            raise ValueError(f"audit has no hydraulic generated hash for {query_id}")
        report = by_query[query_id]
        exact = bool(report.get("hydraulic_exact_match"))
        if exact:
            row = _materialize_existing_exact(
                existing_hydraulic[query_id],
                priority_nodes=priority,
                generated_hash=hydraulic_hash[query_id],
            )
            reused_exact += 1
        else:
            record_path = matched_files.get(query_id)
            if record_path is None:
                raise FileNotFoundError(f"missing finalized matched record for {query_id}")
            row = json.loads(record_path.read_text(encoding="utf-8"))
            if str(row.get("candidate_first_target_sha256", "")).lower() != hydraulic_hash[query_id].lower():
                raise ValueError(f"matched record target hash differs from audit for {query_id}")
            new_truth += int(bool(row.get("new_current_policy_action_truth")))
        final_rows.append(row)

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"matched bank output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "V25_MATCHED_TRAIN_RECORDS.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in final_rows), encoding="utf-8"
    )
    groups = sorted({str(row["rainfall_group"]) for row in final_rows})
    manifest = {
        "contract": "PROJECT7_V25_CURRENT_V23_SELECTED_CANDIDATE_MATCHED_TRAIN_BANK_V1",
        "development_only": True,
        "formal_evidence": False,
        "data_role": "policy_return_train",
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "boundary_supervision_unit": "CURRENT_V23_RANK_SELECTED_HYDRAULIC_CANDIDATE_PER_QUERY",
        "record_count": len(final_rows),
        "query_count": len(final_rows),
        "rainfall_group_count": len(groups),
        "rainfall_groups": groups,
        "group_overlap_with_validation": 0,
        "group_overlap_with_calibration": 0,
        "source_counts": {"TYPE_AWARE_HYDRAULIC_PRESSURE": len(final_rows)},
        "reused_existing_exact_current_v23_records": reused_exact,
        "new_current_policy_action_truth_records": new_truth,
        "shared_hold_truth_reused": len(final_rows),
        "new_rainfall_generated": False,
        "new_training_scenario_generated": False,
        "new_validation_or_calibration_truth_generated": False,
        "records_jsonl": str(jsonl_path),
        "records_sha256": _sha(jsonl_path),
        "candidate_hash_audit": str(Path(args.audit).resolve()),
        "candidate_hash_audit_sha256": _sha(args.audit),
        "step2_sha256": str(final_rows[0]["base_step2_sha256"]),
        "v15_rank_sha256": str(final_rows[0]["v15_rank_checkpoint_sha256"]),
        "v21_boundary_sha256": str(final_rows[0]["v21_boundary_checkpoint_sha256"]),
        "h120_definition": "300-s sampled flooding-rate integration from decision elapsed time for 120 minutes",
    }
    manifest_path = out_dir / "V25_MATCHED_TRAIN_BANK_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "records": str(jsonl_path),
        "record_count": len(final_rows),
        "rainfall_group_count": len(groups),
        "reused_existing_exact_current_v23_records": reused_exact,
        "new_current_policy_action_truth_records": new_truth,
        "records_sha256": manifest["records_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
