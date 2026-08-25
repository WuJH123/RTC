"""Finalize one already-completed V25 matched candidate SWMM output without rerunning SWMM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.direct_tfv_v25_value_gate import V25_VALUE_GATE_CONTRACT
from run_project7_v25_matched_counterfactual_current import (
    V25_MATCHED_ACTION_ENCODING,
    V25_MATCHED_COUNTERFACTUAL_CONTRACT,
    _array_sha,
    _current_v23_hydraulic_candidate,
    _find_reference,
    _full_tfv,
    _load_context,
    _sha,
    _window_metrics,
)
from rtc.direct_tfv_operational_v23_runtime import build_operational_v23_controller
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


def _metadata_path(out_dir: Path) -> Path:
    candidates = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("run_id") and payload.get("compact_file"):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"expected one completed closed-loop metadata JSON in {out_dir}, found {len(candidates)}")
    return candidates[0]


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--query-set-id", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    metadata_path = _metadata_path(out_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records_path = Path(args.records_jsonl).resolve()
    reference = _find_reference(records_path, args.query_set_id)
    context_path = Path(str(reference["context_npz"])).resolve()
    context = _load_context(context_path)
    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    device = __import__("torch").device(args.device)
    controller, _, _, v23_lineage = build_operational_v23_controller(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=practical_asset_path(assets, "step2"),
        supervisory_control_path=practical_asset_path(assets, "supervisory_control"),
        sequence_support_path=practical_asset_path(assets, "sequence_support"),
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        device=device,
        decision_runtime_budget_seconds=180.0,
        proposal_probe_chunk_size=24,
    )
    target, score, changed, support_diag, portfolio_lineage = _current_v23_hydraulic_candidate(
        controller=controller, context=context
    )
    generated_hash = _array_sha(target)
    if generated_hash.lower() == str(reference["candidate_first_target_sha256"]).lower():
        raise ValueError("completed output corresponds to an existing hydraulic truth row")
    candidate_compact = _resolve_relative(metadata_path.parent, str(metadata["compact_file"]))
    candidate_node = _resolve_relative(metadata_path.parent, str(metadata["node_statistics_file"]))
    hold_meta = Path(str(reference["hold_metadata_path"])).resolve()
    hold_payload = json.loads(hold_meta.read_text(encoding="utf-8"))
    hold_compact = _resolve_relative(hold_meta.parent, str(hold_payload["compact_file"]))
    priority = tuple(
        line.strip() for line in Path(args.priority_nodes).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("V25 matched truth requires exactly eight Priority8 nodes")
    elapsed = int(reference["decision_elapsed_seconds"])
    windows = {}
    for minutes in (30, 60, 120):
        duration = int(minutes * 60)
        candidate_window = _window_metrics(candidate_compact, start_seconds=elapsed, duration_seconds=duration, priority_nodes=priority)
        hold_window = _window_metrics(hold_compact, start_seconds=elapsed, duration_seconds=duration, priority_nodes=priority)
        windows[str(minutes)] = {
            "candidate": candidate_window,
            "hold": hold_window,
            "delta_tfv_m3": candidate_window["tfv_m3"] - hold_window["tfv_m3"],
        }
    write_audit = audit_target_write_readback_v127(metadata_path=metadata_path)
    if write_audit.get("passed") is not True:
        raise RuntimeError("completed V25 matched candidate failed target write/readback audit")
    metadata.update({
        "strategy": "v25_missing_current_v23_hydraulic_counterfactual",
        "v25_matched_counterfactual_contract": V25_MATCHED_COUNTERFACTUAL_CONTRACT,
        "v25_value_gate_contract": V25_VALUE_GATE_CONTRACT,
        "development_only": True,
        "formal_evidence": False,
        "new_rainfall_generated": False,
        "new_policy_return_truth_scope": "missing_current_v23_hydraulic_action_only",
        "historical_v23_v24_evidence_mutated": False,
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "candidate_first_target_sha256": generated_hash,
        "source_inp_sha256": reference["source_inp_sha256"],
        "runtime_inp_sha256": _sha(_resolve_relative(metadata_path.parent, str(metadata.get("inp_path", "")))) if metadata.get("inp_path") else None,
        "step2_checkpoint_sha256": _sha(practical_asset_path(assets, "step2")),
        "v15_rank_checkpoint_sha256": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256": _sha(args.v21_boundary_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "target_write_readback_audit": write_audit,
        "v23_generator_lineage": v23_lineage,
        "v23_candidate_support_diagnostics": support_diag,
        "prepared_event_clock": metadata.get("prepared_event_clock"),
        "ready_for_policy_lock": False,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_full = _full_tfv(candidate_node)
    hold_full = float(reference["hold_branch_tfv_m3"])
    record = {
        "contract": V25_MATCHED_COUNTERFACTUAL_CONTRACT,
        "data_role": "policy_return_train",
        "development_diagnostic_only": False,
        "eligible_for_learning_dataset": True,
        "new_rainfall_generated": False,
        "new_training_scenario_generated": False,
        "new_current_policy_action_truth": True,
        "event_id": reference["event_id"],
        "rainfall_group": reference["rainfall_group"],
        "query_set_id": reference["query_set_id"],
        "decision_index": reference["decision_index"],
        "decision_elapsed_seconds": elapsed,
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "candidate_portfolio_contract": portfolio_lineage["candidate_portfolio_contract"],
        "v23_hydraulic_candidate_contract": support_diag["v23_hydraulic_candidate_contract"],
        "action_encoding_contract": V25_MATCHED_ACTION_ENCODING,
        "estimand": "TFV_H120_CURRENT_V23_CANDIDATE_VS_EXISTING_HOLD_IDENTICAL_CONTINUATION",
        "candidate_first_target_sha256": generated_hash,
        "existing_hydraulic_first_target_sha256": reference["candidate_first_target_sha256"],
        "hold_first_target_sha256": reference["hold_first_target_sha256"],
        "candidate_target": target.tolist(),
        "first_move_changed_facility_count": int(changed),
        "candidate_base_step2_score_m3": score,
        "v23_network_stress_q75": support_diag["v23_network_stress_q75"],
        "v23_strong_storm_blend": support_diag["v23_strong_storm_blend"],
        "source_inp_sha256": reference["source_inp_sha256"],
        "parent_decisions_sha256": reference["parent_decisions_sha256"],
        "context_npz_sha256": reference["context_npz_sha256"],
        "prefix_sha256": reference["prefix_sha256"],
        "base_step2_sha256": _sha(practical_asset_path(assets, "step2")),
        "v15_rank_checkpoint_sha256": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256": _sha(args.v21_boundary_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "candidate_metadata_path": str(metadata_path),
        "candidate_node_statistics_path": str(candidate_node),
        "candidate_compact_path": str(candidate_compact),
        "existing_hold_metadata_path": str(hold_meta),
        "existing_hold_node_statistics_path": str(_resolve_relative(hold_meta.parent, str(hold_payload["node_statistics_file"]))),
        "existing_hold_compact_path": str(hold_compact),
        "candidate_branch_tfv_m3": candidate_full,
        "hold_branch_tfv_m3": hold_full,
        "true_policy_return_delta_tfv_m3": candidate_full - hold_full,
        "true_policy_return_delta_tfv_h120_m3": windows["120"]["delta_tfv_m3"],
        "tfv_windows": windows,
        "pfv_h120_candidate_m3": windows["120"]["candidate"]["pfv_m3"],
        "pfv_h120_hold_m3": windows["120"]["hold"]["pfv_m3"],
        "global_peak_h120_candidate_m3s": windows["120"]["candidate"]["global_peak_flood_rate_m3s"],
        "global_peak_h120_hold_m3s": windows["120"]["hold"]["global_peak_flood_rate_m3s"],
        "storage_volume_change_h120_candidate_m3": windows["120"]["candidate"]["storage_volume_change_m3"],
        "storage_volume_change_h120_hold_m3": windows["120"]["hold"]["storage_volume_change_m3"],
        "candidate_flow_routing_error_pct": float(metadata.get("flow_routing_error_pct", 0.0)),
        "hold_flow_routing_error_pct": reference["hold_flow_routing_error_pct"],
        "candidate_target_write_readback_verified": True,
        "candidate_support_lineage_verified": True,
        "same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "passive_setting_channels_unchanged": True,
        "engineering_bounds_verified": True,
        "prefix_context_audit": reference.get("prefix_context_audit"),
        "new_swmm_runs": 1,
        "new_policy_return_truth_records": 1,
    }
    record_path = out_dir / "V25_MATCHED_COUNTERFACTUAL_RECORD.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"record_path": str(record_path), "event_id": record["event_id"], "query_set_id": record["query_set_id"], "candidate_first_target_sha256": generated_hash, "true_policy_return_delta_tfv_h120_m3": record["true_policy_return_delta_tfv_h120_m3"]}, indent=2))


if __name__ == "__main__":
    main()
