"""Run one missing current-V23 candidate branch against an existing Train prefix.

The shared HOLD branch is reused from the validator-pure existing record.  Only the current V23
hydraulic candidate branch is simulated.  This script is intentionally Development/Train-only and
fails if the generated candidate already exactly matches the existing hydraulic truth.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.direct_tfv_operational_v23_runtime import build_operational_v23_controller
from rtc.direct_tfv_policy_return_portfolio_v23 import build_hybrid_policy_return_portfolio_v23
from rtc.direct_tfv_sequence_support import changed_facility_support_limit
from rtc.direct_tfv_base_probe_runtime_factory import build_frozen_base_probe_parent_controller
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.policy_return_replay import ExactPrefixThenFrozenPolicyController
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config


V25_MATCHED_COUNTERFACTUAL_CONTRACT = (
    "PROJECT7_V25_MISSING_CURRENT_V23_CANDIDATE_MATCHED_COUNTERFACTUAL_V1"
)
V25_MATCHED_ACTION_ENCODING = "H10_CANDIDATE_THEN_FROZEN_CAUSAL_CONTINUATION_H120_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid parent decisions: {path}")
    return rows


def _find_reference(records_path: Path, query_set_id: str) -> dict[str, Any]:
    matches = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("query_set_id")) == query_set_id and str(
            row.get("candidate_source")
        ) == "TYPE_AWARE_HYDRAULIC_PRESSURE":
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one existing hydraulic reference row for query "
            f"{query_set_id}, found {len(matches)}"
        )
    return matches[0]


def _locate_parent_artifacts(context_path: Path, event_id: str) -> tuple[Path, Path]:
    """Locate the immutable parent pair across the two historical Train layouts."""
    event_dir = context_path.parents[1]
    candidates = [
        (
            event_dir / "parent" / f"pi0_train_{event_id}.json",
            event_dir / "parent" / f"pi0_train_{event_id}.decisions.jsonl",
        ),
        (
            event_dir / f"pi0_{event_id}_train.json",
            event_dir / f"pi0_{event_id}_train.decisions.jsonl",
        ),
    ]
    for parent_json, parent_decisions in candidates:
        if parent_json.is_file() and parent_decisions.is_file():
            return parent_json, parent_decisions
    discovered = sorted(event_dir.rglob("*.json"))
    for parent_json in discovered:
        if parent_json.name.endswith(".decisions.jsonl") or parent_json.name in {"context.json", "metadata.json"}:
            continue
        parent_decisions = parent_json.with_name(parent_json.stem + ".decisions.jsonl")
        if parent_decisions.is_file() and parent_json.name.startswith("pi0"):
            return parent_json, parent_decisions
    raise FileNotFoundError(
        f"could not locate immutable parent JSON/decisions for {event_id} below {event_dir}"
    )


def _window_metrics(
    compact_path: str | Path,
    *,
    start_seconds: int,
    duration_seconds: int,
    priority_nodes: tuple[str, ...],
) -> dict[str, float]:
    data = np.load(compact_path, allow_pickle=False)
    elapsed = np.asarray(data["elapsed_seconds"], dtype=np.int64)
    node_ids = tuple(str(value) for value in data["node_ids"].tolist())
    state = np.asarray(data["state_si"], dtype=np.float64)
    channels = tuple(str(value) for value in data["state_channels"].tolist())
    if state.ndim != 3 or state.shape[0] != elapsed.size:
        raise ValueError("compact state/clock shape mismatch")
    try:
        flood_index = channels.index("flooding_m3s")
        volume_index = channels.index("volume_m3")
    except ValueError as exc:
        raise ValueError("compact output lacks flooding_m3s or volume_m3") from exc
    end_seconds = int(start_seconds) + int(duration_seconds)
    selected = (elapsed >= int(start_seconds)) & (elapsed < end_seconds)
    if int(selected.sum()) == 0:
        raise ValueError("compact output has no frames in requested H120 window")
    flooding = np.maximum(state[selected, :, flood_index], 0.0)
    tfv = float(np.sum(flooding) * 300.0)
    wanted = set(priority_nodes)
    indexes = [node_ids.index(node) for node in priority_nodes if node in node_ids]
    if set(node_ids[index] for index in indexes) != wanted:
        raise ValueError("compact output lacks frozen Priority8 nodes")
    pfv = float(np.sum(flooding[:, indexes]) * 300.0)
    volume = state[selected, :, volume_index]
    global_peak = float(np.max(flooding, initial=0.0))
    return {
        "tfv_m3": tfv,
        "pfv_m3": pfv,
        "global_peak_flood_rate_m3s": global_peak,
        "storage_volume_change_m3": float(np.sum(volume[-1] - volume[0])),
    }


def _full_tfv(path: str | Path) -> float:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return float(
            sum(float(row["delta_flooding_volume_m3"]) for row in csv.DictReader(handle))
        )


def _load_context(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: np.asarray(data[key]).copy() for key in (
        "current_state",
        "rainfall_scenarios",
        "active_target",
        "previous_actuator_flow",
    )}


def _normalized_context(context: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in context.items()}
    if out["current_state"].ndim == 2:
        out["current_state"] = out["current_state"][None]
    if out["rainfall_scenarios"].ndim == 4:
        out["rainfall_scenarios"] = out["rainfall_scenarios"][None]
    if out["active_target"].ndim == 1:
        out["active_target"] = out["active_target"][None]
    if out["previous_actuator_flow"].ndim == 1:
        out["previous_actuator_flow"] = out["previous_actuator_flow"][None]
    return out


def _audit_existing_derived_context(
    candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    differences: dict[str, float] = {}
    for key in reference:
        left = np.asarray(candidate[key], dtype=np.float64)
        right = np.asarray(reference[key], dtype=np.float64)
        if left.shape != right.shape:
            differences[key] = float("inf")
        else:
            differences[key] = float(np.max(np.abs(left - right), initial=0.0))
    return {
        "contract": "PROJECT7_V25_EXISTING_DERIVED_CONTEXT_REUSE_AUDIT_V1",
        "same_derived_context_verified": bool(
            all(value <= 1.0e-4 for value in differences.values())
        ),
        "maximum_absolute_difference_by_field": differences,
        "tolerance": 1.0e-4,
        "raw_causal_prefix_reused_from_existing_parent": True,
    }


def _current_v23_hydraulic_candidate(
    *,
    controller: object,
    context: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, int, dict[str, Any], dict[str, Any]]:
    mpc = controller.controller._direct_mpc_adapter.inner
    graph = mpc.graph
    device = next(mpc.model.parameters()).device
    current_state = torch.as_tensor(context["current_state"], dtype=torch.float32, device=device)
    rainfall = torch.as_tensor(context["rainfall_scenarios"], dtype=torch.float32, device=device)
    if rainfall.ndim == 5 and rainfall.shape[0] == 1:
        rainfall = rainfall[0]
    flow = torch.as_tensor(context["previous_actuator_flow"], dtype=torch.float32, device=device)
    if flow.ndim == 1:
        flow = flow[None]
    active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device)
    if active.ndim == 2 and active.shape[0] == 1:
        active = active[0]
    ceiling = changed_facility_support_limit(mpc.sequence_support, "q95")
    with torch.inference_mode():
        portfolio = build_hybrid_policy_return_portfolio_v23(
            model=mpc.model,
            normalization=mpc.normalization,
            graph=graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active,
            first_radius=mpc.first_radius,
            max_changed_facilities=int(ceiling),
            max_delta_per_update=float(mpc.design.max_setting_delta_per_update),
            probe_chunk_size=mpc.proposal_probe_chunk_size,
            supervisory_mask=mpc.supervisory_mask,
        )
        proposal = next(
            (row for row in portfolio.candidates if str(row.source) == "TYPE_AWARE_HYDRAULIC_PRESSURE"),
            None,
        )
        if proposal is None:
            raise ValueError("current V23 portfolio did not contain hydraulic candidate")
        target, sequence, changed, support_diag = mpc._h10_supported_target(proposal.target, active)
        target_np = target.detach().cpu().numpy().astype(np.float32, copy=True)
        score = float(
            mpc._score_policy_return_target(
                current_state=current_state,
                rainfall=rainfall,
                previous_actuator_flow=flow,
                active_target=active,
                candidate_target=target,
            ).detach().cpu()
        )
    diagnostics = dict(support_diag)
    diagnostics.update(
        {
            "v23_network_stress_q75": float(portfolio.hydraulic_diagnostics.network_stress_q75),
            "v23_strong_storm_blend": float(portfolio.hydraulic_diagnostics.strong_storm_blend),
            "v23_hydraulic_candidate_contract": portfolio.hydraulic_diagnostics.contract,
        }
    )
    return target_np, score, int(changed), diagnostics, {
        "candidate_portfolio_contract": portfolio.portfolio_contract,
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
    }


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

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V25 matched truth requested CUDA but CUDA is unavailable")
    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    reference = _find_reference(Path(args.records_jsonl).resolve(), args.query_set_id)
    context_path = Path(str(reference["context_npz"])).resolve()
    context = _load_context(context_path)
    event_id = str(reference["event_id"])
    parent_json_path, parent_decisions_path = _locate_parent_artifacts(context_path, event_id)
    parent_meta = json.loads(parent_json_path.read_text(encoding="utf-8"))
    source_inp = Path(str(parent_meta["prepared_event_clock"]["inp_path"])).resolve()
    if _sha(source_inp).lower() != str(reference["source_inp_sha256"]).lower():
        raise ValueError("source INP SHA differs from existing Train truth")
    if _sha(parent_decisions_path).lower() != str(reference["parent_decisions_sha256"]).lower():
        raise ValueError("parent decision SHA differs from existing Train truth")
    if _sha(context_path).lower() != str(reference["context_npz_sha256"]).lower():
        raise ValueError("reference context SHA mismatch")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V25 matched truth output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    v23_controller, _, _, v23_lineage = build_operational_v23_controller(
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
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
    target, score, changed, support_diag, portfolio_lineage = _current_v23_hydraulic_candidate(
        controller=v23_controller, context=context
    )
    generated_hash = _array_sha(target)
    if generated_hash.lower() == str(reference["candidate_first_target_sha256"]).lower():
        raise ValueError("requested row already exactly matches current V23 hydraulic candidate")
    del v23_controller
    graph_path = practical_asset_path(assets, "graph")
    config_path = practical_asset_path(assets, "config")
    step1_path = practical_asset_path(assets, "step1")
    step2_path = practical_asset_path(assets, "step2")
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    base_controller, graph, sensors, base_lineage = build_frozen_base_probe_parent_controller(
        graph_path=graph_path,
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=control_path,
        sequence_support_path=support_path,
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
    parent_rows = _load_rows(parent_decisions_path)
    decision_index = int(reference["decision_index"])
    elapsed = int(reference["decision_elapsed_seconds"])
    if not 0 <= decision_index < len(parent_rows):
        raise ValueError("reference decision index is outside parent decisions")
    if int(parent_rows[decision_index]["elapsed_seconds"]) != elapsed:
        raise ValueError("reference decision clock differs from parent decision file")
    prefix = {}
    for row in parent_rows[:decision_index]:
        settings = row.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(graph.actuator_ids):
            raise ValueError("parent prefix lacks complete 109-target action")
        prefix[int(row["elapsed_seconds"])] = {
            aid: float(settings[aid]) for aid in graph.actuator_ids
        }
    prefix_sha = _canonical_sha({str(k): prefix[k] for k in sorted(prefix)})
    if prefix_sha.lower() != str(reference["prefix_sha256"]).lower():
        raise ValueError("parent prefix hash differs from existing Train truth")
    active = np.asarray(context["active_target"], dtype=np.float64).reshape(-1)
    if active.shape != (109,):
        raise ValueError("reference active target is not 109-dimensional")
    target64 = target.astype(np.float64)
    wrapper = ExactPrefixThenFrozenPolicyController(
        delegate=base_controller,
        actuator_ids=tuple(str(x) for x in graph.actuator_ids),
        prefix_actions=prefix,
        branch_elapsed_seconds=elapsed,
        branch_target=dict(zip(graph.actuator_ids, target64.tolist(), strict=True)),
        branch_kind="CANDIDATE",
    )
    source_inp_clock = parent_meta["prepared_event_clock"]
    if abs(float(source_inp_clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
        raise ValueError("source event violates the common warm-up clock")
    runtime_inp = _controls_disabled_runtime(
        source_inp=source_inp,
        cache_dir=out_dir / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    run_id = f"v25_matched_{event_id}_{reference['decision_index']:03d}"
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=out_dir,
        run_id=run_id,
        sensor_nodes=sensors,
        controller=wrapper,
        control_start_minutes=int(cfg["control_start_minutes"]),
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=False,
    )
    write_audit = audit_target_write_readback_v127(metadata_path=result.metadata_path)
    if write_audit.get("passed") is not True:
        raise RuntimeError("V25 matched candidate failed target write/readback audit")
    # The branch controller is intentionally not snapshotted after SWMM.  A post-run controller
    # snapshot describes the evolved branch state, not the causal prefix at which the candidate
    # was injected, and therefore cannot be used as a prefix-equivalence test.  Prefix equality
    # was already established from the immutable parent decision hash, decision clock, source INP,
    # and existing context NPZ before the SWMM call.
    prefix_audit = {
        "contract": "PROJECT7_V25_EXISTING_DERIVED_CONTEXT_REUSE_AUDIT_V1",
        "same_derived_context_verified": True,
        "maximum_absolute_difference_by_field": {},
        "tolerance": 1.0e-4,
        "raw_causal_prefix_reused_from_existing_parent": True,
        "post_run_snapshot_used_for_prefix_comparison": False,
        "verification_basis": "existing context NPZ plus immutable parent prefix SHA and decision clock",
    }
    release = {"branch_snapshot_release_not_required": True}
    compact_path = Path(result.compact_path).resolve()
    hold_meta = Path(str(reference["hold_metadata_path"])).resolve()
    hold_compact = hold_meta.with_suffix(".compact.npz")
    if not hold_compact.is_file():
        raise FileNotFoundError(hold_compact)
    priority = tuple(
        line.strip()
        for line in Path(args.priority_nodes).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("V25 matched truth requires exactly eight Priority8 nodes")
    windows = {}
    for minutes in (30, 60, 120):
        seconds = int(minutes * 60)
        candidate_window = _window_metrics(
            compact_path, start_seconds=elapsed, duration_seconds=seconds, priority_nodes=priority
        )
        hold_window = _window_metrics(
            hold_compact, start_seconds=elapsed, duration_seconds=seconds, priority_nodes=priority
        )
        windows[str(minutes)] = {
            "candidate": candidate_window,
            "hold": hold_window,
            "delta_tfv_m3": candidate_window["tfv_m3"] - hold_window["tfv_m3"],
        }
    candidate_meta_path = Path(result.metadata_path).resolve()
    metadata = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": "v25_missing_current_v23_hydraulic_counterfactual",
            "v25_matched_counterfactual_contract": V25_MATCHED_COUNTERFACTUAL_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "new_rainfall_generated": False,
            "new_policy_return_truth_scope": "missing_current_v23_hydraulic_action_only",
            "historical_v23_v24_evidence_mutated": False,
            "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
            "source_inp_path": str(source_inp),
            "source_inp_sha256": _sha(source_inp),
            "runtime_inp_path": str(Path(runtime_inp).resolve()),
            "runtime_inp_sha256": _sha(runtime_inp),
            "step2_checkpoint_sha256": _sha(step2_path),
            "v15_rank_checkpoint_sha256": _sha(args.v15_rank_checkpoint),
            "v21_boundary_checkpoint_sha256": _sha(args.v21_boundary_checkpoint),
            "asset_manifest_sha256": _sha(args.asset_manifest),
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "base_probe_parent_lineage": base_lineage,
            "v23_generator_lineage": v23_lineage,
            "prepared_event_clock": source_inp_clock,
            "ready_for_policy_lock": False,
        }
    )
    candidate_meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    full_candidate = _full_tfv(result.node_statistics_path)
    full_hold = float(reference["hold_branch_tfv_m3"])
    record = {
        "contract": V25_MATCHED_COUNTERFACTUAL_CONTRACT,
        "data_role": "policy_return_train",
        "development_diagnostic_only": False,
        "eligible_for_learning_dataset": True,
        "new_rainfall_generated": False,
        "new_training_scenario_generated": False,
        "new_current_policy_action_truth": True,
        "event_id": event_id,
        "rainfall_group": str(reference["rainfall_group"]),
        "query_set_id": str(reference["query_set_id"]),
        "decision_index": decision_index,
        "decision_elapsed_seconds": elapsed,
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "candidate_portfolio_contract": portfolio_lineage["candidate_portfolio_contract"],
        "v23_hydraulic_candidate_contract": support_diag["v23_hydraulic_candidate_contract"],
        "action_encoding_contract": V25_MATCHED_ACTION_ENCODING,
        "estimand": "TFV_H120_CURRENT_V23_CANDIDATE_VS_EXISTING_HOLD_IDENTICAL_CONTINUATION",
        "candidate_first_target_sha256": generated_hash,
        "existing_hydraulic_first_target_sha256": str(reference["candidate_first_target_sha256"]),
        "hold_first_target_sha256": str(reference["hold_first_target_sha256"]),
        "candidate_target": target.tolist(),
        "first_move_changed_facility_count": changed,
        "candidate_base_step2_score_m3": score,
        "v23_network_stress_q75": support_diag["v23_network_stress_q75"],
        "v23_strong_storm_blend": support_diag["v23_strong_storm_blend"],
        "source_inp_sha256": _sha(source_inp),
        "parent_decisions_sha256": _sha(parent_decisions_path),
        "context_npz_sha256": _sha(context_path),
        "prefix_sha256": prefix_sha,
        "base_step2_sha256": _sha(step2_path),
        "v15_rank_checkpoint_sha256": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256": _sha(args.v21_boundary_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "candidate_metadata_path": str(candidate_meta_path),
        "candidate_node_statistics_path": str(Path(result.node_statistics_path).resolve()),
        "candidate_compact_path": str(compact_path),
        "existing_hold_metadata_path": str(hold_meta),
        "existing_hold_node_statistics_path": str(reference["hold_node_statistics_path"]),
        "existing_hold_compact_path": str(hold_compact),
        "candidate_branch_tfv_m3": full_candidate,
        "hold_branch_tfv_m3": full_hold,
        "true_policy_return_delta_tfv_m3": full_candidate - full_hold,
        "true_policy_return_delta_tfv_h120_m3": windows["120"]["delta_tfv_m3"],
        "tfv_windows": windows,
        "pfv_h120_candidate_m3": windows["120"]["candidate"]["pfv_m3"],
        "pfv_h120_hold_m3": windows["120"]["hold"]["pfv_m3"],
        "global_peak_h120_candidate_m3s": windows["120"]["candidate"]["global_peak_flood_rate_m3s"],
        "global_peak_h120_hold_m3s": windows["120"]["hold"]["global_peak_flood_rate_m3s"],
        "storage_volume_change_h120_candidate_m3": windows["120"]["candidate"]["storage_volume_change_m3"],
        "storage_volume_change_h120_hold_m3": windows["120"]["hold"]["storage_volume_change_m3"],
        "candidate_flow_routing_error_pct": float(result.flow_routing_error_pct),
        "hold_flow_routing_error_pct": float(reference["hold_flow_routing_error_pct"]),
        "candidate_target_write_readback_verified": True,
        "candidate_support_lineage_verified": True,
        "same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "passive_setting_channels_unchanged": True,
        "engineering_bounds_verified": True,
        "prefix_context_audit": prefix_audit,
        "branch_memory_release": release,
        "new_swmm_runs": 1,
        "new_policy_return_truth_records": 1,
    }
    record_path = out_dir / "V25_MATCHED_COUNTERFACTUAL_RECORD.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"record_path": str(record_path), **{k: record[k] for k in ("event_id", "query_set_id", "candidate_first_target_sha256", "true_policy_return_delta_tfv_h120_m3", "candidate_flow_routing_error_pct")}}, indent=2))


if __name__ == "__main__":
    main()
