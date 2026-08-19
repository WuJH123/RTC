"""Run one same-prefix Practical policy-return query with a single shared HOLD SWMM branch.

The default first-round continuation is the current masked hybrid Base-H10 parent pi0. HOLD truth is
computed once and reused across all candidates (1+N authoritative branches). Candidate tensors remain
109-channel for the pretrained Step2, but the frozen native supervisory mask requires all passive
channels to equal HOLD. Post-support base-Step2 scores from the portfolio artifact are preserved in
the authoritative records for later sign/rank diagnostics.
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
from rtc.direct_tfv_base_probe_runtime_factory import build_frozen_base_probe_parent_controller
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    sha256_file,
    validate_policy_return_record,
)
from rtc.direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
from rtc.direct_tfv_policy_return_runtime_factory import build_frozen_policy_return_continuation_controller
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.policy_return_replay import (
    ExactPrefixThenFrozenPolicyController,
    audit_policy_return_prefix_contexts,
    snapshot_and_release_policy_return_branch,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime


PRACTICAL_POLICY_RETURN_QUERY_RUNNER_CONTRACT = (
    "PROJECT7_PRACTICAL_POLICY_RETURN_SHARED_HOLD_QUERY_RUNNER_V3_82CONTROL_109REP"
)


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes()).hexdigest()


def _tfv(path: str | Path) -> float:
    total = 0.0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += float(row["delta_flooding_volume_m3"])
    return float(total)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("parent decisions JSONL is empty/invalid")
    return rows


def _load_portfolio(
    path: Path,
    ids: tuple[str, ...],
    *,
    supervisory_mask: np.ndarray,
    supervisory_mask_sha256: str,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("candidate manifest has the wrong current masked hybrid portfolio contract")
    if str(payload.get("supervisory_mask_sha256", "")).lower() != str(supervisory_mask_sha256).lower():
        raise ValueError("candidate manifest was generated under another supervisory-control mask")
    if int(payload.get("supervisory_control_dimension", -1)) != int(supervisory_mask.sum()):
        raise ValueError("candidate manifest has the wrong supervisory-control dimension")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError("current Practical candidate manifest must contain 1-4 candidates")
    result: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    passive = ~np.asarray(supervisory_mask, dtype=bool)
    for row in rows:
        if not isinstance(row, dict) or row.get("contract") != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
            raise ValueError("candidate manifest contains an invalid row")
        row_ids = tuple(str(x) for x in row.get("actuator_ids", ()))
        if row_ids and row_ids != ids:
            raise ValueError("candidate actuator order differs from parent decision")
        source = str(row.get("candidate_source", "")).strip()
        target = np.asarray(row.get("target_settings", ()), dtype=float).reshape(-1)
        score = float(row.get("base_step2_h10_score_m3", float("nan")))
        if not source or target.shape != (109,) or not np.isfinite(target).all() or not np.isfinite(score):
            raise ValueError("candidate row lacks source/109 finite target/base-Step2 score")
        if np.any(target < -1.0e-7) or np.any(target > 1.0 + 1.0e-7):
            raise ValueError("candidate target leaves [0,1]")
        key = np.ascontiguousarray(target, dtype=np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        result.append({"source": source, "target": target, "base_step2_h10_score_m3": score, "passive": passive})
    if not result:
        raise ValueError("candidate portfolio deduplicated to zero actions")
    return result


def _build_delegate(args: argparse.Namespace, assets: dict, device: torch.device):
    common = dict(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=practical_asset_path(assets, "step2"),
        supervisory_control_path=practical_asset_path(assets, "supervisory_control"),
        sequence_support_path=practical_asset_path(assets, "sequence_support"),
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
        projected_gradient_steps=int(args.projected_gradient_steps),
        projected_gradient_step_fraction=float(args.projected_gradient_step_fraction),
    )
    if args.continuation_kind == "base-probe":
        if args.policy_return_checkpoint or args.policy_return_admission:
            raise ValueError("base-probe pi0 must not receive policy-return critic/admission")
        return build_frozen_base_probe_parent_controller(**common)
    if not args.policy_return_checkpoint or not args.policy_return_admission:
        raise ValueError("policy-return pi1 requires critic and matched admission")
    return build_frozen_policy_return_continuation_controller(
        **common,
        policy_return_checkpoint_path=args.policy_return_checkpoint,
        policy_return_admission_path=args.policy_return_admission,
    )


def _run_branch(
    *,
    args: argparse.Namespace,
    assets: dict,
    branch_kind: str,
    branch_target: np.ndarray,
    ids: tuple[str, ...],
    prefix_actions: dict[int, dict[str, float]],
    branch_elapsed: int,
    device: torch.device,
    out_root: Path,
    suffix: str,
):
    delegate, graph, sensors, lineage = _build_delegate(args, assets, device)
    if tuple(str(x) for x in graph.actuator_ids) != ids:
        raise ValueError("parent decision actuator order differs from frozen graph")
    wrapper = ExactPrefixThenFrozenPolicyController(
        delegate=delegate,
        actuator_ids=ids,
        prefix_actions=prefix_actions,
        branch_elapsed_seconds=branch_elapsed,
        branch_target=dict(zip(ids, branch_target.tolist(), strict=True)),
        branch_kind=branch_kind,
    )
    cfg = json.loads(Path(practical_asset_path(assets, "config")).read_text(encoding="utf-8"))
    out = out_root / suffix
    runtime_inp = _controls_disabled_runtime(
        source_inp=Path(args.inp).resolve(),
        cache_dir=out / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=out,
        run_id=f"{args.run_id}_{suffix}",
        sensor_nodes=sensors,
        controller=wrapper,
        control_start_minutes=int(cfg["control_start_minutes"]),
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=False,
    )
    return result, wrapper, lineage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-manifest", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--parent-decisions", required=True)
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--decision-index", type=int, required=True)
    p.add_argument("--rainfall-group", required=True)
    p.add_argument("--event-id", required=True)
    p.add_argument(
        "--data-role",
        choices=("policy_return_train", "policy_return_validation", "policy_return_calibration"),
        required=True,
    )
    p.add_argument("--continuation-kind", choices=("base-probe", "policy-return"), default="base-probe")
    p.add_argument("--policy-return-checkpoint")
    p.add_argument("--policy-return-admission")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--probe-chunk-size", type=int, default=24)
    p.add_argument("--projected-gradient-steps", type=int, default=6)
    p.add_argument("--projected-gradient-step-fraction", type=float, default=0.25)
    args = p.parse_args()

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    inp = Path(args.inp).resolve()
    decisions_path = Path(args.parent_decisions).resolve()
    portfolio_path = Path(args.candidate_manifest).resolve()
    if not inp.is_file() or not decisions_path.is_file() or not portfolio_path.is_file():
        raise FileNotFoundError("query runner requires existing INP, decisions and candidate manifest")
    rows = _load_rows(decisions_path)
    if not 0 <= int(args.decision_index) < len(rows):
        raise ValueError("decision-index lies outside parent decisions")
    selected = rows[int(args.decision_index)]
    diagnostics = selected.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("selected parent decision lacks diagnostics")
    ids = tuple(str(x) for x in diagnostics.get("counterfactual_actuator_ids", ()))
    hold_target = np.asarray(diagnostics.get("hold_reference_settings", ()), dtype=float).reshape(-1)
    if len(ids) != 109 or len(set(ids)) != 109 or hold_target.shape != (109,):
        raise ValueError("selected parent decision lacks 109-channel HOLD context")
    control, mask = load_native_supervisory_control(
        practical_asset_path(assets, "supervisory_control"),
        actuator_ids=ids,
    )
    candidates = _load_portfolio(
        portfolio_path,
        ids,
        supervisory_mask=mask,
        supervisory_mask_sha256=str(control["supervisory_mask_sha256"]),
    )
    for row in candidates:
        if np.any(np.abs(row["target"][~mask] - hold_target[~mask]) > 1.0e-7):
            raise ValueError(f"candidate {row['source']} changes a passive setting channel relative to HOLD")
    branch_elapsed = int(selected["elapsed_seconds"])
    prefix_actions: dict[int, dict[str, float]] = {}
    for row in rows[: int(args.decision_index)]:
        settings = row.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(ids):
            raise ValueError("parent prefix decision lacks complete 109-target action")
        prefix_actions[int(row["elapsed_seconds"])] = {aid: float(settings[aid]) for aid in ids}
    prefix_sha = _canonical_sha({str(k): prefix_actions[k] for k in sorted(prefix_actions)})

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    hold_result, hold_wrapper, hold_lineage = _run_branch(
        args=args,
        assets=assets,
        branch_kind="HOLD",
        branch_target=hold_target,
        ids=ids,
        prefix_actions=prefix_actions,
        branch_elapsed=branch_elapsed,
        device=device,
        out_root=out_root,
        suffix="hold_shared",
    )
    hold_context, hold_release = snapshot_and_release_policy_return_branch(hold_wrapper, device=device)
    del hold_wrapper
    hold_tfv = _tfv(hold_result.node_statistics_path)
    continuation_policy_sha = _canonical_sha(
        {"continuation_kind": args.continuation_kind, "lineage": hold_lineage}
    )
    query_set_id = _canonical_sha(
        {
            "event_id": args.event_id,
            "rainfall_group": args.rainfall_group,
            "decision_index": int(args.decision_index),
            "decision_elapsed_seconds": branch_elapsed,
            "prefix_sha256": prefix_sha,
            "hold_first_target_sha256": _array_sha(hold_target),
            "continuation_policy_sha256": continuation_policy_sha,
            "supervisory_mask_sha256": control["supervisory_mask_sha256"],
        }
    )

    records: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {"hold_release": hold_release, "candidate_releases": {}}
    for index, candidate in enumerate(candidates):
        source = str(candidate["source"])
        target = np.asarray(candidate["target"], dtype=np.float64)
        base_score = float(candidate["base_step2_h10_score_m3"])
        changed = int(np.sum(np.abs(target - hold_target) > 1.0e-7))
        if changed <= 0:
            continue
        suffix = f"candidate_{index:02d}"
        candidate_result, wrapper, candidate_lineage = _run_branch(
            args=args,
            assets=assets,
            branch_kind="CANDIDATE",
            branch_target=target,
            ids=ids,
            prefix_actions=prefix_actions,
            branch_elapsed=branch_elapsed,
            device=device,
            out_root=out_root,
            suffix=suffix,
        )
        candidate_context, release = snapshot_and_release_policy_return_branch(wrapper, device=device)
        del wrapper
        lifecycle["candidate_releases"][source] = release
        if candidate_lineage != hold_lineage:
            raise RuntimeError(f"candidate {source} used a different continuation policy")
        audit = audit_policy_return_prefix_contexts(candidate_context, hold_context)
        if audit["same_authoritative_prefix_verified"] is not True:
            raise RuntimeError(f"candidate {source} raw causal prefix differs from shared HOLD")
        candidate_tfv = _tfv(candidate_result.node_statistics_path)
        truth = float(candidate_tfv - hold_tfv)
        context_path = out_root / f"{args.run_id}.{index:02d}.policy_return_context.npz"
        np.savez_compressed(
            context_path,
            contract=np.asarray(DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT),
            estimand=np.asarray(DIRECT_TFV_POLICY_RETURN_ESTIMAND),
            action_encoding_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING),
            data_role=np.asarray(args.data_role),
            current_state=candidate_context["current_state"][None],
            rainfall_scenarios=candidate_context["rainfall_scenarios"][None],
            active_target=candidate_context["active_target"][None],
            candidate_target=target.astype(np.float32)[None],
            previous_actuator_flow=candidate_context["previous_actuator_flow"][None],
            true_policy_return_delta_tfv_m3=np.asarray([truth], dtype=np.float64),
            rainfall_group=np.asarray([args.rainfall_group]),
            query_set_id=np.asarray([query_set_id]),
            candidate_source=np.asarray([source]),
            candidate_portfolio_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT),
            supervisory_mask_sha256=np.asarray([str(control["supervisory_mask_sha256"])]),
        )
        record = {
            "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
            "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            "data_role": args.data_role,
            "rainfall_group": args.rainfall_group,
            "event_id": args.event_id,
            "decision_index": int(args.decision_index),
            "decision_elapsed_seconds": branch_elapsed,
            "query_set_id": query_set_id,
            "candidate_source": source,
            "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            "supervisory_control_contract": control["contract"],
            "supervisory_control_dimension": int(mask.sum()),
            "model_action_channel_count": 109,
            "supervisory_mask_sha256": control["supervisory_mask_sha256"],
            "passive_setting_channels_unchanged": True,
            "base_step2_h10_score_m3": base_score,
            "first_move_changed_facility_count": changed,
            "true_policy_return_delta_tfv_m3": truth,
            "candidate_branch_tfv_m3": candidate_tfv,
            "hold_branch_tfv_m3": hold_tfv,
            "shared_hold_branch": True,
            "same_prefix_verified": True,
            "same_prefix_definition": "RAW_CAUSAL_INPUTS_AND_RECORDED_SUPERVISORY_PREFIX",
            "same_prefix_derived_step1_reconstruction_max_abs": float(
                audit["derived_step1_reconstruction_max_abs"]
            ),
            "same_continuation_policy_verified": True,
            "future_realized_rainfall_used_online": False,
            "continuation_kind": args.continuation_kind,
            "continuation_policy_sha256": continuation_policy_sha,
            "prefix_sha256": prefix_sha,
            "candidate_first_target_sha256": _array_sha(target),
            "hold_first_target_sha256": _array_sha(hold_target),
            "context_npz": str(context_path),
            "context_npz_sha256": sha256_file(context_path),
            "candidate_metadata_path": candidate_result.metadata_path,
            "hold_metadata_path": hold_result.metadata_path,
            "candidate_node_statistics_path": candidate_result.node_statistics_path,
            "hold_node_statistics_path": hold_result.node_statistics_path,
            "prefix_context_audit": audit,
            "continuation_lineage": candidate_lineage,
        }
        validate_policy_return_record(record)
        records.append(record)

    if not records:
        raise RuntimeError("query runner produced no non-HOLD candidate records")
    records_path = out_root / f"{args.run_id}.policy_return_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    lifecycle_path = out_root / f"{args.run_id}.branch_memory_lifecycle.json"
    lifecycle_path.write_text(json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "contract": PRACTICAL_POLICY_RETURN_QUERY_RUNNER_CONTRACT,
        "event_id": args.event_id,
        "rainfall_group": args.rainfall_group,
        "decision_index": int(args.decision_index),
        "query_set_id": query_set_id,
        "continuation_kind": args.continuation_kind,
        "candidate_count": len(records),
        "candidate_sources": [row["candidate_source"] for row in records],
        "shared_hold_branch_count": 1,
        "candidate_branch_count": len(records),
        "authoritative_branch_count": 1 + len(records),
        "hold_tfv_m3": hold_tfv,
        "true_policy_return_delta_tfv_m3": {
            row["candidate_source"]: row["true_policy_return_delta_tfv_m3"] for row in records
        },
        "base_step2_h10_score_m3": {
            row["candidate_source"]: row["base_step2_h10_score_m3"] for row in records
        },
        "supervisory_control_dimension": int(mask.sum()),
        "model_action_channel_count": 109,
        "supervisory_mask_sha256": control["supervisory_mask_sha256"],
        "all_passive_setting_channels_unchanged": True,
        "all_same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "asset_manifest": str(Path(args.asset_manifest).resolve()),
        "candidate_manifest": str(portfolio_path),
        "records_jsonl": str(records_path),
        "records_jsonl_sha256": sha256_file(records_path),
        "lifecycle_path": str(lifecycle_path),
        "future_realized_rainfall_used_online": False,
        "legacy_v12_assets_used": False,
    }
    summary_path = out_root / f"{args.run_id}.policy_return_query_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
