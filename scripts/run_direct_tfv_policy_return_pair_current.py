"""Run one exact-prefix CANDIDATE/HOLD pair followed by one frozen continuation policy.

This is offline authoritative label generation only. Candidate and HOLD replay the identical recorded
supervisory prefix, differ for exactly one H10 command, and then delegate to the same frozen policy.
The resulting context is stamped with the same H10-candidate/H350-HOLD action-token contract used by
policy-return training, calibration and runtime. Branch controllers are sequentially released so one
8-GB GPU never retains two complete continuation stacks.
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
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
)
from rtc.direct_tfv_policy_return_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
from rtc.direct_tfv_policy_return_runtime_factory import build_frozen_policy_return_continuation_controller
from rtc.direct_tfv_runtime_factory import build_frozen_v12_continuation_controller
from rtc.policy_return_replay import (
    ExactPrefixThenFrozenPolicyController,
    audit_policy_return_prefix_contexts,
    snapshot_and_release_policy_return_branch,
)
from rtc.production_cli import _controls_disabled_runtime


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


def _load_decisions(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("parent decision JSONL contains a non-object row")
            rows.append(row)
    if not rows:
        raise ValueError("parent decision JSONL is empty")
    return rows


def _targets(row: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("selected parent decision lacks diagnostics")
    ids = tuple(str(x) for x in diagnostics.get("counterfactual_actuator_ids", ()))
    blocks = diagnostics.get("optimized_free_control_blocks")
    hold = diagnostics.get("hold_reference_settings")
    if len(ids) != 109 or len(set(ids)) != 109:
        raise ValueError("selected parent decision lacks 109 counterfactual actuator IDs")
    if not isinstance(blocks, list) or not blocks or not isinstance(blocks[0], list) or len(blocks[0]) != 109:
        raise ValueError("selected parent decision lacks executable first candidate target")
    if not isinstance(hold, list) or len(hold) != 109:
        raise ValueError("selected parent decision lacks previous-target HOLD reference")
    candidate = np.asarray(blocks[0], dtype=float)
    hold_target = np.asarray(hold, dtype=float)
    if not np.isfinite(candidate).all() or not np.isfinite(hold_target).all():
        raise ValueError("selected policy-return targets contain non-finite values")
    return ids, candidate, hold_target


def _candidate_override(
    path: str | Path | None,
    *,
    ids: tuple[str, ...],
    parent_candidate: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    if path is None:
        # Context/smoke mode only. These rows are not accepted by the practical dataset compiler.
        return parent_candidate, "LEGACY_PARENT_CONTEXT_PROBE", ""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("candidate target JSON has the wrong practical portfolio contract")
    payload_ids = tuple(str(value) for value in payload.get("actuator_ids", ()))
    if payload_ids and payload_ids != ids:
        raise ValueError("candidate target JSON actuator order differs from parent decision")
    target = np.asarray(payload.get("target_settings", ()), dtype=float).reshape(-1)
    if target.shape != (109,) or not np.isfinite(target).all():
        raise ValueError("candidate target JSON must contain 109 finite settings")
    if np.any(target < -1.0e-7) or np.any(target > 1.0 + 1.0e-7):
        raise ValueError("candidate target JSON leaves physical [0,1] bounds")
    source = str(payload.get("candidate_source", "")).strip()
    if not source:
        raise ValueError("candidate target JSON lacks candidate_source")
    return target, source, DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT


def _build_delegate(args: argparse.Namespace, device: torch.device):
    common = dict(
        graph_path=args.graph,
        sensors_path=args.sensors,
        config_path=args.config,
        step1_path=args.step1,
        step2_path=args.step2,
        policy_admission_path=args.policy_admission,
        sequence_support_path=args.sequence_support,
        device=device,
        lbfgsb_maxiter=int(args.lbfgsb_maxiter),
        optimizer_deadline_seconds=float(args.optimizer_deadline_seconds),
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        first_move_maxiter=int(args.first_move_maxiter),
        first_move_deadline_seconds=float(args.first_move_deadline_seconds),
    )
    if args.continuation_kind == "v12":
        if args.policy_return_checkpoint or args.policy_return_admission:
            raise ValueError("V12 parent continuation must not receive policy-return critic/admission")
        return build_frozen_v12_continuation_controller(
            **common,
            first_move_admission_path=args.v12_first_move_admission,
        )
    if not args.policy_return_checkpoint or not args.policy_return_admission:
        raise ValueError("policy-return continuation requires critic and matched admission")
    return build_frozen_policy_return_continuation_controller(
        **common,
        v12_first_move_admission_path=args.v12_first_move_admission,
        policy_return_checkpoint_path=args.policy_return_checkpoint,
        policy_return_admission_path=args.policy_return_admission,
    )


def _run_branch(
    *,
    args: argparse.Namespace,
    branch_kind: str,
    branch_target: np.ndarray,
    ids: tuple[str, ...],
    prefix_actions: dict[int, dict[str, float]],
    branch_elapsed: int,
    device: torch.device,
    out_root: Path,
):
    delegate, graph, sensors, lineage = _build_delegate(args, device)
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
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = out_root / branch_kind.lower()
    runtime_inp = _controls_disabled_runtime(
        source_inp=Path(args.inp),
        cache_dir=out / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=out,
        run_id=f"{args.run_id}_{branch_kind.lower()}",
        sensor_nodes=sensors,
        controller=wrapper,
        control_start_minutes=int(cfg["control_start_minutes"]),
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=False,
    )
    if wrapper.branch_context is None:
        raise RuntimeError("policy-return replay did not capture branch causal context")
    return result, wrapper, lineage


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inp", required=True)
    p.add_argument("--parent-decisions", required=True)
    p.add_argument("--decision-index", type=int, required=True)
    p.add_argument("--rainfall-group", required=True)
    p.add_argument("--event-id", required=True)
    p.add_argument("--data-role", choices=("policy_return_train", "policy_return_validation", "policy_return_calibration"), required=True)
    p.add_argument("--continuation-kind", choices=("v12", "policy-return"), default="v12")
    p.add_argument("--candidate-target-json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--policy-admission", required=True)
    p.add_argument("--v12-first-move-admission", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--policy-return-checkpoint")
    p.add_argument("--policy-return-admission")
    p.add_argument("--device", default="cpu")
    # Legacy parent-continuation knobs remain only for pi0 label generation. Practical online control
    # does not use L-BFGS-B after the first policy-return critic is trained.
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--optimizer-deadline-seconds", type=float, default=120.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--first-move-maxiter", type=int, default=12)
    p.add_argument("--first-move-deadline-seconds", type=float, default=30.0)
    args = p.parse_args()

    rows = _load_decisions(args.parent_decisions)
    if not 0 <= args.decision_index < len(rows):
        raise ValueError("decision-index lies outside parent decision JSONL")
    selected = rows[args.decision_index]
    ids, parent_candidate, hold_target = _targets(selected)
    candidate_target, candidate_source, portfolio_contract = _candidate_override(
        args.candidate_target_json, ids=ids, parent_candidate=parent_candidate
    )
    branch_elapsed = int(selected["elapsed_seconds"])
    if branch_elapsed <= 0:
        raise ValueError("policy-return branch must occur after causal warm-up")
    changed = int(np.sum(np.abs(candidate_target - hold_target) > 1.0e-7))
    if changed <= 0:
        raise ValueError("selected policy-return query has zero changed facilities")
    prefix_actions: dict[int, dict[str, float]] = {}
    for row in rows[: args.decision_index]:
        elapsed = int(row["elapsed_seconds"])
        settings = row.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(ids):
            raise ValueError("parent prefix decision lacks a complete 109-target action")
        prefix_actions[elapsed] = {aid: float(settings[aid]) for aid in ids}
    prefix_sha = _canonical_sha({str(k): prefix_actions[k] for k in sorted(prefix_actions)})

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    lifecycle_path = out_root / f"{args.run_id}.branch_memory_lifecycle.json"
    prefix_audit_path = out_root / f"{args.run_id}.prefix_context_audit.json"

    candidate_result, candidate_wrapper, candidate_lineage = _run_branch(
        args=args, branch_kind="CANDIDATE", branch_target=candidate_target, ids=ids,
        prefix_actions=prefix_actions, branch_elapsed=branch_elapsed, device=device, out_root=out_root,
    )
    candidate_context, candidate_release = snapshot_and_release_policy_return_branch(candidate_wrapper, device=device)
    del candidate_wrapper
    lifecycle: dict[str, Any] = {
        "candidate_release": candidate_release,
        "hold_release": None,
        "sequential_branch_controller_residency": True,
        "scientific_action_changed_by_cleanup": False,
    }
    _write_json(lifecycle_path, lifecycle)

    hold_result, hold_wrapper, hold_lineage = _run_branch(
        args=args, branch_kind="HOLD", branch_target=hold_target, ids=ids,
        prefix_actions=prefix_actions, branch_elapsed=branch_elapsed, device=device, out_root=out_root,
    )
    hold_context, hold_release = snapshot_and_release_policy_return_branch(hold_wrapper, device=device)
    del hold_wrapper
    lifecycle["hold_release"] = hold_release
    _write_json(lifecycle_path, lifecycle)

    if candidate_lineage != hold_lineage:
        raise RuntimeError("candidate/HOLD branches used different continuation-policy artifacts")
    prefix_audit = audit_policy_return_prefix_contexts(candidate_context, hold_context)
    prefix_audit.update(
        {
            "event_id": args.event_id,
            "rainfall_group": args.rainfall_group,
            "decision_index": int(args.decision_index),
            "decision_elapsed_seconds": branch_elapsed,
            "recorded_prefix_action_sha256": prefix_sha,
            "same_continuation_policy_verified": True,
        }
    )
    _write_json(prefix_audit_path, prefix_audit)
    if prefix_audit["same_authoritative_prefix_verified"] is not True:
        raise RuntimeError(f"candidate/HOLD raw causal prefix mismatch; see {prefix_audit_path}")

    candidate_tfv = _tfv(candidate_result.node_statistics_path)
    hold_tfv = _tfv(hold_result.node_statistics_path)
    truth = float(candidate_tfv - hold_tfv)
    continuation_policy_sha = _canonical_sha({"continuation_kind": args.continuation_kind, "lineage": candidate_lineage})
    query_set_id = _canonical_sha(
        {
            "event_id": args.event_id,
            "rainfall_group": args.rainfall_group,
            "decision_index": int(args.decision_index),
            "decision_elapsed_seconds": branch_elapsed,
            "prefix_sha256": prefix_sha,
            "hold_first_target_sha256": _array_sha(hold_target),
            "continuation_policy_sha256": continuation_policy_sha,
        }
    )
    context_path = out_root / f"{args.run_id}.policy_return_context.npz"
    np.savez_compressed(
        context_path,
        contract=np.asarray(DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT),
        estimand=np.asarray(DIRECT_TFV_POLICY_RETURN_ESTIMAND),
        action_encoding_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING),
        data_role=np.asarray(args.data_role),
        current_state=candidate_context["current_state"][None],
        rainfall_scenarios=candidate_context["rainfall_scenarios"][None],
        active_target=candidate_context["active_target"][None],
        candidate_target=candidate_target.astype(np.float32)[None],
        previous_actuator_flow=candidate_context["previous_actuator_flow"][None],
        true_policy_return_delta_tfv_m3=np.asarray([truth], dtype=np.float64),
        rainfall_group=np.asarray([args.rainfall_group]),
        query_set_id=np.asarray([query_set_id]),
        candidate_source=np.asarray([candidate_source]),
        candidate_portfolio_contract=np.asarray(portfolio_contract),
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
        "candidate_source": candidate_source,
        "candidate_portfolio_contract": portfolio_contract,
        "first_move_changed_facility_count": changed,
        "true_policy_return_delta_tfv_m3": truth,
        "candidate_branch_tfv_m3": candidate_tfv,
        "hold_branch_tfv_m3": hold_tfv,
        "same_prefix_verified": True,
        "same_prefix_definition": "RAW_CAUSAL_INPUTS_AND_RECORDED_SUPERVISORY_PREFIX",
        "same_prefix_derived_step1_reconstruction_max_abs": float(prefix_audit["derived_step1_reconstruction_max_abs"]),
        "same_continuation_policy_verified": True,
        "future_realized_rainfall_used_online": False,
        "continuation_kind": args.continuation_kind,
        "continuation_policy_sha256": continuation_policy_sha,
        "prefix_sha256": prefix_sha,
        "candidate_first_target_sha256": _array_sha(candidate_target),
        "hold_first_target_sha256": _array_sha(hold_target),
        "context_npz": str(context_path.resolve()),
        "context_npz_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "candidate_metadata_path": candidate_result.metadata_path,
        "hold_metadata_path": hold_result.metadata_path,
        "candidate_node_statistics_path": candidate_result.node_statistics_path,
        "hold_node_statistics_path": hold_result.node_statistics_path,
        "branch_memory_lifecycle_path": str(lifecycle_path.resolve()),
        "branch_memory_lifecycle": lifecycle,
        "prefix_context_audit_path": str(prefix_audit_path.resolve()),
        "prefix_context_audit": prefix_audit,
        "continuation_lineage": candidate_lineage,
    }
    record_path = out_root / f"{args.run_id}.policy_return_record.json"
    _write_json(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
