"""Run one exact same-prefix three-family policy-return query with one shared HOLD branch."""
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
)
from rtc.direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
from rtc.direct_tfv_policy_return_portfolio_admission import (
    CURRENT_THREE_FAMILY_SOURCES,
    validate_policy_return_portfolio_record,
)
from rtc.direct_tfv_policy_return_runtime_factory import build_frozen_policy_return_continuation_controller
from rtc.direct_tfv_sequence_support import changed_facility_support_limit, validate_direct_tfv_sequence_support
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.policy_return_replay import (
    ExactPrefixThenFrozenPolicyController,
    audit_policy_return_prefix_contexts,
    snapshot_and_release_policy_return_branch,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime


PRACTICAL_POLICY_RETURN_QUERY_RUNNER_CONTRACT = (
    "PROJECT7_PRACTICAL_POLICY_RETURN_SHARED_HOLD_QUERY_RUNNER_V4_EXACT_LINEAGE_82CONTROL_109REP"
)
PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_V4_EXACT_LINEAGE_82CONTROL_109REP"
)
_DIAGNOSTIC_ROLE = "policy_return_development_diagnostic"


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tfv(path: str | Path) -> float:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return float(sum(float(row["delta_flooding_volume_m3"]) for row in csv.DictReader(handle)))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or any(not isinstance(x, dict) for x in rows):
        raise ValueError("parent decisions JSONL is empty/invalid")
    return rows


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


def _run_branch(args, assets, kind, target, ids, prefix, elapsed, device, root, suffix):
    delegate, graph, sensors, lineage = _build_delegate(args, assets, device)
    if tuple(str(x) for x in graph.actuator_ids) != ids:
        raise ValueError("parent decision actuator order differs from frozen graph")
    wrapper = ExactPrefixThenFrozenPolicyController(
        delegate=delegate,
        actuator_ids=ids,
        prefix_actions=prefix,
        branch_elapsed_seconds=elapsed,
        branch_target=dict(zip(ids, target.tolist(), strict=True)),
        branch_kind=kind,
    )
    cfg = json.loads(Path(practical_asset_path(assets, "config")).read_text(encoding="utf-8"))
    out = root / suffix
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


def _load_portfolio(path, ids, mask, mask_sha, q95_ceiling):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("candidate manifest has the wrong current portfolio contract")
    if payload.get("context_contract") != PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT:
        raise ValueError("candidate manifest lacks current exact-lineage context")
    if str(payload.get("supervisory_mask_sha256", "")).lower() != mask_sha.lower():
        raise ValueError("candidate manifest uses another supervisory mask")
    if int(payload.get("supervisory_control_dimension", -1)) != 82 or int(mask.sum()) != 82:
        raise ValueError("candidate manifest is not the frozen 82-control policy")
    if int(payload.get("model_action_channel_count", -1)) != 109:
        raise ValueError("candidate manifest lost the 109-channel representation")
    if int(payload.get("candidate_family_count_max", -1)) != 3:
        raise ValueError("candidate manifest does not declare three-family maximum")
    if tuple(str(x) for x in payload.get("candidate_family_contract", ())) != CURRENT_THREE_FAMILY_SOURCES:
        raise ValueError("candidate manifest family contract drifted")
    if payload.get("projected_gradient_online") is not False or payload.get("lbfgsb_used") is not False:
        raise ValueError("candidate manifest escaped current finite three-family policy")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 3:
        raise ValueError("current three-family manifest must contain 1-3 candidates")
    out, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get("contract") != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
            raise ValueError("candidate manifest contains invalid row")
        row_ids = tuple(str(x) for x in row.get("actuator_ids", ()))
        if row_ids and row_ids != ids:
            raise ValueError("candidate actuator order differs from parent")
        source = str(row.get("candidate_source", ""))
        if source not in CURRENT_THREE_FAMILY_SOURCES:
            raise ValueError(f"unexpected current candidate family: {source}")
        target = np.asarray(row.get("target_settings", ()), dtype=float).reshape(-1)
        score = float(row.get("base_step2_h10_score_m3", float("nan")))
        changed = int(row.get("changed_facility_count", -1))
        if target.shape != (109,) or not np.isfinite(target).all() or not np.isfinite(score):
            raise ValueError("candidate row lacks finite 109-target/base score")
        if np.any((target < -1e-7) | (target > 1.0 + 1e-7)):
            raise ValueError("candidate target leaves [0,1]")
        if row.get("joint_sequence_support_quantile") != "q95":
            raise ValueError("candidate is not on canonical q95 support")
        if not 1 <= changed <= q95_ceiling or int(row.get("active_support_ceiling", -1)) != q95_ceiling:
            raise ValueError("candidate changed-K differs from q95 support")
        if row.get("passive_setting_channels_unchanged") is not True:
            raise ValueError("candidate row changed passive channels")
        key = np.ascontiguousarray(target, dtype=np.float64).tobytes()
        if key not in seen:
            seen.add(key)
            out.append({"source": source, "target": target, "score": score, "changed": changed})
    if not out:
        raise ValueError("candidate portfolio deduplicated to zero actions")
    return payload, out


def _require_lineage(payload, expected: dict[str, str | int]) -> None:
    for key, value in expected.items():
        actual = payload.get(key)
        if isinstance(value, int):
            if int(actual if actual is not None else -1) != value:
                raise ValueError(f"candidate manifest {key} differs from exact query")
        elif str(actual or "").lower() != value.lower():
            raise ValueError(f"candidate manifest {key} differs from exact query")


def _validate_record(record: dict[str, Any]) -> None:
    for key in (
        "same_prefix_verified",
        "same_continuation_policy_verified",
        "passive_setting_channels_unchanged",
        "target_write_readback_verified",
        "engineering_bounds_verified",
        "candidate_manifest_support_lineage_verified",
    ):
        if record.get(key) is not True:
            raise ValueError(f"authoritative truth failed gate: {key}")
    if record.get("projected_gradient_online") is not False or record.get("online_lbfgsb_used") is not False:
        raise ValueError("authoritative truth escaped current finite portfolio")
    for key in (
        "candidate_manifest_sha256",
        "parent_decisions_sha256",
        "source_inp_sha256",
        "asset_manifest_sha256",
        "graph_sha256",
        "base_step2_sha256",
        "sequence_support_sha256",
        "supervisory_control_sha256",
    ):
        value = str(record.get(key, "")).lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"authoritative truth lacks canonical {key}")
    for key in ("candidate_flow_routing_error_pct", "hold_flow_routing_error_pct"):
        if not np.isfinite(float(record.get(key, float("nan")))):
            raise ValueError(f"authoritative truth lacks finite {key}")
    validate_policy_return_portfolio_record(record)


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
        choices=(
            "policy_return_train",
            "policy_return_validation",
            "policy_return_calibration",
            _DIAGNOSTIC_ROLE,
        ),
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
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    inp = Path(args.inp).resolve()
    decisions_path = Path(args.parent_decisions).resolve()
    portfolio_path = Path(args.candidate_manifest).resolve()
    if not all(x.is_file() for x in (inp, decisions_path, portfolio_path)):
        raise FileNotFoundError("query runner requires INP, parent decisions and candidate manifest")
    rows = _load_rows(decisions_path)
    if not 0 <= args.decision_index < len(rows):
        raise ValueError("decision-index lies outside parent decisions")
    selected = rows[args.decision_index]
    diag = selected.get("diagnostics")
    if not isinstance(diag, dict):
        raise ValueError("selected parent decision lacks diagnostics")
    ids = tuple(str(x) for x in diag.get("counterfactual_actuator_ids", ()))
    hold = np.asarray(diag.get("hold_reference_settings", ()), dtype=float).reshape(-1)
    if len(ids) != 109 or len(set(ids)) != 109 or hold.shape != (109,):
        raise ValueError("selected parent decision lacks 109-channel HOLD context")

    graph_path = practical_asset_path(assets, "graph")
    step2_path = practical_asset_path(assets, "step2")
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
    control, mask = load_native_supervisory_control(control_path, actuator_ids=ids)
    support = json.loads(Path(support_path).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=ids,
        step2_checkpoint_sha256=sha256_file(step2_path),
        supervisory_mask=mask,
        supervisory_control_contract=str(control["contract"]),
    )
    q95_ceiling = changed_facility_support_limit(support, "q95")
    manifest, candidates = _load_portfolio(
        portfolio_path, ids, mask, str(control["supervisory_mask_sha256"]), q95_ceiling
    )

    elapsed = int(selected["elapsed_seconds"])
    prefix = {}
    for row in rows[: args.decision_index]:
        settings = row.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(ids):
            raise ValueError("parent prefix lacks complete 109-target action")
        prefix[int(row["elapsed_seconds"])] = {aid: float(settings[aid]) for aid in ids}
    prefix_sha = _canonical_sha({str(k): prefix[k] for k in sorted(prefix)})
    expected = {
        "source_inp_sha256": sha256_file(inp),
        "parent_decisions_sha256": sha256_file(decisions_path),
        "asset_manifest_sha256": sha256_file(args.asset_manifest),
        "graph_sha256": sha256_file(graph_path),
        "step2_checkpoint_sha256": sha256_file(step2_path),
        "sequence_support_sha256": sha256_file(support_path),
        "supervisory_control_sha256": sha256_file(control_path),
        "event_id": args.event_id,
        "rainfall_group": args.rainfall_group,
        "decision_index": args.decision_index,
        "decision_elapsed_seconds": elapsed,
        "recorded_prefix_action_sha256": prefix_sha,
        "continuation_kind": args.continuation_kind,
    }
    _require_lineage(manifest, expected)
    context_path = Path(str(manifest.get("context_npz", "")))
    if not context_path.is_file() or sha256_file(context_path).lower() != str(
        manifest.get("context_npz_sha256", "")
    ).lower():
        raise ValueError("candidate manifest context is missing or SHA-mismatched")

    for row in candidates:
        target = row["target"]
        changed = int(np.count_nonzero(np.abs(target - hold) > 1e-7))
        if changed != row["changed"]:
            raise ValueError(f"candidate {row['source']} changed-K declaration is stale")
        if np.any(np.abs(target[~mask] - hold[~mask]) > 1e-7):
            raise ValueError(f"candidate {row['source']} changes passive settings")
        if float(np.max(np.abs(target - hold), initial=0.0)) > 0.5 + 1e-7:
            raise ValueError(f"candidate {row['source']} exceeds 0.5 command slew")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("policy-return query requested CUDA but CUDA is unavailable")
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        print(json.dumps({
            "contract": PRACTICAL_POLICY_RETURN_QUERY_RUNNER_CONTRACT,
            "preflight_only": True,
            "swmm_truth_started": False,
            "candidate_count": len(candidates),
            "candidate_sources": [x["source"] for x in candidates],
            "candidate_family_count_max": 3,
            "supervisory_control_dimension": 82,
            "model_action_channel_count": 109,
            "projected_gradient_online": False,
            "online_lbfgsb_used": False,
            "exact_query_lineage_verified": True,
            "q95_support_lineage_verified": True,
        }, indent=2))
        return

    hold_result, hold_wrapper, hold_lineage = _run_branch(
        args, assets, "HOLD", hold, ids, prefix, elapsed, device, root, "hold_shared"
    )
    hold_write = audit_target_write_readback_v127(metadata_path=hold_result.metadata_path)
    if hold_write.get("passed") is not True:
        raise RuntimeError("shared HOLD failed target write/readback audit")
    hold_context, hold_release = snapshot_and_release_policy_return_branch(hold_wrapper, device=device)
    del hold_wrapper
    hold_tfv = _tfv(hold_result.node_statistics_path)
    continuation_sha = _canonical_sha(
        {"continuation_kind": args.continuation_kind, "lineage": hold_lineage}
    )
    if str(manifest.get("continuation_policy_sha256", "")).lower() != continuation_sha.lower():
        raise RuntimeError("candidate manifest continuation lineage differs from runtime")
    query_id = _canonical_sha({
        "event_id": args.event_id,
        "rainfall_group": args.rainfall_group,
        "decision_index": args.decision_index,
        "decision_elapsed_seconds": elapsed,
        "prefix_sha256": prefix_sha,
        "hold_first_target_sha256": _array_sha(hold),
        "continuation_policy_sha256": continuation_sha,
        "supervisory_mask_sha256": control["supervisory_mask_sha256"],
    })
    hashes = {
        "candidate_manifest_sha256": sha256_file(portfolio_path),
        "parent_decisions_sha256": sha256_file(decisions_path),
        "source_inp_sha256": sha256_file(inp),
        "asset_manifest_sha256": sha256_file(args.asset_manifest),
        "graph_sha256": sha256_file(graph_path),
        "base_step2_sha256": sha256_file(step2_path),
        "sequence_support_sha256": sha256_file(support_path),
        "supervisory_control_sha256": sha256_file(control_path),
    }

    records, releases = [], {"hold_release": hold_release, "candidate_releases": {}}
    for index, candidate in enumerate(candidates):
        source, target, score = candidate["source"], candidate["target"], candidate["score"]
        changed = int(np.count_nonzero(np.abs(target - hold) > 1e-7))
        result, wrapper, lineage = _run_branch(
            args, assets, "CANDIDATE", target, ids, prefix, elapsed, device, root, f"candidate_{index:02d}"
        )
        write_audit = audit_target_write_readback_v127(metadata_path=result.metadata_path)
        if write_audit.get("passed") is not True:
            raise RuntimeError(f"candidate {source} failed target write/readback audit")
        context, release = snapshot_and_release_policy_return_branch(wrapper, device=device)
        del wrapper
        releases["candidate_releases"][source] = release
        if lineage != hold_lineage:
            raise RuntimeError(f"candidate {source} used different continuation policy")
        prefix_audit = audit_policy_return_prefix_contexts(context, hold_context)
        if prefix_audit["same_authoritative_prefix_verified"] is not True:
            raise RuntimeError(f"candidate {source} raw causal prefix differs from shared HOLD")
        candidate_tfv = _tfv(result.node_statistics_path)
        truth = float(candidate_tfv - hold_tfv)
        context_out = root / f"{args.run_id}.{index:02d}.policy_return_context.npz"
        np.savez_compressed(
            context_out,
            contract=np.asarray(DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT),
            estimand=np.asarray(DIRECT_TFV_POLICY_RETURN_ESTIMAND),
            action_encoding_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING),
            data_role=np.asarray(args.data_role),
            current_state=context["current_state"][None],
            rainfall_scenarios=context["rainfall_scenarios"][None],
            active_target=context["active_target"][None],
            candidate_target=target.astype(np.float32)[None],
            previous_actuator_flow=context["previous_actuator_flow"][None],
            true_policy_return_delta_tfv_m3=np.asarray([truth], dtype=np.float64),
            rainfall_group=np.asarray([args.rainfall_group]),
            query_set_id=np.asarray([query_id]),
            candidate_source=np.asarray([source]),
            candidate_portfolio_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT),
            supervisory_mask_sha256=np.asarray([str(control["supervisory_mask_sha256"])]),
        )
        diagnostic = args.data_role == _DIAGNOSTIC_ROLE
        record = {
            "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
            "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            "data_role": args.data_role,
            "development_diagnostic_only": diagnostic,
            "eligible_for_learning_dataset": not diagnostic,
            "rainfall_group": args.rainfall_group,
            "event_id": args.event_id,
            "decision_index": args.decision_index,
            "decision_elapsed_seconds": elapsed,
            "query_set_id": query_id,
            "candidate_source": source,
            "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            "supervisory_control_contract": control["contract"],
            "supervisory_control_dimension": 82,
            "model_action_channel_count": 109,
            "supervisory_mask_sha256": control["supervisory_mask_sha256"],
            "passive_setting_channels_unchanged": True,
            "base_step2_h10_score_m3": float(score),
            "first_move_changed_facility_count": changed,
            "true_policy_return_delta_tfv_m3": truth,
            "candidate_branch_tfv_m3": candidate_tfv,
            "hold_branch_tfv_m3": hold_tfv,
            "shared_hold_branch": True,
            "same_prefix_verified": True,
            "same_prefix_definition": "RAW_CAUSAL_INPUTS_AND_RECORDED_SUPERVISORY_PREFIX",
            "same_prefix_derived_step1_reconstruction_max_abs": float(
                prefix_audit["derived_step1_reconstruction_max_abs"]
            ),
            "same_continuation_policy_verified": True,
            "future_realized_rainfall_used_online": False,
            "continuation_kind": args.continuation_kind,
            "continuation_policy_sha256": continuation_sha,
            "prefix_sha256": prefix_sha,
            "candidate_first_target_sha256": _array_sha(target),
            "hold_first_target_sha256": _array_sha(hold),
            "context_npz": str(context_out),
            "context_npz_sha256": sha256_file(context_out),
            **hashes,
            "target_write_readback_verified": True,
            "candidate_target_write_readback_audit": write_audit,
            "hold_target_write_readback_audit": hold_write,
            "engineering_bounds_verified": True,
            "candidate_manifest_support_lineage_verified": True,
            "projected_gradient_online": False,
            "online_lbfgsb_used": False,
            "candidate_flow_routing_error_pct": float(result.flow_routing_error_pct),
            "hold_flow_routing_error_pct": float(hold_result.flow_routing_error_pct),
            "routing_quality_threshold_applied": False,
            "routing_quality_note": "raw SWMM routing errors recorded; no new threshold invented",
            "candidate_metadata_path": result.metadata_path,
            "hold_metadata_path": hold_result.metadata_path,
            "candidate_node_statistics_path": result.node_statistics_path,
            "hold_node_statistics_path": hold_result.node_statistics_path,
            "prefix_context_audit": prefix_audit,
            "continuation_lineage": lineage,
        }
        _validate_record(record)
        records.append(record)

    if not records:
        raise RuntimeError("query runner produced no non-HOLD candidate records")
    records_path = root / f"{args.run_id}.policy_return_records.jsonl"
    records_path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in records), encoding="utf-8")
    lifecycle_path = root / f"{args.run_id}.branch_memory_lifecycle.json"
    lifecycle_path.write_text(json.dumps(releases, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "contract": PRACTICAL_POLICY_RETURN_QUERY_RUNNER_CONTRACT,
        "event_id": args.event_id,
        "rainfall_group": args.rainfall_group,
        "data_role": args.data_role,
        "development_diagnostic_only": args.data_role == _DIAGNOSTIC_ROLE,
        "eligible_for_learning_dataset": args.data_role != _DIAGNOSTIC_ROLE,
        "decision_index": args.decision_index,
        "query_set_id": query_id,
        "continuation_kind": args.continuation_kind,
        "candidate_count": len(records),
        "candidate_sources": [x["candidate_source"] for x in records],
        "candidate_family_count_max": 3,
        "shared_hold_branch_count": 1,
        "candidate_branch_count": len(records),
        "authoritative_branch_count": 1 + len(records),
        "hold_tfv_m3": hold_tfv,
        "true_policy_return_delta_tfv_m3": {
            x["candidate_source"]: x["true_policy_return_delta_tfv_m3"] for x in records
        },
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "supervisory_mask_sha256": control["supervisory_mask_sha256"],
        "all_passive_setting_channels_unchanged": True,
        "all_same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "all_target_write_readback_verified": True,
        "exact_query_lineage_verified": True,
        "q95_support_lineage_verified": True,
        "projected_gradient_online": False,
        "online_lbfgsb_used": False,
        **hashes,
        "records_jsonl": str(records_path),
        "records_jsonl_sha256": sha256_file(records_path),
        "lifecycle_path": str(lifecycle_path),
        "future_realized_rainfall_used_online": False,
        "legacy_v12_assets_used": False,
    }
    summary_path = root / f"{args.run_id}.policy_return_query_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
