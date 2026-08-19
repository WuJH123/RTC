"""Capture one exact-prefix causal query context and stop before the branch action is written.

The default first-round continuation is the current Practical base-H10-probe parent pi0, not a
historical V12 optimizer. After a policy-return critic exists the same script can capture contexts
under frozen pi1. It produces no TFV truth and cannot be used as evaluation evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.direct_tfv_base_probe_runtime_factory import build_frozen_base_probe_parent_controller
from rtc.direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
from rtc.direct_tfv_policy_return_runtime_factory import build_frozen_policy_return_continuation_controller
from rtc.policy_return_replay import ExactPrefixThenFrozenPolicyController, snapshot_and_release_policy_return_branch
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime


PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT = "PROJECT7_PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_V2_BASE_PROBE_PI0"


class _ContextCaptured(RuntimeError):
    pass


class _CaptureOnlyController(ExactPrefixThenFrozenPolicyController):
    def decide(self, obs, *, observation_already_recorded: bool = False):
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        if int(obs.elapsed_seconds) == int(self.branch_elapsed_seconds):
            if self.branch_context is None:
                raise RuntimeError("capture-only controller failed to capture causal context")
            raise _ContextCaptured("causal policy-return context captured before branch write")
        return action


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("parent decisions JSONL is empty/invalid")
    return rows


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_delegate(args: argparse.Namespace, assets: dict, device: torch.device):
    common = dict(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=practical_asset_path(assets, "step2"),
        sequence_support_path=practical_asset_path(assets, "sequence_support"),
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
    )
    if args.continuation_kind == "base-probe":
        if args.policy_return_checkpoint or args.policy_return_admission:
            raise ValueError("base-probe pi0 must not receive policy-return critic/admission")
        return build_frozen_base_probe_parent_controller(
            **common,
            proposal_probe_chunk_size=int(args.probe_chunk_size),
        )
    if not args.policy_return_checkpoint or not args.policy_return_admission:
        raise ValueError("policy-return pi1 context capture requires critic and admission")
    return build_frozen_policy_return_continuation_controller(
        **common,
        policy_return_checkpoint_path=args.policy_return_checkpoint,
        policy_return_admission_path=args.policy_return_admission,
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-manifest", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--parent-decisions", required=True)
    p.add_argument("--decision-index", type=int, required=True)
    p.add_argument("--event-id", required=True)
    p.add_argument("--rainfall-group", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--continuation-kind", choices=("base-probe", "policy-return"), default="base-probe")
    p.add_argument("--policy-return-checkpoint")
    p.add_argument("--policy-return-admission")
    p.add_argument("--device", default="cuda")
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--probe-chunk-size", type=int, default=24)
    args = p.parse_args()

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    inp = Path(args.inp).resolve()
    decisions = Path(args.parent_decisions).resolve()
    if not inp.is_file() or not decisions.is_file():
        raise FileNotFoundError("context capture requires existing INP and parent decisions")
    rows = _load_rows(decisions)
    if not 0 <= int(args.decision_index) < len(rows):
        raise ValueError("decision-index lies outside parent decisions")
    selected = rows[int(args.decision_index)]
    diagnostics = selected.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("selected parent decision lacks diagnostics")
    ids = tuple(str(x) for x in diagnostics.get("counterfactual_actuator_ids", ()))
    hold = np.asarray(diagnostics.get("hold_reference_settings", ()), dtype=float).reshape(-1)
    if len(ids) != 109 or len(set(ids)) != 109 or hold.shape != (109,):
        raise ValueError("selected parent decision lacks 109-actuator HOLD context")
    branch_elapsed = int(selected["elapsed_seconds"])
    prefix: dict[int, dict[str, float]] = {}
    for row in rows[: int(args.decision_index)]:
        settings = row.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(ids):
            raise ValueError("parent prefix decision lacks complete 109-target action")
        prefix[int(row["elapsed_seconds"])] = {aid: float(settings[aid]) for aid in ids}
    prefix_sha = _canonical_sha({str(k): prefix[k] for k in sorted(prefix)})

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    delegate, graph, sensors, lineage = _build_delegate(args, assets, device)
    if tuple(str(x) for x in graph.actuator_ids) != ids:
        raise ValueError("parent decision actuator order differs from frozen graph")
    wrapper = _CaptureOnlyController(
        delegate=delegate,
        actuator_ids=ids,
        prefix_actions=prefix,
        branch_elapsed_seconds=branch_elapsed,
        branch_target=dict(zip(ids, hold.tolist(), strict=True)),
        branch_kind="HOLD",
    )
    cfg = json.loads(Path(practical_asset_path(assets, "config")).read_text(encoding="utf-8"))
    work = Path(args.work_dir).resolve()
    runtime_inp = _controls_disabled_runtime(
        source_inp=inp,
        cache_dir=work / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    intentionally_stopped = False
    try:
        run_authoritative_closed_loop(
            inp_path=runtime_inp,
            output_dir=work,
            run_id="context_capture_only",
            sensor_nodes=sensors,
            controller=wrapper,
            control_start_minutes=int(cfg["control_start_minutes"]),
            control_update_seconds=600,
            observation_update_seconds=300,
            record_stride_seconds=300,
            exact_global_peak=False,
        )
    except _ContextCaptured:
        intentionally_stopped = True
    if not intentionally_stopped or wrapper.branch_context is None:
        raise RuntimeError("context capture did not stop at the requested causal branch point")
    context, release = snapshot_and_release_policy_return_branch(wrapper, device=device)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        contract=np.asarray(PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT),
        action_encoding_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING),
        event_id=np.asarray(args.event_id),
        rainfall_group=np.asarray(args.rainfall_group),
        decision_index=np.asarray(int(args.decision_index), dtype=np.int64),
        decision_elapsed_seconds=np.asarray(branch_elapsed, dtype=np.int64),
        recorded_prefix_action_sha256=np.asarray(prefix_sha),
        current_state=context["current_state"][None],
        rainfall_scenarios=context["rainfall_scenarios"][None],
        active_target=context["active_target"][None],
        previous_actuator_flow=context["previous_actuator_flow"][None],
    )
    summary = {
        "contract": PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "event_id": args.event_id,
        "rainfall_group": args.rainfall_group,
        "decision_index": int(args.decision_index),
        "decision_elapsed_seconds": branch_elapsed,
        "recorded_prefix_action_sha256": prefix_sha,
        "continuation_kind": args.continuation_kind,
        "continuation_lineage": lineage,
        "asset_manifest": str(Path(args.asset_manifest).resolve()),
        "context_npz": str(out),
        "context_npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "intentional_prefix_only_stop": True,
        "authoritative_tfv_truth_generated": False,
        "future_realized_rainfall_used_online": False,
        "branch_release": release,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
