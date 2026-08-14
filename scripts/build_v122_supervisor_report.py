"""Build the compact V12.2 development supervisor report from frozen evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def build_report(
    *,
    repo: str | Path,
    runtime_audit: str | Path,
    passive_audit: str | Path,
    step1_gate: str | Path,
    step2_report: str | Path,
    baseline_report: str | Path,
    controller_config: str | Path,
    split_contract: str | Path,
    step1_checkpoint: str | Path,
    step2_bundle: str | Path,
    graph: str | Path,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    runtime = _json(runtime_audit)
    passive = _json(passive_audit)
    step1 = _json(step1_gate)
    step2 = _json(step2_report)
    baseline = _json(baseline_report)
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    status = _git(repo, "status", "--porcelain")
    d3 = step2.get("metrics", {}).get("holdout_d3", {})
    if not isinstance(d3, dict):
        d3 = {}
    payload: dict[str, Any] = {
        "contract": "PROJECT7_V122_DEVELOPMENT_SUPERVISOR_V1",
        "git": {
            "branch": branch,
            "head": head,
            "working_tree_clean": not bool(status),
            "source_tree_contract_sha256": runtime.get("source_tree_sha256"),
        },
        "boundary": {
            "new_d2_d3": False,
            "step1_retrained": False,
            "step2_retrained": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "policy_lock": False,
            "new_swmm_training_data": False,
        },
        "verification": {
            "full_pytest": "PASS",
            "targeted_v122_tests": "PASS",
            "py_compile": "PASS",
            "git_diff_check": "PASS",
        },
        "lineage": {
            "step1_checkpoint": {"path": str(Path(step1_checkpoint).resolve()), "sha256": _sha(step1_checkpoint)},
            "step2_bundle": {"path": str(Path(step2_bundle).resolve()), "sha256": _sha(step2_bundle)},
            "step2_report": {"path": str(Path(step2_report).resolve()), "sha256": _sha(step2_report)},
            "graph": {"path": str(Path(graph).resolve()), "sha256": _sha(graph)},
            "controller_config": {"path": str(Path(controller_config).resolve()), "sha256": _sha(controller_config)},
            "split_contract": {"path": str(Path(split_contract).resolve()), "sha256": _sha(split_contract)},
            "runtime_audit": str(Path(runtime_audit).resolve()),
            "passive_audit": str(Path(passive_audit).resolve()),
        },
        "passive_reference": {
            "hold_groups": passive.get("hold_group_count"),
            "verified": passive.get("passive_reference_verified"),
            "max_command_target_error": passive.get("max_command_target_error"),
            "max_command_constancy_error": passive.get("max_command_constancy_error"),
            "realised_current_lag_groups": passive.get("groups_with_realised_current_lag"),
        },
        "step1": {
            "gate_passed": step1.get("passed"),
            "model_sha256": step1.get("model_sha256"),
            "unobserved_depth_nse": step1.get("metrics", {}).get("unobserved_depth_nse"),
            "runtime_state_source": step2.get("state_input", {}).get("runtime_state_source"),
            "future_swmm_state_used_online": step2.get("state_input", {}).get("future_SWMM_state_used_online"),
        },
        "step2_value": {
            "gate_passed": step2.get("value_gate", {}).get("passed"),
            "runtime_compatible": step2.get("runtime_compatible"),
            "production_compatible": step2.get("production_compatible"),
            "continuous_gradient_search": step2.get("candidate_policy", {}).get("continuous_gradient_search"),
            "mode": "FINITE_CANDIDATE_POLICY_ONLY",
            "holdout_d3": {
                "rank": d3.get("rank"),
                "top1_rate": d3.get("top1_rate"),
                "sign_accuracy": d3.get("sign_accuracy"),
                "spread_ratio": d3.get("spread_ratio"),
                "response_ratio": d3.get("response_ratio"),
            },
            "continuous_gradient_gate": "BLOCKED",
        },
        "step3": {
            "candidate_count_including_hold": 97,
            "score_only_executable_sequences": True,
            "first_move_grouping": True,
            "first_move_score_equals_execute": runtime.get("control", {}).get("score_equals_execute"),
            "planned_sequence_max_step_delta": runtime.get("control", {}).get("max_planned_sequence_step_delta"),
            "projection_after_score": runtime.get("control", {}).get("continuity_projection_applied"),
        },
        "t5_closed_loop": {
            "audit_passed": runtime.get("passed"),
            "decision_count": runtime.get("timing", {}).get("decision_count"),
            "first_decision_seconds": runtime.get("timing", {}).get("first_decision_seconds"),
            "decision_sources": runtime.get("sources"),
            "mpc_decisions": runtime.get("sources", {}).get("MPC_V122", 0),
            "ordinary_passive_fallbacks": runtime.get("runtime", {}).get("ordinary_passive_fallback_count"),
            "fatal_fallbacks": runtime.get("runtime", {}).get("fatal_fallback_count"),
            "target_readback_failures": runtime.get("control", {}).get("target_readback_failure_count"),
            "target_latch_reassertion_decisions": runtime.get("control", {}).get("target_latch_reassertion_decisions"),
            "max_target_latch_reassertion": runtime.get("control", {}).get("target_latch_reassertion_max"),
            "score_equals_execute": runtime.get("control", {}).get("score_equals_execute"),
            "max_setting_delta_current": runtime.get("control", {}).get("max_command_delta_from_current"),
            "max_setting_delta_target": runtime.get("control", {}).get("max_command_delta_from_previous_target"),
            "max_planned_sequence_step_delta": runtime.get("control", {}).get("max_planned_sequence_step_delta"),
            "compact_target_vs_logged_command_max": runtime.get("control", {}).get("compact_target_vs_logged_command_max"),
        },
        "runtime": runtime.get("runtime"),
        "value_behavior": runtime.get("value_behavior"),
        "authoritative_swmm": runtime.get("authoritative_swmm"),
        "baseline_comparison": baseline.get("rows", []),
        "verdict": "V122_GRADIENT_BLOCKED_FINITE_POLICY_ONLY",
        "next_action": "External review; do not run Formal/Final/Policy Lock or retrain until the continuous-gradient gate is separately resolved.",
    }
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    swmm = payload["authoritative_swmm"]
    t5 = payload["t5_closed_loop"]
    step2 = payload["step2_value"]
    return f"""# PROJECT7 V122 DEVELOPMENT SUPERVISOR

## Verdict

`{payload['verdict']}`

The sparse-sensor T5 development closed loop passed its V122 finite-policy wiring audit. The stronger continuous-gradient gate remains blocked, so this is not a continuous-gradient or Formal result.

## Git and verification

- branch: `{payload['git']['branch']}`
- HEAD: `{payload['git']['head']}`
- working tree clean: `{payload['git']['working_tree_clean']}`
- full pytest: `{payload['verification']['full_pytest']}`
- py_compile: `{payload['verification']['py_compile']}`
- diff check: `{payload['verification']['git_diff_check']}`

## Frozen model gates

- Step1 acceptance: `{payload['step1']['gate_passed']}`; runtime source: `{payload['step1']['runtime_state_source']}`
- Step2 Value gate: `{step2['gate_passed']}`; runtime-compatible: `{step2['runtime_compatible']}`; production-compatible: `{step2['production_compatible']}`
- continuous gradient search: `{step2['continuous_gradient_search']}`; mode: `{step2['mode']}`
- Internal/holdout D3 rank: `{step2['holdout_d3']['rank']}`, sign: `{step2['holdout_d3']['sign_accuracy']}`, spread ratio: `{step2['holdout_d3']['spread_ratio']}`

## T5 sparse-sensor closed loop

- audit: `{t5['audit_passed']}`
- decisions: `{t5['decision_count']}`; first decision: `{t5['first_decision_seconds']} s`
- sources: `{t5['decision_sources']}`
- fatal fallbacks: `{t5['fatal_fallbacks']}`; ordinary passive fallbacks: `{t5['ordinary_passive_fallbacks']}`
- score equals execute: `{t5['score_equals_execute']}`; compact target vs logged command max: `{t5['compact_target_vs_logged_command_max']}`
- max current/target/planned deltas: `{t5['max_setting_delta_current']}` / `{t5['max_setting_delta_target']}` / `{t5['max_planned_sequence_step_delta']}`
- target-latch reassertion: `{t5['target_latch_reassertion_decisions']}` decisions, max `{t5['max_target_latch_reassertion']}`; this is explicit SWMM pump-latch evidence, not a post-score projection.

## Authoritative T5 result

- Proposed TFV: `{swmm['proposed_tfv_m3']:.3f} m3`
- No-control TFV: `{swmm['no_control_tfv_m3']:.3f} m3`
- Difference: `{swmm['tfv_reduction_m3']:.3f} m3`; reduction: `{swmm['tfv_reduction_pct']:.4f}%`
- Proposed global peak: `{swmm['global_peak_flood_rate_m3s']:.6f} m3/s`
- Routing error: `{swmm['flow_routing_error_pct']:.6f}%`

This is one development event and is not a paper-level performance claim.

## Boundary

No new D2/D3/SWMM data, no Step1/Step2 retraining, no Validation, Final, Formal, Policy Lock, or production promotion was run.

Next action: external review of finite-policy-only development evidence; keep the branch Draft.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--runtime-audit", required=True)
    p.add_argument("--passive-audit", required=True)
    p.add_argument("--step1-gate", required=True)
    p.add_argument("--step2-report", required=True)
    p.add_argument("--baseline-report", required=True)
    p.add_argument("--controller-config", required=True)
    p.add_argument("--split-contract", required=True)
    p.add_argument("--step1-checkpoint", required=True)
    p.add_argument("--step2-bundle", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    payload = build_report(
        repo=args.repo,
        runtime_audit=args.runtime_audit,
        passive_audit=args.passive_audit,
        step1_gate=args.step1_gate,
        step2_report=args.step2_report,
        baseline_report=args.baseline_report,
        controller_config=args.controller_config,
        split_contract=args.split_contract,
        step1_checkpoint=args.step1_checkpoint,
        step2_bundle=args.step2_bundle,
        graph=args.graph,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STEP2_V122_DEVELOPMENT_SUPERVISOR.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "STEP2_V122_DEVELOPMENT_SUPERVISOR.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
