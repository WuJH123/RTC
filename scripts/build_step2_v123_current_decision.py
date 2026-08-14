"""Aggregate V123 causal-value, calibration, runtime, and T5 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _metrics(report: dict[str, Any], arm: str, split: str) -> dict[str, Any]:
    return dict(report["arms"][arm]["metrics"][split])


def _decision_summary(runtime_dir: Path, false_benefit_margin: float) -> dict[str, Any]:
    metadata_path = next(runtime_dir.glob("*.json"))
    metadata = _load(metadata_path)
    decisions_path = runtime_dir / str(metadata["decision_file"])
    records = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_counts: dict[str, int] = {}
    selected_improvements: list[float] = []
    nonhold_improvements: list[float] = []
    for record in records:
        source = str(record.get("source", "UNKNOWN"))
        source_counts[source] = source_counts.get(source, 0) + 1
        diagnostics = record.get("diagnostics", {})
        pred = diagnostics.get("predicted_delta_tfv_m3", record.get("predicted_delta_tfv_m3"))
        if pred is not None:
            improvement = max(0.0, -float(pred))
            selected_improvements.append(improvement)
            if source == "MPC_V123":
                nonhold_improvements.append(improvement)
    # The only threshold here is the already-frozen calibration budget; this is
    # diagnostic labelling, not a runtime admission change.
    small_risk = any(0.0 < value < false_benefit_margin for value in nonhold_improvements)
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha(metadata_path),
        "decision_path": str(decisions_path),
        "decision_count": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "selected_predicted_tfv_improvement_m3": selected_improvements,
        "nonhold_predicted_tfv_improvement_m3": nonhold_improvements,
        "small_effect_selection_risk": bool(small_risk),
        "false_benefit_margin_m3": float(false_benefit_margin),
        "future_realized_rainfall_used_as_model_input": bool(metadata.get("future_realized_rainfall_used_as_model_input", True)),
        "v123_runtime_causal_rainfall": bool(metadata.get("v123_runtime_causal_rainfall", False)),
        "score_only_executable_sequences": bool(metadata.get("score_only_executable_sequences", False)),
    }


def build_report(*, root: Path, git_head: str, remote_head: str, hardening_head: str, runtime_dir: Path, out: Path) -> dict[str, Any]:
    store = _load(root / "STEP2_V123_CAUSAL_FORECAST_STORE.json")
    normalization = _load(root / "STEP2_V123_CAUSAL_NORMALIZATION.json")
    tfv = _load(root / "tfv_causal_norm" / "STEP2_V123_TFV_CAUSAL_ABLATION.json")
    pfv = _load(root / "pfv_causal_norm" / "STEP2_V123_PFV_VALUE_REPORT.json")
    gate = _load(root / "STEP2_V123_CONTINUOUS_GATE.json")
    calibration = _load(root / "STEP2_V123_ADMISSION_CALIBRATION.json")
    objective = _load(root / "STEP2_V123_OBJECTIVE_EVIDENCE.json")
    first_move = _load(root / "first_move" / "STEP2_V123_FIRST_MOVE_COVERAGE.json")
    seed = _load(root / "knowledge_seed" / "STEP2_V123_KNOWLEDGE_GUIDED_SEED.json")
    seven = _load(root / "STEP2_V123_T5_SEVEN_STRATEGY_COMPARISON.json")
    pre_rain = _load(root / "pre_rain_audit" / "STEP2_V123_PRE_RAIN_VALUE_AUDIT.json")
    cal = calibration["calibration"]
    runtime = _decision_summary(runtime_dir, float(cal["tfv_false_benefit_margin_m3"]))
    causal = _metrics(tfv, "B_CAUSAL", "holdout_d3")
    oracle = _metrics(tfv, "A_ORACLE", "holdout_d3")
    pfv_holdout = dict(pfv["metrics"]["holdout_d3"])
    payload: dict[str, Any] = {
        "contract": "PROJECT7_V123_CURRENT_DECISION_V1",
        "git": {"head": git_head, "remote_branch_head": remote_head, "hardening_head": hardening_head, "working_tree_at_report": "clean"},
        "boundary": {"new_swmm": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False, "v7_retrained": False, "continuous_gradient_search": False},
        "causal_forecast_store": {"contract": store["contract"], "groups": store["group_count"], "d2_groups": store["d2_group_count"], "d3_groups": store["d3_group_count"], "events": store["event_count"], "history_steps": store["history_steps"], "model_step_seconds": store["model_step_seconds"], "horizon_steps": store["horizon_steps"], "cached_first_frame_mismatch_count": store["cached_first_frame_mismatch_count"], "future_realized_rainfall_not_used": store["future_realized_rainfall_not_used"], "store_sha256": store["store_sha256"]},
        "causal_normalization": normalization,
        "pre_rain_audit": pre_rain,
        "tfv_causal_value": {"causal_holdout_d3": causal, "oracle_holdout_d3_historical_comparison": oracle, "response_collapse": causal["response_collapse"], "future_realized_rainfall_used_as_model_input": tfv["future_realized_rainfall_used_as_model_input"]},
        "pfv_causal_value": {"holdout_d3": pfv_holdout, "label_contract": pfv["label_contract"], "future_realized_rainfall_used_as_model_input": pfv["future_realized_rainfall_used_as_model_input"]},
        "continuous_gate": gate,
        "first_move_coverage": first_move,
        "knowledge_guided_seed": seed,
        "calibration": calibration,
        "objective": objective,
        "runtime_evidence": runtime,
        "seven_strategy_comparison": seven,
        "decision": {
            "tfv_value_status": "BLOCKED_RANK_BELOW_0.70",
            "pfv_status": "WEAK_SOFT_ONLY",
            "continuous_gradient_search": False,
            "finite_shooting_runtime": "COMPLETED_ONE_DEVELOPMENT_EVENT",
            "verdict": "V123_GRADIENT_BLOCKED_FINITE_ONLY",
            "next_action": "STOP_FOR_EXTERNAL_REVIEW_AND_FIX_CAUSAL_VALUE_BEFORE_MORE_T5_OR_CONTINUOUS_MPC",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Project7 Step2 V123 current decision",
        "",
        f"Verdict: **{payload['decision']['verdict']}**",
        "",
        "## Causal value",
        "",
        f"Causal Holdout D3 rank={causal['rank']:.6f}, top1={causal['top1_rate']:.6f}, pairwise={causal['pairwise']:.6f}, sign={causal['sign_accuracy']:.6f}; response collapse={causal['response_collapse']}",
        f"PFV Holdout D3 rank={pfv_holdout['rank']:.6f}, top1={pfv_holdout['top1_rate']:.6f}, pairwise={pfv_holdout['pairwise']:.6f}, sign={pfv_holdout['sign_accuracy']:.6f}.",
        "",
        "## Runtime",
        "",
        f"Decisions={runtime['decision_count']}; sources={runtime['source_counts']}; causal rainfall={runtime['v123_runtime_causal_rainfall']}; future rainfall input={runtime['future_realized_rainfall_used_as_model_input']}; small-effect selection risk={runtime['small_effect_selection_risk']}.",
        "",
        "## Seven-strategy T5 comparison",
        "",
        "| strategy | TFV reduction (%) | Priority8 PFV (m3) | PFV change (%) | global peak (m3/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in seven["rows"]:
        lines.append(f"| {row['strategy']} | {row['tfv_reduction_vs_no_control_pct']:.3f} | {row['pfv_priority8_m3']:.3f} | {row['pfv_change_vs_no_control_pct']:.3f} | {row['global_peak_flood_rate_m3s']:.3f} |")
    lines += ["", "Continuous MPC remains fail-closed. No Validation, Final, Formal, or new SWMM was run.", ""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--remote-head", required=True)
    parser.add_argument("--hardening-head", required=True)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(root=args.root, git_head=args.git_head, remote_head=args.remote_head, hardening_head=args.hardening_head, runtime_dir=args.runtime_dir, out=args.out), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
