"""Aggregate the bounded V124 knowledge-guided development evidence.

This report intentionally keeps the failed V124 Value diagnostic separate from
the frozen V70 runtime used for the three-arm T5 comparison.  It never reads
Validation/Final/Formal assets and never recomputes a metric from SWMM rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _decision_summary(runtime_dir: Path) -> dict[str, Any]:
    metadata_path = next(runtime_dir.glob("*.json"))
    metadata = _load(metadata_path)
    decision_path = runtime_dir / str(metadata["decision_file"])
    records = [
        json.loads(line)
        for line in decision_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    diagnostics = [dict(record.get("diagnostics", {})) for record in records]
    source_counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("source", "UNKNOWN"))
        source_counts[source] = source_counts.get(source, 0) + 1
    values = {
        key: [float(item[key]) for item in diagnostics if isinstance(item.get(key), (int, float))]
        for key in (
            "predicted_delta_tfv_m3",
            "predicted_delta_pfv_m3",
            "decision_runtime_seconds",
            "target_change_max",
            "command_delta_from_current_max",
            "command_delta_from_previous_target_max",
            "planned_sequence_max_step_delta",
        )
    }
    def summary(values_: list[float]) -> dict[str, float | None]:
        if not values_:
            return {"min": None, "mean": None, "max": None}
        return {
            "min": min(values_),
            "mean": sum(values_) / len(values_),
            "max": max(values_),
        }
    anchor_selected = sum(bool(item.get("knowledge_anchor_selected", False)) for item in diagnostics)
    anchor_fallback = sum(bool(item.get("knowledge_anchor_fallback_used", False)) for item in diagnostics)
    passive = sum(bool(item.get("passive_no_new_command", False)) for item in diagnostics)
    learned_override = sum(
        str(item.get("v123_policy_mode", "")) == "hybrid"
        and not bool(item.get("knowledge_anchor_selected", False))
        and str(record.get("source", "")) == "MPC_V123"
        for record, item in zip(records, diagnostics)
    )
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha(metadata_path),
        "decision_path": str(decision_path),
        "decision_count": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "policy_modes": sorted({str(item.get("v123_policy_mode", "")) for item in diagnostics}),
        "knowledge_anchor_selected_count": anchor_selected,
        "knowledge_anchor_fallback_used_count": anchor_fallback,
        "learned_override_count": learned_override,
        "passive_decision_count": passive,
        "active_decision_count": len(records) - passive,
        "score_equals_execute_all": all(bool(item.get("score_equals_execute", False)) for item in diagnostics),
        "continuity_guard_all_passed": all(bool(item.get("continuity_guard_passed", False)) for item in diagnostics),
        "candidate_count_min": min((int(item.get("candidate_count", 0)) for item in diagnostics), default=0),
        "candidate_count_max": max((int(item.get("candidate_count", 0)) for item in diagnostics), default=0),
        "tail_only_noop_count_max": max((int(item.get("tail_only_noop_candidate_count", 0)) for item in diagnostics), default=0),
        "metrics": {key: summary(value) for key, value in values.items()},
        "future_realized_rainfall_used_as_model_input": bool(
            metadata.get("future_realized_rainfall_used_as_model_input", True)
        ),
        "knowledge_data_fusion": bool(metadata.get("knowledge_data_fusion", False)),
        "runtime_causal_rainfall": bool(metadata.get("v123_runtime_causal_rainfall", False)),
        "score_only_executable_sequences": bool(metadata.get("score_only_executable_sequences", False)),
    }


def _comparison_row(path: Path, strategy: str) -> dict[str, Any]:
    report = _load(path)
    for row in report["rows"]:
        if row["strategy"] == strategy:
            return {
                key: row[key]
                for key in (
                    "strategy",
                    "tfv_m3",
                    "tfv_reduction_vs_no_control_pct",
                    "pfv_priority8_m3",
                    "pfv_change_vs_no_control_pct",
                    "global_peak_flood_rate_m3s",
                    "decisions",
                    "flow_routing_error_pct",
                )
            }
    raise KeyError(f"{strategy!r} missing from {path}")


def build_report(
    *,
    root: Path,
    parity: Path,
    value: Path,
    pfv: Path,
    first_move: Path,
    anchor_comparison: Path,
    learned_comparison: Path,
    hybrid_comparison: Path,
    anchor_runtime: Path,
    learned_runtime: Path,
    hybrid_runtime: Path,
    git_head: str,
    out: Path,
) -> dict[str, Any]:
    parity_report = _load(parity)
    value_report = _load(value)
    pfv_report = _load(pfv)
    first_move_report = _load(first_move)
    old_tfv = _load(root / "tfv_causal_norm" / "STEP2_V123_TFV_CAUSAL_ABLATION.json")
    calibration = _load(root / "STEP2_V123_ADMISSION_CALIBRATION.json")
    objective = _load(root / "STEP2_V123_OBJECTIVE_EVIDENCE.json")
    baseline_rows = _load(anchor_comparison)["rows"][:6]
    arms = {
        "anchor_only": {
            "runtime": _decision_summary(anchor_runtime),
            "authoritative": _comparison_row(anchor_comparison, "proposed_v123"),
        },
        "learned_only_v70": {
            "runtime": _decision_summary(learned_runtime),
            "authoritative": _comparison_row(learned_comparison, "proposed_v123"),
        },
        "hybrid_v70": {
            "runtime": _decision_summary(hybrid_runtime),
            "authoritative": _comparison_row(hybrid_comparison, "proposed_v123"),
        },
    }
    payload: dict[str, Any] = {
        "contract": "PROJECT7_V124_KNOWLEDGE_GUIDED_FIX_LOOP_V1",
        "git": {"report_git_head": git_head},
        "boundary": {
            "new_d2": False,
            "new_d3": False,
            "new_swmm_training_data": False,
            "t5_development_swmm_allowed": True,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "policy_lock": False,
            "continuous_mpc": False,
            "v7_value_retrained": False,
        },
        "sparse_rbc_parity": {
            "matched_groups": parity_report["dataset"]["matched_groups"],
            "fit_d2_groups": parity_report["split"]["fit_d2_groups"],
            "endpoint_depth": parity_report["endpoint_depth_reconstruction"],
            "overall": parity_report["actuator_target_parity"]["overall"],
            "by_type": parity_report["actuator_target_parity"]["by_type"],
            "by_phase": parity_report["actuator_target_parity"]["by_phase"],
            "contract": parity_report["rbc_contract"],
            "true_state_offline_only": parity_report["interpretation"]["true_rbc_offline_only"],
            "sparse_uses_true_state": parity_report["interpretation"]["sparse_rbc_uses_true_state"],
        },
        "v70_causal_baseline": old_tfv["arms"]["B_CAUSAL"]["metrics"]["holdout_d3"],
        "v124_causal_value": {
            "architecture": value_report["architecture"],
            "metrics": value_report["metrics"],
            "accepted_as_finite_value": False,
            "acceptance_reason": "Holdout D3 rank and pairwise did not improve over frozen V70; response did not collapse.",
            "checkpoint": value_report["checkpoint"],
            "checkpoint_sha256": _sha(Path(value_report["checkpoint"])),
        },
        "pfv_causal_value": {
            "holdout_d3": pfv_report["metrics"]["holdout_d3"],
            "trainfit_d3": pfv_report["metrics"]["trainfit_d3"],
            "label_contract": pfv_report["label_contract"],
        },
        "first_move_coverage": {
            "raw_candidate_count": first_move_report["raw_candidate_count"],
            "unique_first_move_after_projection": first_move_report["unique_first_move_count_after_projection"],
            "tail_only_or_passive_like_after_projection": first_move_report["tail_only_or_passive_like_count_after_projection"],
            "control_groups": first_move_report["control_groups_covered_by_nonhold_first_move"],
            "control_groups_total": first_move_report["control_groups_total"],
            "actuators": first_move_report["actuators_covered_by_nonhold_first_move"],
            "actuators_total": first_move_report["actuators_total"],
            "directions": first_move_report["direction_counts"],
            "families": first_move_report["first_move_family_counts"],
            "temporal_basis_zero_active_nonhold": first_move_report["temporal_basis_zero_active_nonhold"],
        },
        "calibration": {
            "existing_trainfit_margin": calibration["calibration"],
            "source_contract": calibration["contract"],
            "cross_fitted_oof_executed": False,
            "cross_fitted_oof_required_before_future_admission": True,
        },
        "objective": objective["objective"],
        "t5_baselines": [
            {key: row[key] for key in ("strategy", "tfv_m3", "tfv_reduction_vs_no_control_pct", "pfv_priority8_m3", "pfv_change_vs_no_control_pct", "global_peak_flood_rate_m3s", "decisions", "flow_routing_error_pct")}
            for row in baseline_rows
        ],
        "t5_arms": arms,
        "d4": {
            "generated": False,
            "authoritative_branches": 0,
            "next_stage_authorized": True,
            "reason": "V124 Holdout D3 rank remains in the historical 0.4 range; freeze the model and design a TrainFit-only knowledge-carrying D4 before generating SWMM branches.",
        },
        "decision": {
            "step1_endpoint_state_blocked": False,
            "sparse_rbc_anchor_verified": True,
            "v124_value_accepted": False,
            "continuous_mpc": False,
            "verdict": "D4_ACTION_SUPPORT_REQUIRED",
            "next_action": "Freeze a deterministic TrainFit-only D4 knowledge-carrying branch plan; do not generate branches or run D3 until that plan is reviewed.",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# PROJECT7 V124 KNOWLEDGE-GUIDED FIX LOOP",
        "",
        f"Verdict: **{payload['decision']['verdict']}**",
        "",
        "## Sparse-RBC parity",
        "",
        f"Endpoint depth reconstruction: NSE={payload['sparse_rbc_parity']['endpoint_depth']['nse']:.6f}, RMSE={payload['sparse_rbc_parity']['endpoint_depth']['rmse_m']:.6f} m, bias={payload['sparse_rbc_parity']['endpoint_depth']['bias_m']:.6f} m.",
        f"Action target parity: MAE={payload['sparse_rbc_parity']['overall']['mae']:.6f}, direction={payload['sparse_rbc_parity']['overall']['direction_accuracy']:.6f}, active overlap={payload['sparse_rbc_parity']['overall']['active_overlap']:.6f}, Pearson={payload['sparse_rbc_parity']['overall']['pearson']:.6f}.",
        "",
        "## Value",
        "",
        f"Frozen V70 causal Holdout D3: rank={payload['v70_causal_baseline']['rank']:.6f}, pairwise={payload['v70_causal_baseline']['pairwise']:.6f}, top1={payload['v70_causal_baseline']['top1_rate']:.6f}.",
        f"V124 causal Holdout D3: rank={payload['v124_causal_value']['metrics']['holdout_d3']['rank']:.6f}, pairwise={payload['v124_causal_value']['metrics']['holdout_d3']['pairwise']:.6f}, top1={payload['v124_causal_value']['metrics']['holdout_d3']['top1_rate']:.6f}; not accepted.",
        f"PFV causal Holdout D3: rank={payload['pfv_causal_value']['holdout_d3']['rank']:.6f}, pairwise={payload['pfv_causal_value']['holdout_d3']['pairwise']:.6f}, top1={payload['pfv_causal_value']['holdout_d3']['top1_rate']:.6f}.",
        "",
        "## T5 development arms",
        "",
        "| arm | TFV reduction (%) | PFV change (%) | peak (m3/s) | anchor selected | learned overrides | passive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in payload["t5_arms"].items():
        runtime = arm["runtime"]
        row = arm["authoritative"]
        lines.append(
            f"| {name} | {row['tfv_reduction_vs_no_control_pct']:.3f} | {row['pfv_change_vs_no_control_pct']:.3f} | {row['global_peak_flood_rate_m3s']:.3f} | {runtime['knowledge_anchor_selected_count']} | {runtime['learned_override_count']} | {runtime['passive_decision_count']} |"
        )
    lines += [
        "",
        "No Validation, Final, Formal, Policy Lock, new D2/D3, or continuous MPC was run. D4 branch generation is not started.",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--pfv", type=Path, required=True)
    parser.add_argument("--first-move", type=Path, required=True)
    parser.add_argument("--anchor-comparison", type=Path, required=True)
    parser.add_argument("--learned-comparison", type=Path, required=True)
    parser.add_argument("--hybrid-comparison", type=Path, required=True)
    parser.add_argument("--anchor-runtime", type=Path, required=True)
    parser.add_argument("--learned-runtime", type=Path, required=True)
    parser.add_argument("--hybrid-runtime", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(**vars(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
