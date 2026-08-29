"""Audit the completed Project7 V28 q95-matched residual Benchmark5 output.

This is a read-only result auditor.  It does not invoke SWMM and does not alter any
controller or benchmark artifact.  V28 action/HOLD and candidate statistics are read from
the structured ``v28_*`` diagnostics written by the runtime; no scipy-message parsing is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


V28_TFV_CONTRACT = (
    "SYSTEM_WIDE_CUMULATIVE_TFV_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return float(statistics.mean(values)) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return float(statistics.median(values)) if values else None


def _percent(reduction_m3: float, baseline_m3: float) -> float:
    return float(-100.0 * float(reduction_m3) / float(baseline_m3)) if baseline_m3 else 0.0


def _decision_path(event: dict[str, Any], metadata: dict[str, Any], root: Path) -> Path:
    raw = metadata.get("decision_file") or event.get("proposed_metadata_path", "").replace(
        "__proposed.json", "__proposed.decisions.jsonl"
    )
    path = Path(str(raw))
    if not path.is_file():
        path = root / str(event["event_id"]) / f"{event['event_id']}__proposed.decisions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"decision JSONL not found for {event['event_id']}: {path}")
    return path.resolve()


def _metadata_path(event: dict[str, Any], root: Path) -> Path:
    path = Path(str(event.get("proposed_metadata_path", "")))
    if not path.is_file():
        path = root / str(event["event_id"]) / f"{event['event_id']}__proposed.json"
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found for {event['event_id']}: {path}")
    return path.resolve()


def _load_decisions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"decision row is not an object: {path}")
        rows.append(value)
    return rows


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _selected_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = row.get("diagnostics") or {}
    selected_source = str(diagnostics.get("v28_selected_candidate_source", ""))
    if not selected_source:
        selected_source = str(diagnostics.get("v28_selected_source", ""))
    for candidate in diagnostics.get("v28_candidate_telemetry", []) or []:
        if isinstance(candidate, dict) and str(candidate.get("candidate_source", "")) == selected_source:
            return candidate
    return None


def _event_audit(
    event: dict[str, Any],
    root: Path,
    passive_channel_ids: list[str],
) -> dict[str, Any]:
    metadata_path = _metadata_path(event, root)
    metadata = _read_json(metadata_path)
    decision_path = _decision_path(event, metadata, root)
    decisions = _load_decisions(decision_path)
    action_count = 0
    hold_count = 0
    selected_sources: dict[str, int] = {}
    candidate_sources: dict[str, int] = {}
    contributor_sources: dict[str, int] = {}
    candidate_count = 0
    q95_binding_candidate_count = 0
    q95_binding_selected_count = 0
    duplicate_count = 0
    unique_counts: list[int] = []
    supported_counts: list[int] = []
    raw_counts: list[int] = []
    q95_scales: list[float] = []
    residuals: list[float] = []
    q27_scores: list[float] = []
    q28_scores: list[float] = []
    selected_residuals: list[float] = []
    selected_action_residuals: list[float] = []
    selected_q95_scales: list[float] = []
    selected_action_q95_scales: list[float] = []
    selected_q95_binding_count = 0
    residual_by_source: dict[str, list[float]] = {}
    runtimes: list[float] = []
    max_changed = 0
    max_delta = 0.0
    support_ratios: list[float] = []
    continuity_projection_count = 0
    continuity_numeric_equivalence_count = 0
    passive_diagnostic_values: list[Any] = []
    support_diagnostic_values: list[Any] = []
    action_class_values: list[str] = []
    high_stress_holds: list[dict[str, Any]] = []
    passive_violations: list[dict[str, Any]] = []
    for row in decisions:
        diagnostics = row.get("diagnostics") or {}
        action_class = str(diagnostics.get("v28_action_class", ""))
        if action_class not in {"ACTION", "HOLD"}:
            raise ValueError(f"missing structured V28 action class in {decision_path}")
        action_class_values.append(action_class)
        if action_class == "ACTION":
            action_count += 1
        else:
            hold_count += 1
        selected_source = str(diagnostics.get("v28_selected_source", "HOLD"))
        if action_class == "ACTION":
            selected_sources[selected_source] = selected_sources.get(selected_source, 0) + 1
        candidates = diagnostics.get("v28_candidate_telemetry", []) or []
        if not isinstance(candidates, list):
            raise ValueError(f"invalid V28 candidate telemetry in {decision_path}")
        candidate_count += len(candidates)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"invalid V28 candidate row in {decision_path}")
            source = str(candidate.get("candidate_source", ""))
            candidate_sources[source] = candidate_sources.get(source, 0) + 1
            for contributor in candidate.get("contributing_sources", []) or []:
                text = str(contributor)
                contributor_sources[text] = contributor_sources.get(text, 0) + 1
            if bool(candidate.get("q95_binding", False)):
                q95_binding_candidate_count += 1
            q95_scales.append(_number(candidate.get("q95_scale")))
            residuals.append(_number(candidate.get("q27_residual_m3")))
            residual_by_source.setdefault(source, []).append(_number(candidate.get("q27_residual_m3")))
            q27_scores.append(_number(candidate.get("q27_score_m3")))
            q28_scores.append(_number(candidate.get("q28_score_m3")))
        selected_candidate = _selected_candidate(row)
        if selected_candidate is not None:
            selected_residuals.append(_number(selected_candidate.get("q27_residual_m3")))
            selected_q95_scales.append(_number(selected_candidate.get("q95_scale")))
            if action_class == "ACTION":
                selected_action_residuals.append(_number(selected_candidate.get("q27_residual_m3")))
                selected_action_q95_scales.append(_number(selected_candidate.get("q95_scale")))
            if bool(selected_candidate.get("q95_binding", False)):
                selected_q95_binding_count += 1
        raw_counts.append(int(diagnostics.get("v28_raw_candidate_count", 0)))
        supported_counts.append(int(diagnostics.get("v28_q95_supported_candidate_count", 0)))
        unique_counts.append(int(diagnostics.get("v28_post_q95_unique_candidate_count", 0)))
        duplicate_count += int(diagnostics.get("v28_post_q95_duplicate_count", 0))
        runtime = diagnostics.get("guarded_decision_runtime_seconds")
        if runtime is None:
            runtime = diagnostics.get("decision_runtime_seconds", row.get("elapsed_seconds"))
        runtimes.append(_number(runtime))
        max_changed = max(max_changed, int(diagnostics.get("first_move_changed_facility_count", 0)))
        max_delta = max(max_delta, _number(diagnostics.get("max_setting_delta_per_update_max")))
        support_ratios.append(_number(diagnostics.get("joint_sequence_support_max_ratio")))
        if bool(diagnostics.get("continuity_projection_applied", False)):
            continuity_projection_count += 1
        if bool(diagnostics.get("continuity_numerical_equivalence_applied", False)):
            continuity_numeric_equivalence_count += 1
        passive_diagnostic_values.append(diagnostics.get("passive_no_new_command"))
        support_diagnostic_values.append(diagnostics.get("joint_sequence_support_used"))
        stress = _number(
            (selected_candidate or {}).get("network_stress_q75", diagnostics.get("network_stress_q75"))
        )
        if action_class == "HOLD" and stress >= 0.75:
            high_stress_holds.append(
                {
                    "datetime": row.get("datetime"),
                    "network_stress_q75": stress,
                    "strong_storm_blend": _number(
                        (selected_candidate or {}).get("strong_storm_blend", diagnostics.get("strong_storm_blend"))
                    ),
                    "reason": diagnostics.get("tfv_value_gate_reason", "V28_Q28_NONNEGATIVE_HOLD"),
                    "selected_q28_score_m3": _number(diagnostics.get("v28_selected_q28_score_m3")),
                }
            )
        settings = row.get("settings") or {}
        for channel in passive_channel_ids:
            if channel not in settings:
                passive_violations.append(
                    {"datetime": row.get("datetime"), "channel": channel, "reason": "missing_setting"}
                )
    if decisions and passive_channel_ids:
        reference = decisions[0].get("settings") or {}
        for row in decisions[1:]:
            settings = row.get("settings") or {}
            for channel in passive_channel_ids:
                if channel in reference and channel in settings and abs(
                    _number(settings[channel]) - _number(reference[channel])
                ) > 1.0e-7:
                    passive_violations.append(
                        {
                            "datetime": row.get("datetime"),
                            "channel": channel,
                            "reason": "value_changed",
                            "reference": _number(reference[channel]),
                            "observed": _number(settings[channel]),
                        }
                    )
    comparison = _read_json(root / "OPERATIONAL_BENCHMARK5_COMPARISON.json")
    event_result = next(item for item in comparison["event_results"] if item["event_id"] == event["event_id"])
    target_audit = metadata.get("target_write_readback_audit") or {}
    runtime_lineage = metadata.get("runtime_factory_lineage") or {}
    return {
        "event_id": str(event["event_id"]),
        "metadata_path": str(metadata_path),
        "decision_path": str(decision_path),
        "decisions": len(decisions),
        "action_count": action_count,
        "hold_count": hold_count,
        "action_fraction": float(action_count / len(decisions)) if decisions else 0.0,
        "selected_source_counts": dict(sorted(selected_sources.items())),
        "candidate_source_counts": dict(sorted(candidate_sources.items())),
        "contributing_source_counts": dict(sorted(contributor_sources.items())),
        "raw_candidate_count_total": int(sum(raw_counts)),
        "q95_supported_candidate_count_total": int(sum(supported_counts)),
        "post_q95_unique_candidate_count_total": int(sum(unique_counts)),
        "post_q95_duplicate_count": int(duplicate_count),
        "q95_binding_candidate_count": int(q95_binding_candidate_count),
        "q95_binding_selected_count": int(selected_q95_binding_count),
        "mean_q95_scale_candidates": _mean(q95_scales),
        "mean_q95_scale_selected": _mean(selected_q95_scales),
        "mean_q27_score_m3": _mean(q27_scores),
        "mean_residual_m3": _mean(residuals),
        "mean_selected_residual_m3": _mean(selected_residuals),
        "mean_selected_action_residual_m3": _mean(selected_action_residuals),
        "mean_q28_score_m3": _mean(q28_scores),
        "mean_selected_action_q95_scale": _mean(selected_action_q95_scales),
        "residual_by_source_m3": {
            source: _mean(values) for source, values in sorted(residual_by_source.items())
        },
        "max_support_ratio": max(support_ratios) if support_ratios else None,
        "max_changed_facilities": int(max_changed),
        "max_setting_delta_per_update": float(max_delta),
        "continuity_projection_count": int(continuity_projection_count),
        "continuity_numerical_equivalence_count": int(continuity_numeric_equivalence_count),
        "passive_diagnostic_values": sorted({str(value) for value in passive_diagnostic_values}),
        "support_diagnostic_values": sorted({str(value) for value in support_diagnostic_values}),
        "runtime_seconds": {
            "min": min(runtimes) if runtimes else None,
            "mean": _mean(runtimes),
            "median": _median(runtimes),
            "p95": sorted(runtimes)[max(0, int(0.95 * len(runtimes)) - 1)] if runtimes else None,
            "max": max(runtimes) if runtimes else None,
        },
        "high_stress_hold_count": len(high_stress_holds),
        "high_stress_holds": high_stress_holds,
        "passive_channel_violation_count": len(passive_violations),
        "passive_channel_violations": passive_violations,
        "target_write_readback_audit": target_audit,
        "metadata_lineage_checks": {
            "development_only": metadata.get("development_only"),
            "formal_evidence": metadata.get("formal_evidence"),
            "ready_for_policy_lock": metadata.get("ready_for_policy_lock"),
            "v28_q95_mandatory": metadata.get("v28_q95_mandatory"),
            "v28_raw_action_executable": metadata.get("v28_raw_action_executable"),
            "v28_q27_frozen": metadata.get("v28_q27_frozen"),
            "v28_auto_rbc_shadow_candidate_only": metadata.get("v28_auto_rbc_shadow_candidate_only"),
            "q95_support_execution": metadata.get("q95_support_execution"),
            "event_id_feature": metadata.get("v28_event_id_feature"),
            "runtime_contract": metadata.get("operational_development_runtime_contract"),
            "tfv_estimand": metadata.get("v28_tfv_value_estimand"),
            "parent_lineage": runtime_lineage,
        },
        "benchmark": {
            "proposed_tfv_m3": _number(event_result.get("proposed_tfv_m3")),
            "proposed_pfv_m3": _number(event_result.get("proposed_pfv_m3")),
            "pfv_no_control_m3": _number(event_result.get("pfv_no_control_m3")),
            "pfv_limit_m3": _number(event_result.get("pfv_safety_limit_m3")),
            "pfv_pass": bool(event_result.get("proposed_pfv_safety_pass")),
            "global_peak_flood_rate_m3s": _number(event_result.get("proposed_global_peak_flood_rate_m3s")),
            "flow_routing_error_pct": _number(event_result.get("flow_routing_error_pct")),
            "comparisons": event_result.get("comparisons", {}),
        },
    }


def _aggregate(events: list[dict[str, Any]], comparison: dict[str, Any], baseline: str) -> dict[str, Any]:
    rows = []
    for event in events:
        row = event["benchmark"]
        comp = row["comparisons"][baseline]
        rows.append(
            {
                "event_id": event["event_id"],
                "proposed_tfv_m3": row["proposed_tfv_m3"],
                "baseline_tfv_m3": _number(comp.get("baseline_tfv_m3")),
                "delta_tfv_m3": _number(comp.get("proposed_minus_baseline_tfv_m3")),
                "reduction_pct": _number(comp.get("tfv_reduction_pct")),
                "better": bool(comp.get("proposed_better_tfv")),
            }
        )
    aggregate = comparison.get("aggregate", {}).get(baseline, {})
    return {
        "baseline": baseline,
        "events": rows,
        "event_count": len(rows),
        "events_better": sum(bool(row["better"]) for row in rows),
        "event_balanced_mean_reduction_pct": _mean(row["reduction_pct"] for row in rows),
        "event_balanced_median_reduction_pct": _median(row["reduction_pct"] for row in rows),
        "aggregate_volume_reduction_pct": _number(aggregate.get("aggregate_volume_tfv_reduction_pct")),
        "mean_delta_tfv_m3": _mean(row["delta_tfv_m3"] for row in rows),
        "sum_proposed_tfv_m3": sum(row["proposed_tfv_m3"] for row in rows),
        "sum_baseline_tfv_m3": sum(row["baseline_tfv_m3"] for row in rows),
    }


def _lineage_comparable(v28: dict[str, Any], v27: dict[str, Any], v28_events: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "benchmark_manifest_sha256": (v28.get("benchmark_manifest_sha256"), v27.get("benchmark_manifest_sha256")),
        "baseline_cache_sha256": (v28.get("baseline_cache_sha256"), v27.get("baseline_cache_sha256")),
        "v15_rank_checkpoint_sha256": (v28.get("v15_rank_checkpoint_sha256"), v27.get("v15_rank_checkpoint_sha256")),
        "asset_manifest_sha256": (
            v28_events[0]["metadata_lineage_checks"]["parent_lineage"].get("asset_manifest_sha256"),
            None,
        ),
        "step2_checkpoint_sha256": (
            v28_events[0]["metadata_lineage_checks"]["parent_lineage"].get("base_step2_sha256"),
            None,
        ),
        "v21_boundary_checkpoint_sha256": (
            v28_events[0]["metadata_lineage_checks"]["parent_lineage"].get("v21_boundary_checkpoint_sha256"),
            None,
        ),
    }
    # V27's comparison predates explicit top-level asset/Step2/V21 fields.  Its event metadata
    # is the authoritative comparison source for those three identities.
    return {
        "comparison_fields": fields,
        "benchmark_and_baseline_match": all(
            str(left).lower() == str(right).lower() for left, right in fields.values() if right is not None
        ),
        "v27_metadata_lineage_required": True,
        "note": "V27 metadata must be checked separately for asset/Step2/V21 because its comparison JSON has no top-level fields.",
    }


def _runtime_lineage(metadata: dict[str, Any]) -> dict[str, str | None]:
    factory = metadata.get("runtime_factory_lineage") or {}
    return {
        "asset_manifest_sha256": str(metadata.get("asset_manifest_sha256") or factory.get("asset_manifest_sha256") or "").lower(),
        "base_step2_sha256": str(factory.get("base_step2_sha256") or "").lower(),
        "v15_rank_checkpoint_sha256": str(
            metadata.get("v15_rank_checkpoint_sha256") or factory.get("v15_rank_checkpoint_sha256") or ""
        ).lower(),
        "v21_boundary_checkpoint_sha256": str(
            metadata.get("v21_boundary_checkpoint_sha256") or factory.get("v21_boundary_checkpoint_sha256") or ""
        ).lower(),
    }


def _v27_comparison(
    v27_path: Path,
    v28_comparison: dict[str, Any],
    v28_events: list[dict[str, Any]],
) -> dict[str, Any]:
    v27 = _read_json(v27_path)
    v27_root = v27_path.parent
    v27_events = {str(event["event_id"]): event for event in v27.get("event_results", [])}
    rows: list[dict[str, Any]] = []
    lineage_checks: dict[str, Any] = {}
    for event in v28_events:
        event_id = str(event["event_id"])
        if event_id not in v27_events:
            raise ValueError(f"V27 comparison lacks event {event_id}")
        old_event = v27_events[event_id]
        v27_metadata_path = _metadata_path(old_event, v27_root)
        old_metadata = _read_json(v27_metadata_path)
        current_lineage = _runtime_lineage(
            _read_json(Path(event["metadata_path"]))
        )
        old_lineage = _runtime_lineage(old_metadata)
        pairs = {
            "benchmark_manifest_sha256": (
                str(v28_comparison.get("benchmark_manifest_sha256", "")).lower(),
                str(v27.get("benchmark_manifest_sha256", "")).lower(),
            ),
            "baseline_cache_sha256": (
                str(v28_comparison.get("baseline_cache_sha256", "")).lower(),
                str(v27.get("baseline_cache_sha256", "")).lower(),
            ),
        }
        pairs.update({key: (current_lineage[key], old_lineage[key]) for key in current_lineage})
        lineage_checks[event_id] = {
            "v28": current_lineage,
            "v27": old_lineage,
            "matches": {key: left == right and bool(left) for key, (left, right) in pairs.items()},
            "metadata_paths": {
                "v28": event["metadata_path"],
                "v27": str(v27_metadata_path),
            },
        }
        comparable = all(lineage_checks[event_id]["matches"].values())
        v28_tfv = float(event["benchmark"]["proposed_tfv_m3"])
        v27_tfv = _number(old_event.get("proposed_tfv_m3"))
        rows.append(
            {
                "event_id": event_id,
                "v27_tfv_m3": v27_tfv,
                "v28_tfv_m3": v28_tfv,
                "delta_v28_minus_v27_m3": v28_tfv - v27_tfv,
                "v28_reduction_vs_v27_pct": _percent(v28_tfv - v27_tfv, v27_tfv),
                "v28_better_than_v27": v28_tfv < v27_tfv,
                "lineage_comparable": comparable,
            }
        )
    comparable = all(row["lineage_comparable"] for row in rows)
    deltas = [float(row["delta_v28_minus_v27_m3"]) for row in rows]
    reductions = [float(row["v28_reduction_vs_v27_pct"]) for row in rows]
    return {
        "path": str(v27_path),
        "sha256": _sha(v27_path),
        "lineage_comparable": comparable,
        "lineage_checks": lineage_checks,
        "events": rows,
        "v27_aggregate_tfv_m3": sum(float(row["v27_tfv_m3"]) for row in rows),
        "v28_aggregate_tfv_m3": sum(float(row["v28_tfv_m3"]) for row in rows),
        "aggregate_delta_v28_minus_v27_m3": sum(deltas),
        "aggregate_reduction_vs_v27_pct": _percent(sum(deltas), sum(float(row["v27_tfv_m3"]) for row in rows)),
        "event_balanced_mean_reduction_vs_v27_pct": _mean(reductions),
        "event_balanced_median_reduction_vs_v27_pct": _median(reductions),
        "events_v28_better": sum(bool(row["v28_better_than_v27"]) for row in rows),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Project7 V28 q95-matched residual Benchmark5 audit",
        "",
        f"- Benchmark events: {report['event_count']}",
        f"- Structured decision rows: {report['decision_count']}",
        f"- ACTION/HOLD: {report['action_count']}/{report['hold_count']}",
        f"- Q95 mandatory: `{report['lineage']['q95_mandatory']}`; raw executable: `{report['lineage']['raw_action_executable']}`",
        f"- V28 development-only: `{report['lineage']['development_only']}`; formal evidence: `{report['lineage']['formal_evidence']}`",
        "",
        "## Event results",
        "",
        "| Event | V28 TFV | Auto-RBC TFV | ΔTFV | reduction % | V28 win | PFV | PFV limit | PFV pass | ACTION | HOLD | max runtime s |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for event in report["events"]:
        auto = event["benchmark"]["comparisons"]["auto_rbc"]
        lines.append(
            f"| {event['event_id']} | {event['benchmark']['proposed_tfv_m3']:.2f} | "
            f"{_number(auto.get('baseline_tfv_m3')):.2f} | {_number(auto.get('proposed_minus_baseline_tfv_m3')):.2f} | "
            f"{_number(auto.get('tfv_reduction_pct')):.4f} | {bool(auto.get('proposed_better_tfv'))} | "
            f"{event['benchmark']['proposed_pfv_m3']:.2f} | {event['benchmark']['pfv_limit_m3']:.2f} | "
            f"{event['benchmark']['pfv_pass']} | {event['action_count']} | {event['hold_count']} | "
            f"{event['runtime_seconds']['max']:.4f} |"
        )
    lines.extend(["", "## Aggregate comparisons", ""])
    for baseline, aggregate in report["aggregates"].items():
        lines.append(
            f"- **{baseline}**: {aggregate['events_better']}/{aggregate['event_count']} wins; "
            f"event-balanced mean {aggregate['event_balanced_mean_reduction_pct']:.4f}%; "
            f"median {aggregate['event_balanced_median_reduction_pct']:.4f}%; "
            f"aggregate reduction {aggregate['aggregate_volume_reduction_pct']:.4f}%; "
            f"mean ΔTFV {aggregate['mean_delta_tfv_m3']:.2f} m3."
        )
    lines.extend(
        [
            "",
            "## Engineering and mechanism",
            "",
            f"- Q95 binding candidates: {report['q95_binding_candidate_count']}; selected Q95-binding decisions: {report['q95_binding_selected_count']}",
            f"- Post-Q95 duplicate candidates: {report['post_q95_duplicate_count']}",
            f"- Max changed facilities: {report['max_changed_facilities']}; max setting delta: {report['max_setting_delta_per_update']:.6f}",
            f"- Continuity projections: {report['continuity_projection_count']}; numerical-equivalence projections: {report['continuity_numerical_equivalence_count']}",
            f"- Passive-channel violations: {report['passive_channel_violation_count']}; q95 support violations: {report['support_violation_count']}",
            f"- High-stress HOLD rows: {report['high_stress_hold_count']}",
            f"- Readback passed: `{report['readback_all_passed']}`; PFV passed: `{report['pfv_all_passed']}`; engineering pass: `{report['engineering_pass']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="V28 Benchmark5 output directory")
    parser.add_argument("--v27-comparison", help="Comparable V27 comparison JSON")
    parser.add_argument("--supervisory-control", help="Supervisory control JSON used to audit passive channels")
    parser.add_argument("--out", required=True, help="V28 audit JSON output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    comparison_path = root / "OPERATIONAL_BENCHMARK5_COMPARISON.json"
    comparison = _read_json(comparison_path)
    event_results = comparison.get("event_results")
    if not isinstance(event_results, list) or len(event_results) != 5:
        raise ValueError("V28 audit requires exactly five benchmark event results")
    passive_channel_ids: list[str] = []
    if args.supervisory_control:
        control = _read_json(Path(args.supervisory_control).resolve())
        passive_channel_ids = [str(value) for value in control.get("passive_setting_channel_ids", [])]
    events = [_event_audit(event, root, passive_channel_ids) for event in event_results]
    all_decisions = sum(int(event["decisions"]) for event in events)
    source_counts: dict[str, int] = {}
    selected_source_counts: dict[str, int] = {}
    residual_source_values: dict[str, list[float]] = {}
    for event in events:
        for source, count in event["candidate_source_counts"].items():
            source_counts[source] = source_counts.get(source, 0) + int(count)
        for source, count in event["selected_source_counts"].items():
            selected_source_counts[source] = selected_source_counts.get(source, 0) + int(count)
        for source, value in event["residual_by_source_m3"].items():
            residual_source_values.setdefault(source, []).append(float(value))
    lineage = events[0]["metadata_lineage_checks"]
    lineage_consistent = all(event["metadata_lineage_checks"] == lineage for event in events)
    aggregates = {
        baseline: _aggregate(events, comparison, baseline)
        for baseline in ("no_control", "internal_rtc", "auto_rbc", "efd")
    }
    v27_comparison = None
    if args.v27_comparison:
        v27_path = Path(args.v27_comparison).resolve()
        v27_comparison = _read_json(v27_path)
    report: dict[str, Any] = {
        "contract": "PROJECT7_V28_Q95_MATCHED_RESIDUAL_BENCHMARK5_RESULT_AUDIT_V1",
        "root": str(root),
        "comparison_path": str(comparison_path),
        "comparison_sha256": _sha(comparison_path),
        "event_count": len(events),
        "decision_count": all_decisions,
        "action_count": sum(int(event["action_count"]) for event in events),
        "hold_count": sum(int(event["hold_count"]) for event in events),
        "action_fraction": sum(int(event["action_count"]) for event in events) / all_decisions,
        "events": events,
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "q95_binding_candidate_count": sum(int(event["q95_binding_candidate_count"]) for event in events),
        "q95_binding_selected_count": sum(int(event["q95_binding_selected_count"]) for event in events),
        "post_q95_duplicate_count": sum(int(event["post_q95_duplicate_count"]) for event in events),
        "mean_q95_scale": _mean(
            event["mean_q95_scale_selected"] for event in events if event["mean_q95_scale_selected"] is not None
        ),
        "mean_residual_m3": _mean(
            event["mean_selected_residual_m3"] for event in events if event["mean_selected_residual_m3"] is not None
        ),
        "mean_selected_action_residual_m3": _mean(
            event["mean_selected_action_residual_m3"]
            for event in events
            if event["mean_selected_action_residual_m3"] is not None
        ),
        "mean_selected_action_q95_scale": _mean(
            event["mean_selected_action_q95_scale"]
            for event in events
            if event["mean_selected_action_q95_scale"] is not None
        ),
        "residual_by_source_m3": {
            source: _mean(values) for source, values in sorted(residual_source_values.items())
        },
        "max_changed_facilities": max(int(event["max_changed_facilities"]) for event in events),
        "max_setting_delta_per_update": max(
            float(event["max_setting_delta_per_update"]) for event in events
        ),
        "max_support_ratio": max(
            float(event["max_support_ratio"])
            for event in events
            if event["max_support_ratio"] is not None
        ),
        "continuity_numerical_equivalence_count": sum(
            int(event["continuity_numerical_equivalence_count"]) for event in events
        ),
        "lineage": {
            "contract": lineage["runtime_contract"],
            "q95_mandatory": lineage["v28_q95_mandatory"],
            "raw_action_executable": lineage["v28_raw_action_executable"],
            "q27_frozen": lineage["v28_q27_frozen"],
            "auto_rbc_shadow_candidate_only": lineage["v28_auto_rbc_shadow_candidate_only"],
            "event_id_feature": lineage["event_id_feature"],
            "development_only": lineage["development_only"],
            "formal_evidence": lineage["formal_evidence"],
            "ready_for_policy_lock": lineage["ready_for_policy_lock"],
            "tfv_value_estimand": lineage["tfv_estimand"],
            "metadata_lineage_consistent": lineage_consistent,
        },
        "aggregates": aggregates,
        "readback_all_passed": all(bool(event["target_write_readback_audit"].get("passed")) for event in events),
        "pfv_all_passed": all(bool(event["benchmark"]["pfv_pass"]) for event in events),
        "routing_all_zero": all(float(event["benchmark"]["flow_routing_error_pct"]) == 0.0 for event in events),
        "runtime_max_seconds": max(event["runtime_seconds"]["max"] for event in events),
        "runtime_under_600_seconds": all(float(event["runtime_seconds"]["max"]) < 600.0 for event in events),
        "continuity_projection_count": sum(int(event["continuity_projection_count"]) for event in events),
        "passive_diagnostic_values": sorted({value for event in events for value in event["passive_diagnostic_values"]}),
        "support_diagnostic_values": sorted({value for event in events for value in event["support_diagnostic_values"]}),
        "high_stress_hold_count": sum(int(event["high_stress_hold_count"]) for event in events),
        "passive_channel_violation_count": sum(
            int(event["passive_channel_violation_count"]) for event in events
        ),
        "support_violation_count": sum(
            int(event["max_support_ratio"] > 1.0 + 1.0e-6)
            or int("True" not in event["support_diagnostic_values"])
            for event in events
        ),
        "v27_comparison": (
            _v27_comparison(Path(args.v27_comparison).resolve(), comparison, events)
            if v27_comparison is not None
            else {"provided": False}
        ),
    }
    report["engineering_pass"] = bool(
        report["readback_all_passed"]
        and report["pfv_all_passed"]
        and report["routing_all_zero"]
        and report["runtime_under_600_seconds"]
        and report["passive_channel_violation_count"] == 0
        and report["support_violation_count"] == 0
        and report["max_setting_delta_per_update"] <= 0.5 + 1.0e-7
        and report["continuity_projection_count"] == 0
        and report["lineage"]["metadata_lineage_consistent"]
        and report["lineage"]["q95_mandatory"]
        and report["lineage"]["raw_action_executable"] is False
        and report["lineage"]["auto_rbc_shadow_candidate_only"]
        and report["lineage"]["event_id_feature"] is False
    )
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "v28_tfv_m3",
                "auto_rbc_tfv_m3",
                "delta_tfv_m3",
                "reduction_pct",
                "v28_better_than_auto_rbc",
                "pfv_v28_m3",
                "pfv_limit_m3",
                "pfv_pass",
                "action_count",
                "hold_count",
                "action_fraction",
                "selected_source_counts",
                "q95_binding_selected_count",
                "mean_selected_residual_m3",
                "runtime_max_seconds",
                "engineering_pass",
            ],
        )
        writer.writeheader()
        for event in events:
            auto = event["benchmark"]["comparisons"]["auto_rbc"]
            writer.writerow(
                {
                    "event_id": event["event_id"],
                    "v28_tfv_m3": event["benchmark"]["proposed_tfv_m3"],
                    "auto_rbc_tfv_m3": auto.get("baseline_tfv_m3"),
                    "delta_tfv_m3": auto.get("proposed_minus_baseline_tfv_m3"),
                    "reduction_pct": auto.get("tfv_reduction_pct"),
                    "v28_better_than_auto_rbc": auto.get("proposed_better_tfv"),
                    "pfv_v28_m3": event["benchmark"]["proposed_pfv_m3"],
                    "pfv_limit_m3": event["benchmark"]["pfv_limit_m3"],
                    "pfv_pass": event["benchmark"]["pfv_pass"],
                    "action_count": event["action_count"],
                    "hold_count": event["hold_count"],
                    "action_fraction": event["action_fraction"],
                    "selected_source_counts": json.dumps(event["selected_source_counts"], sort_keys=True),
                    "q95_binding_selected_count": event["q95_binding_selected_count"],
                    "mean_selected_residual_m3": event["mean_selected_residual_m3"],
                    "runtime_max_seconds": event["runtime_seconds"]["max"],
                    "engineering_pass": report["engineering_pass"],
                }
            )
    _write_markdown(out_path.with_suffix(".md"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
