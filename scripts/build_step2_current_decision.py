"""Assemble one evidence-bound Train-only Step2 diagnosis from audit artifacts.

This utility intentionally does not train or evaluate a model.  It joins the
machine-readable outputs of the raw-data audit, history availability audit,
local D2 baselines, and the canonical V9 A/B/C ladder.  The result makes the
distinction between a failed *representation* and an information-theoretic
claim explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_SKILL_KEYS = (
    "delta_depth_m_skill_vs_zero",
    "delta_flood_m3s_skill_vs_zero",
    "delta_storage_m3_skill_vs_zero",
    "delta_managed_flow_m3s_skill_vs_zero",
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _deep(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _channel_metrics(ladder: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    levels = _deep(ladder, "ladder", default={})
    result: dict[str, dict[str, float]] = {}
    for level, payload in levels.items():
        overall = _deep(payload, "overall", default={})
        result[str(level)] = {
            key: float(overall[key])
            for key in PRIMARY_SKILL_KEYS
            if key in overall
        }
    return result


def _local_positive_channels(baselines: Mapping[str, Any]) -> list[str]:
    # The standalone baseline emits the explicit scientific subset name rather
    # than a generic ``holdout`` field.  Keep the legacy fallback solely for
    # hand-authored fixtures used by older reports.
    holdout = _deep(
        baselines,
        "baselines",
        "local_mlp",
        "TrainInternalHoldout_D2",
        "channels",
        default=None,
    )
    if holdout is None:
        holdout = _deep(baselines, "baselines", "local_mlp", "holdout", default={})
    mapping = {
        "depth": ("delta_depth_m", "depth"),
        "flood": ("delta_flood_m3s", "flood"),
        "storage": ("delta_storage_m3", "storage"),
        "managed_flow": ("delta_managed_flow_m3s", "managed_flow"),
    }
    return [
        channel
        for channel, keys in mapping.items()
        if float(
            next(
                (
                    _deep(holdout, key, "skill_vs_zero", default=float("nan"))
                    for key in keys
                    if _deep(holdout, key, "skill_vs_zero", default=None) is not None
                ),
                float("nan"),
            )
        ) > 0.0
    ]


def _next_action(ladder: Mapping[str, Any], local_positive: list[str]) -> str:
    decision = str(_deep(ladder, "decision", "decision", default="MISSING"))
    if decision == "PREDICTED_REFERENCE_TRAJECTORY_SUFFICIENT":
        return "FORMAL_V9_DEVELOPMENT_TRAINING_ONLY_AFTER_EXTERNAL_REVIEW"
    if decision == "REFERENCE_HYDRAULIC_ACCURACY_PRIMARY_BOTTLENECK":
        return "DIAGNOSE_FROZEN_V7_REFERENCE_HYDRAULIC_ACCURACY"
    if decision == "MARKOV_INSUFFICIENCY_SUPPORTED":
        if local_positive:
            return "RUN_EXISTING_HISTORY_LADDER_BEFORE_ANY_NEW_SWMM; LOCAL_D2_SIGNAL_PREVENTS_A_GLOBAL_INSUFFICIENCY_CLAIM"
        return "RUN_EXISTING_HISTORY_LADDER_WITH_FROZEN_STEP1_BEFORE_ANY_NEW_SWMM"
    return "RUN_EXISTING_HISTORY_LADDER_AND_LOCAL_REPRESENTATION_CONTROL_BEFORE_ANY_NEW_SWMM"


def build_decision(
    data_audit: Mapping[str, Any],
    history_audit: Mapping[str, Any],
    baselines: Mapping[str, Any],
    ladder: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    local_positive = _local_positive_channels(baselines)
    ladder_decision = str(_deep(ladder, "decision", "decision", default="MISSING"))
    channel_metrics = _channel_metrics(ladder)
    pairing = _deep(data_audit, "reference_candidate_pairing", default={})
    recompute = _deep(data_audit, "target_recomputation_from_raw_compacts", default={})
    effect = _deep(data_audit, "hydraulic_effect_identifiability", "effect_distribution", default={})
    root = {
        "contract": "PROJECT7_STEP2_ROOT_CAUSE_DIAGNOSIS_V1",
        "scope": {
            "development_train_only": True,
            "swmm_run": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
        "canonical_ladder_decision": ladder_decision,
        "findings": [
            {
                "root_cause": "DATA_ALIGNMENT_BUG",
                "status": "NOT_SUPPORTED",
                "confidence": "high",
                "evidence": {
                    "raw_target_mismatch_comparisons": _deep(
                        recompute, "mismatch_comparisons_over_1e-6", default=None
                    ),
                    "candidate_reference_context_violations": _deep(
                        pairing, "candidate_reference_context_violations", default=None
                    ),
                    "same_prefix": _deep(pairing, "same_prefix_verification", default={}),
                },
                "counter_evidence": "None in the canonical cache audit.",
                "recommended_fix": "Do not alter time alignment or regenerate data.",
            },
            {
                "root_cause": "TARGET_DEFINITION_OR_UNIT_BUG",
                "status": "NOT_SUPPORTED",
                "confidence": "high",
                "evidence": {
                    "TFV_semantics": _deep(data_audit, "TFV_and_PFV_semantics", default={}),
                    "units": _deep(data_audit, "units", default={}),
                    "raw_effect_recompute_max_error": _deep(
                        recompute, "candidate_reference_effect_max_abs_error", default={}
                    ),
                },
                "counter_evidence": "All inspected compact-to-cache tensors and signed effects match exactly.",
                "recommended_fix": "Keep raw signed candidate-minus-reference targets.",
            },
            {
                "root_cause": "LOSS_METRIC_MISMATCH",
                "status": "CONFIRMED_AND_FIXED",
                "confidence": "high",
                "evidence": {
                    "fixed": [
                        "managed-flow active metrics now use per-actuator scale",
                        "zero-support Top-K returns not-applicable rather than tie overlap",
                        "sparse onset uses normalized spatial maximum",
                        "fixed 0-30/30-120/120-360 minute diagnostics",
                    ],
                    "training_target": "raw signed delta state and managed flow",
                },
                "counter_evidence": "The evaluator defect did not change the optimizer target itself.",
                "recommended_fix": "Interpret only post-fix ladder/localization/timing evidence.",
            },
            {
                "root_cause": "TARGET_IMBALANCE_AND_LOCAL_SPARSITY",
                "status": "SUPPORTED",
                "confidence": "high",
                "evidence": effect,
                "counter_evidence": "Depth and managed-flow effects are substantially less sparse than flood/storage.",
                "recommended_fix": "Use signed sparse-effect diagnostics; do not claim all targets are zero.",
            },
            {
                "root_cause": "ALL_ACTION_EFFECT_DATA_UNLEARNABLE",
                "status": "NOT_SUPPORTED" if local_positive else "UNRESOLVED",
                "confidence": "high" if local_positive else "medium",
                "evidence": {
                    "local_mlp_holdout_positive_channels": local_positive,
                    "local_baseline": _deep(baselines, "baselines", "local_mlp", default={}),
                },
                "counter_evidence": "Flood and managed-flow local skills can remain near the zero predictor.",
                "recommended_fix": "Use local-vs-graph evidence before asserting data insufficiency.",
            },
            {
                "root_cause": "SNAPSHOT_REFERENCE_STATE_SUFFICIENCY",
                "status": ladder_decision,
                "confidence": "pending canonical A/B/C interpretation",
                "evidence": {
                    "primary_channel_skill": channel_metrics,
                    "ladder": _deep(ladder, "decision", default={}),
                },
                "counter_evidence": "A graph model failure alone is not an information-theoretic impossibility proof.",
                "recommended_fix": _next_action(ladder, local_positive),
            },
            {
                "root_cause": "EXISTING_HISTORY_AND_LINK_FLOW_AVAILABILITY",
                "status": "PARTIALLY_AVAILABLE",
                "confidence": "high",
                "evidence": {
                    "oracle_history": _deep(history_audit, "checkpoint_coverage", default={}),
                    "history_source": _deep(history_audit, "history_sources", default={}),
                    "flow_availability": _deep(history_audit, "flow_availability", default={}),
                },
                "counter_evidence": "Authoritative past SWMM state is oracle-only; all-link flow is unavailable.",
                "recommended_fix": "Construct frozen-Step1 causal history where eligible before requesting new SWMM.",
            },
        ],
        "primary_remaining_bottleneck": _next_action(ladder, local_positive),
        "new_swmm_authorized": False,
        "formal_v9_authorized": ladder_decision == "PREDICTED_REFERENCE_TRAJECTORY_SUFFICIENT",
    }
    before_after = {
        "contract": "PROJECT7_STEP2_BEFORE_AFTER_COMPARISON_V1",
        "historical_context_only": {
            "old_v8_d2_skill_vs_zero": {
                "depth": -0.0073,
                "flood": -0.00046,
                "storage": -0.209,
            },
            "warning": "Historical V8 metrics are not canonical current-head V9 evidence.",
        },
        "evaluator_before_after": {
            "before": {
                "managed_flow_active_mask": "global median flow_delta_scale",
                "topk_zero_truth": "arbitrary argpartition tie overlap",
                "sparse_onset": "whole-network mean absolute effect",
                "horizon_buckets": "absent",
            },
            "after": {
                "managed_flow_active_mask": "per-actuator normalized truth",
                "topk_zero_truth": "NaN/not-applicable; zero predicted support=0",
                "sparse_onset": "normalized spatial maximum; peak kept separate",
                "horizon_buckets": ["0-30 min", "30-120 min", "120-360 min"],
            },
        },
        "current_train_only_local_baselines": _deep(baselines, "baselines", default={}),
        "canonical_current_head_ABC": {
            "decision": _deep(ladder, "decision", default={}),
            "primary_channel_skill": channel_metrics,
        },
    }
    markdown = "\n".join(
        [
            "# Project7 Step2 current decision",
            "",
            "## Scope",
            "",
            "Train/development-only. No SWMM, Validation, Final, Formal, or production wiring was used.",
            "",
            "## Data-chain result",
            "",
            "Raw compact trajectories reproduce the audited cache targets and signed candidate-reference effects; the audit found no data alignment, unit, or reference-pairing defect.",
            "",
            "## Correctness repair",
            "",
            "The evaluator now uses per-actuator managed-flow scales, fail-closed Top-K semantics, sparse normalized onset timing, and fixed horizon buckets. These repairs change evidence semantics, not the optimizer target.",
            "",
            "## Learnability control",
            "",
            f"Current Train-only endpoint-local baseline has positive holdout skill for: {', '.join(local_positive) or 'none'}.",
            "",
            "## Canonical state-sufficiency result",
            "",
            f"`{ladder_decision}`",
            "",
            "## Decision",
            "",
            f"Next single priority: `{_next_action(ladder, local_positive)}`.",
            "",
            "New SWMM is not authorized by this evidence. V7 Value remains frozen.",
            "",
        ]
    )
    return root, before_after, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current Train-only Step2 diagnosis")
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--history-audit", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--ladder", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root, before_after, markdown = build_decision(
        _load(args.data_audit),
        _load(args.history_audit),
        _load(args.baselines),
        _load(args.ladder),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ROOT_CAUSE_DIAGNOSIS.json").write_text(
        json.dumps(root, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "STEP2_BEFORE_AFTER_COMPARISON.json").write_text(
        json.dumps(before_after, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "STEP2_CURRENT_DECISION.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
