"""Summarize the precommitted seen-event policy-return mechanism panel.

This is Development-only diagnostics. It aggregates exact same-prefix query records without changing
the controller, training roles, conformal margin or untouched evaluation sets. Current records must
share one 82-control/109-channel supervisory-mask lineage.

An all-harmful query means HOLD is oracle-optimal *within the generated portfolio at that exact state*;
it is not proof that the frozen Step2 representation or every feasible action is globally useless.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from rtc.direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
from rtc.direct_tfv_policy_return_portfolio_admission import validate_policy_return_portfolio_record


MECHANISM_PANEL_CONTRACT = "PROJECT7_POLICY_RETURN_SEEN_MECHANISM_PANEL_V2_82CONTROL_109REP"


def _read_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"mechanism panel records missing: {path}")
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            validate_policy_return_portfolio_record(row)
            if str(row.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
                raise ValueError(f"{path}:{line_number}: wrong masked hybrid portfolio contract")
            if int(row.get("supervisory_control_dimension", -1)) != 82:
                raise ValueError(f"{path}:{line_number}: wrong supervisory-control dimension")
            if int(row.get("model_action_channel_count", -1)) != 109:
                raise ValueError(f"{path}:{line_number}: lost 109-channel model representation")
            mask_sha = str(row.get("supervisory_mask_sha256", "")).lower()
            if len(mask_sha) != 64:
                raise ValueError(f"{path}:{line_number}: missing supervisory-mask SHA")
            if row.get("passive_setting_channels_unchanged") is not True:
                raise ValueError(f"{path}:{line_number}: passive setting channel changed")
            score = float(row.get("base_step2_h10_score_m3", float("nan")))
            if not np.isfinite(score):
                raise ValueError(f"{path}:{line_number}: missing post-support base-Step2 score")
            records.append(row)
    if not records:
        raise ValueError("mechanism panel received zero records")
    mask_shas = {str(row["supervisory_mask_sha256"]).lower() for row in records}
    if len(mask_shas) != 1:
        raise ValueError("mechanism panel mixes supervisory-control mask lineages")
    return records


def _classify_query(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truths = np.asarray(
        [float(row["true_policy_return_delta_tfv_m3"]) for row in rows],
        dtype=np.float64,
    )
    predictions = np.asarray(
        [float(row["base_step2_h10_score_m3"]) for row in rows],
        dtype=np.float64,
    )
    if truths.size < 2 or not np.isfinite(truths).all() or not np.isfinite(predictions).all():
        raise ValueError("mechanism query requires >=2 finite candidate truths/predictions")
    sources = [str(row["candidate_source"]) for row in rows]
    oracle_index = int(np.argmin(truths))
    predicted_index = int(np.argmin(predictions))
    best_truth = float(truths[oracle_index])
    if best_truth >= 0.0:
        sign_class = "ALL_CANDIDATES_HARMFUL_HOLD_ORACLE"
        oracle_source = "HOLD"
        oracle_value = 0.0
    elif bool(np.all(truths < 0.0)):
        sign_class = "ALL_CANDIDATES_BENEFICIAL"
        oracle_source = sources[oracle_index]
        oracle_value = best_truth
    else:
        sign_class = "MIXED_BENEFICIAL_AND_HARMFUL"
        oracle_source = sources[oracle_index]
        oracle_value = best_truth

    gradient_indices = [
        index for index, source in enumerate(sources) if source == "SUPPORT_CONSTRAINED_GRADIENT_H10"
    ]
    gradient_truth = float(truths[gradient_indices[0]]) if len(gradient_indices) == 1 else None
    gradient_is_best = bool(
        len(gradient_indices) == 1 and best_truth < 0.0 and gradient_indices[0] == oracle_index
    )

    comparisons = correct = 0
    for left in range(len(truths)):
        for right in range(left + 1, len(truths)):
            if abs(truths[left] - truths[right]) <= 1.0:
                continue
            comparisons += 1
            correct += int(
                np.sign(predictions[left] - predictions[right])
                == np.sign(truths[left] - truths[right])
            )
    informative = np.abs(truths) > 1.0
    sign_accuracy = (
        float(np.mean(np.sign(predictions[informative]) == np.sign(truths[informative])))
        if np.any(informative)
        else 1.0
    )

    return {
        "query_set_id": str(rows[0]["query_set_id"]),
        "event_id": str(rows[0]["event_id"]),
        "rainfall_group": str(rows[0]["rainfall_group"]),
        "decision_index": int(rows[0]["decision_index"]),
        "decision_elapsed_seconds": int(rows[0].get("decision_elapsed_seconds", -1)),
        "candidate_count": len(rows),
        "candidate_sources": sources,
        "base_step2_h10_score_m3": predictions.astype(float).tolist(),
        "true_policy_return_delta_tfv_m3": truths.astype(float).tolist(),
        "best_candidate_source": sources[oracle_index],
        "best_candidate_true_policy_return_m3": best_truth,
        "base_step2_top_candidate_source": sources[predicted_index],
        "base_step2_candidate_top1_correct": bool(predicted_index == oracle_index),
        "base_step2_sign_accuracy": sign_accuracy,
        "base_step2_false_beneficial_count": int(np.sum((predictions < 0.0) & (truths >= 0.0))),
        "base_step2_false_reject_count": int(np.sum((predictions >= 0.0) & (truths < 0.0))),
        "base_step2_pairwise_rank_accuracy": float(correct / comparisons) if comparisons else 1.0,
        "base_step2_pairwise_comparison_count": int(comparisons),
        "oracle_action_with_hold": oracle_source,
        "oracle_action_value_m3": float(oracle_value),
        "sign_class": sign_class,
        "beneficial_candidate_count": int(np.sum(truths < 0.0)),
        "harmful_candidate_count": int(np.sum(truths >= 0.0)),
        "gradient_true_policy_return_m3": gradient_truth,
        "gradient_is_oracle_beneficial_candidate": gradient_is_best,
        "supervisory_mask_sha256": str(rows[0]["supervisory_mask_sha256"]),
        "same_prefix_verified": all(row.get("same_prefix_verified") is True for row in rows),
        "same_continuation_policy_verified": all(
            row.get("same_continuation_policy_verified") is True for row in rows
        ),
        "passive_setting_channels_unchanged": all(
            row.get("passive_setting_channels_unchanged") is True for row in rows
        ),
        "future_realized_rainfall_used_online": any(
            row.get("future_realized_rainfall_used_online") is not False for row in rows
        ),
    }


def _source_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped_truth: dict[str, list[float]] = defaultdict(list)
    grouped_prediction: dict[str, list[float]] = defaultdict(list)
    for row in records:
        source = str(row["candidate_source"])
        grouped_truth[source].append(float(row["true_policy_return_delta_tfv_m3"]))
        grouped_prediction[source].append(float(row["base_step2_h10_score_m3"]))
    result: dict[str, dict[str, Any]] = {}
    for source, values in sorted(grouped_truth.items()):
        truth = np.asarray(values, dtype=np.float64)
        pred = np.asarray(grouped_prediction[source], dtype=np.float64)
        result[source] = {
            "count": int(truth.size),
            "beneficial_count": int(np.sum(truth < 0.0)),
            "beneficial_fraction": float(np.mean(truth < 0.0)),
            "mean_true_policy_return_m3": float(np.mean(truth)),
            "median_true_policy_return_m3": float(median(truth.tolist())),
            "min_true_policy_return_m3": float(np.min(truth)),
            "max_true_policy_return_m3": float(np.max(truth)),
            "base_step2_sign_accuracy": float(np.mean(np.sign(pred) == np.sign(truth))),
            "base_step2_false_beneficial_fraction": float(np.mean((pred < 0.0) & (truth >= 0.0))),
            "base_step2_false_reject_fraction": float(np.mean((pred >= 0.0) & (truth < 0.0))),
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records-jsonl", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minimum-query-sets-for-directional-diagnosis", type=int, default=6)
    args = p.parse_args()
    if int(args.minimum_query_sets_for_directional_diagnosis) < 2:
        raise ValueError("minimum mechanism-panel query count must be >=2")

    records = _read_records(list(args.records_jsonl))
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_query[str(row["query_set_id"])].append(row)
    queries = [_classify_query(by_query[key]) for key in sorted(by_query)]

    if not all(row["same_prefix_verified"] for row in queries):
        raise RuntimeError("mechanism panel contains a raw-prefix verification failure")
    if not all(row["same_continuation_policy_verified"] for row in queries):
        raise RuntimeError("mechanism panel contains a continuation-policy verification failure")
    if not all(row["passive_setting_channels_unchanged"] for row in queries):
        raise RuntimeError("mechanism panel contains a passive-setting change")
    if any(row["future_realized_rainfall_used_online"] for row in queries):
        raise RuntimeError("mechanism panel indicates future-rainfall leakage")

    classes = Counter(row["sign_class"] for row in queries)
    beneficial_queries = sum(row["beneficial_candidate_count"] > 0 for row in queries)
    gradient_queries = [row for row in queries if row["gradient_true_policy_return_m3"] is not None]
    gradient_beneficial = sum(
        float(row["gradient_true_policy_return_m3"]) < 0.0 for row in gradient_queries
    )
    gradient_oracle = sum(
        bool(row["gradient_is_oracle_beneficial_candidate"]) for row in gradient_queries
    )
    candidate_truth = np.asarray(
        [float(row["true_policy_return_delta_tfv_m3"]) for row in records], dtype=np.float64
    )
    candidate_prediction = np.asarray(
        [float(row["base_step2_h10_score_m3"]) for row in records], dtype=np.float64
    )
    informative = np.abs(candidate_truth) > 1.0
    pair_correct = sum(
        int(round(float(row["base_step2_pairwise_rank_accuracy"]) * int(row["base_step2_pairwise_comparison_count"])))
        for row in queries
    )
    pair_count = sum(int(row["base_step2_pairwise_comparison_count"]) for row in queries)
    minimum = int(args.minimum_query_sets_for_directional_diagnosis)
    if len(queries) < minimum:
        interpretation = "INSUFFICIENT_SEEN_MECHANISM_PANEL_DO_NOT_CHANGE_STEP2_YET"
    elif beneficial_queries == 0:
        interpretation = (
            "NO_BENEFICIAL_GENERATED_ACTION_ACROSS_SEEN_PANEL_"
            "INSPECT_PROPOSAL_COVERAGE_AND_STEP2_STATE_VALUE_REPRESENTATION_BEFORE_BULK"
        )
    else:
        interpretation = (
            "BENEFICIAL_GENERATED_ACTIONS_EXIST_EXACT_POLICY_RETURN_CRITIC_LEARNING_IS_JUSTIFIED"
        )

    payload = {
        "contract": MECHANISM_PANEL_CONTRACT,
        "development_only": True,
        "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "supervisory_mask_sha256": str(records[0]["supervisory_mask_sha256"]),
        "query_selection_requirement": "PRECOMMITTED_LABEL_BLIND_FROM_SEEN_DEVELOPMENT_PARENT_TRAJECTORIES",
        "query_set_count": len(queries),
        "candidate_record_count": len(records),
        "event_count": len({str(row["event_id"]) for row in queries}),
        "events": sorted({str(row["event_id"]) for row in queries}),
        "rainfall_group_count": len({str(row["rainfall_group"]) for row in queries}),
        "sign_class_counts": dict(sorted(classes.items())),
        "all_harmful_hold_oracle_query_fraction": float(
            classes["ALL_CANDIDATES_HARMFUL_HOLD_ORACLE"] / len(queries)
        ),
        "query_with_any_beneficial_candidate_fraction": float(beneficial_queries / len(queries)),
        "gradient_query_count": len(gradient_queries),
        "gradient_beneficial_query_fraction": (
            float(gradient_beneficial / len(gradient_queries)) if gradient_queries else None
        ),
        "gradient_oracle_beneficial_query_fraction": (
            float(gradient_oracle / len(gradient_queries)) if gradient_queries else None
        ),
        "base_step2_candidate_sign_accuracy": (
            float(np.mean(np.sign(candidate_prediction[informative]) == np.sign(candidate_truth[informative])))
            if np.any(informative)
            else 1.0
        ),
        "base_step2_candidate_false_beneficial_fraction": float(
            np.mean((candidate_prediction < 0.0) & (candidate_truth >= 0.0))
        ),
        "base_step2_candidate_false_reject_fraction": float(
            np.mean((candidate_prediction >= 0.0) & (candidate_truth < 0.0))
        ),
        "base_step2_within_query_pairwise_rank_accuracy": float(pair_correct / pair_count) if pair_count else 1.0,
        "base_step2_candidate_top1_accuracy": float(
            np.mean([bool(row["base_step2_candidate_top1_correct"]) for row in queries])
        ),
        "candidate_source_summary": _source_summary(records),
        "directional_interpretation": interpretation,
        "important_scope_note": (
            "ALL_CANDIDATES_HARMFUL means HOLD is oracle-optimal only within the generated portfolio "
            "for that exact state under the 82-control supervisory mask. It does not prove that no "
            "other engineering-feasible action exists or that the frozen 109-channel hydraulic "
            "representation is universally invalid."
        ),
        "queries": queries,
        "ready_for_policy_lock": False,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
