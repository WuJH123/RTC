"""Summarize a preregistered seen-event policy-return mechanism panel.

This is Development-only diagnostics. It aggregates exact same-prefix query records without changing
the controller, training data roles, conformal margin, or untouched evaluation sets. An all-harmful
query means HOLD is oracle-optimal *within the generated portfolio at that state*; it is not by itself
proof that the global Step2 representation or candidate family has failed.

Recommended first use: two label-blind query points from each already-seen T8/T30/T80 Development
event (six query sets total). Query points must be frozen before candidate SWMM truth is inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from rtc.direct_tfv_policy_return import validate_policy_return_record
from rtc.direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
)


MECHANISM_PANEL_CONTRACT = "PROJECT7_POLICY_RETURN_SEEN_MECHANISM_PANEL_V1"


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
            validate_policy_return_record(row)
            if str(row.get("candidate_portfolio_contract", "")) != (
                DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
            ):
                raise ValueError(f"{path}:{line_number}: wrong hybrid portfolio contract")
            records.append(row)
    if not records:
        raise ValueError("mechanism panel received zero records")
    return records


def _classify_query(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truths = np.asarray(
        [float(row["true_policy_return_delta_tfv_m3"]) for row in rows],
        dtype=np.float64,
    )
    if truths.size < 2 or not np.isfinite(truths).all():
        raise ValueError("mechanism query requires >=2 finite candidate truths")
    sources = [str(row["candidate_source"]) for row in rows]
    oracle_index = int(np.argmin(truths))
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
    gradient_truth = (
        float(truths[gradient_indices[0]]) if len(gradient_indices) == 1 else None
    )
    gradient_is_best = bool(
        len(gradient_indices) == 1
        and best_truth < 0.0
        and gradient_indices[0] == oracle_index
    )

    return {
        "query_set_id": str(rows[0]["query_set_id"]),
        "event_id": str(rows[0]["event_id"]),
        "rainfall_group": str(rows[0]["rainfall_group"]),
        "decision_index": int(rows[0]["decision_index"]),
        "decision_elapsed_seconds": int(rows[0].get("decision_elapsed_seconds", -1)),
        "candidate_count": len(rows),
        "candidate_sources": sources,
        "true_policy_return_delta_tfv_m3": truths.astype(float).tolist(),
        "best_candidate_source": sources[oracle_index],
        "best_candidate_true_policy_return_m3": best_truth,
        "oracle_action_with_hold": oracle_source,
        "oracle_action_value_m3": float(oracle_value),
        "sign_class": sign_class,
        "beneficial_candidate_count": int(np.sum(truths < 0.0)),
        "harmful_candidate_count": int(np.sum(truths >= 0.0)),
        "gradient_true_policy_return_m3": gradient_truth,
        "gradient_is_oracle_beneficial_candidate": gradient_is_best,
        "same_prefix_verified": all(row.get("same_prefix_verified") is True for row in rows),
        "same_continuation_policy_verified": all(
            row.get("same_continuation_policy_verified") is True for row in rows
        ),
        "future_realized_rainfall_used_online": any(
            row.get("future_realized_rainfall_used_online") is not False for row in rows
        ),
    }


def _source_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[str(row["candidate_source"])].append(
            float(row["true_policy_return_delta_tfv_m3"])
        )
    result: dict[str, dict[str, Any]] = {}
    for source, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        result[source] = {
            "count": int(array.size),
            "beneficial_count": int(np.sum(array < 0.0)),
            "beneficial_fraction": float(np.mean(array < 0.0)),
            "mean_true_policy_return_m3": float(np.mean(array)),
            "median_true_policy_return_m3": float(median(array.tolist())),
            "min_true_policy_return_m3": float(np.min(array)),
            "max_true_policy_return_m3": float(np.max(array)),
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
    if any(row["future_realized_rainfall_used_online"] for row in queries):
        raise RuntimeError("mechanism panel indicates future-rainfall leakage")

    classes = Counter(row["sign_class"] for row in queries)
    beneficial_queries = sum(row["beneficial_candidate_count"] > 0 for row in queries)
    gradient_queries = [
        row for row in queries if row["gradient_true_policy_return_m3"] is not None
    ]
    gradient_beneficial = sum(
        float(row["gradient_true_policy_return_m3"]) < 0.0 for row in gradient_queries
    )
    gradient_oracle = sum(
        bool(row["gradient_is_oracle_beneficial_candidate"]) for row in gradient_queries
    )
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
        "query_selection_requirement": (
            "PRECOMMITTED_LABEL_BLIND_FROM_SEEN_DEVELOPMENT_PARENT_TRAJECTORIES"
        ),
        "query_set_count": len(queries),
        "candidate_record_count": len(records),
        "event_count": len({str(row["event_id"]) for row in queries}),
        "events": sorted({str(row["event_id"]) for row in queries}),
        "rainfall_group_count": len({str(row["rainfall_group"]) for row in queries}),
        "sign_class_counts": dict(sorted(classes.items())),
        "all_harmful_hold_oracle_query_fraction": float(
            classes["ALL_CANDIDATES_HARMFUL_HOLD_ORACLE"] / len(queries)
        ),
        "query_with_any_beneficial_candidate_fraction": float(
            beneficial_queries / len(queries)
        ),
        "gradient_query_count": len(gradient_queries),
        "gradient_beneficial_query_fraction": (
            float(gradient_beneficial / len(gradient_queries))
            if gradient_queries
            else None
        ),
        "gradient_oracle_beneficial_query_fraction": (
            float(gradient_oracle / len(gradient_queries)) if gradient_queries else None
        ),
        "candidate_source_summary": _source_summary(records),
        "directional_interpretation": interpretation,
        "important_scope_note": (
            "ALL_CANDIDATES_HARMFUL means HOLD is oracle-optimal only within the generated portfolio "
            "for that exact state. It is not evidence that no beneficial 109-D engineering-feasible "
            "action exists elsewhere."
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
