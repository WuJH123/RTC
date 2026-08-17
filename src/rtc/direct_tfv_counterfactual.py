"""Diagnostic selection for exact same-prefix Direct-TFV H360 counterfactual replay.

V2 selects raw optimizer plans whether calibrated admission accepted or rejected them. This is
necessary to test the scientific claim made by Step3 V5: rejected extrema should contain the weak or
false-beneficial plans while admitted extrema should remain beneficial. The helper never changes the
executed controller action.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence


DIRECT_TFV_COUNTERFACTUAL_MANIFEST_CONTRACT = (
    "PROJECT7_DIRECT_TFV_DECISION_COUNTERFACTUAL_MANIFEST_V2"
)
DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT = (
    "PROJECT7_DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_V2"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _eligible(
    rows: Sequence[dict[str, Any]], *, latest_elapsed_seconds: int | None
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        elapsed = int(row.get("elapsed_seconds", -1))
        if latest_elapsed_seconds is not None and elapsed > int(latest_elapsed_seconds):
            continue
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        if diagnostics.get("counterfactual_plan_telemetry_contract") != (
            DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT
        ):
            continue
        raw_predicted = float(
            diagnostics.get(
                "raw_optimized_predicted_delta_tfv_m3",
                diagnostics.get("predicted_delta_tfv_m3", math.nan),
            )
        )
        blocks = diagnostics.get("optimized_free_control_blocks")
        hold = diagnostics.get("hold_reference_settings")
        actuator_ids = diagnostics.get("counterfactual_actuator_ids")
        valid_actuator_ids = bool(
            isinstance(actuator_ids, list)
            and len(actuator_ids) == 109
            and len({str(value) for value in actuator_ids}) == 109
        )
        if not math.isfinite(raw_predicted) or raw_predicted >= 0.0:
            continue
        if not (
            valid_actuator_ids
            and isinstance(blocks, list)
            and len(blocks) == 12
            and all(isinstance(block, list) and len(block) == 109 for block in blocks)
            and isinstance(hold, list)
            and len(hold) == 109
        ):
            continue
        eligible.append(
            {
                "decision_index": index,
                "elapsed_seconds": elapsed,
                "executed_source": str(row.get("source", "")),
                "optimizer_selected_source": str(
                    diagnostics.get("direct_tfv_selected_source", "")
                ),
                "raw_optimized_predicted_delta_tfv_m3": raw_predicted,
                "admission_margin_m3": float(
                    diagnostics.get("admission_margin_m3", math.nan)
                ),
                "admission_upper_bound_m3": float(
                    diagnostics.get("admission_upper_bound_m3", math.nan)
                ),
                "admission_margin_kind": str(
                    diagnostics.get("admission_margin_kind", "")
                ),
                "admission_passed": bool(diagnostics.get("admission_passed", False)),
                "best_screening_predicted_delta_tfv_m3": float(
                    diagnostics.get("best_screening_predicted_delta_tfv_m3", math.nan)
                ),
                "optimizer_gain_beyond_best_screening_m3": float(
                    diagnostics.get("optimizer_gain_beyond_best_screening_m3", math.nan)
                ),
                "predicted_beneficial_facility_count": int(
                    diagnostics.get("predicted_beneficial_facility_count", -1)
                ),
                "active_facility_count": int(diagnostics.get("active_facility_count", -1)),
                "active_facility_ids": list(diagnostics.get("active_facility_ids", [])),
                "active_set_ceiling_binding": bool(
                    diagnostics.get("active_set_ceiling_binding", False)
                ),
                "first_move_changed_facility_count": int(
                    diagnostics.get("first_move_changed_facility_count", -1)
                ),
                "counterfactual_actuator_ids": [str(value) for value in actuator_ids],
                "hold_reference_settings": hold,
                "optimized_free_control_blocks": blocks,
                "counterfactual_reference_semantics": str(
                    diagnostics.get("counterfactual_reference_semantics", "")
                ),
                "counterfactual_candidate_semantics": str(
                    diagnostics.get("counterfactual_candidate_semantics", "")
                ),
            }
        )
    return eligible


def select_counterfactual_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    max_decisions: int = 6,
    latest_elapsed_seconds: int | None = None,
) -> list[dict[str, Any]]:
    """Select strong/median/mild raw optimizer predictions across admission outcomes."""

    if max_decisions <= 0 or max_decisions > 6:
        raise ValueError("counterfactual diagnostic budget must lie in [1,6]")
    if latest_elapsed_seconds is not None and latest_elapsed_seconds < 0:
        raise ValueError("latest_elapsed_seconds must be non-negative")
    eligible = sorted(
        _eligible(rows, latest_elapsed_seconds=latest_elapsed_seconds),
        key=lambda row: (
            float(row["raw_optimized_predicted_delta_tfv_m3"]),
            int(row["decision_index"]),
        ),
    )
    if len(eligible) <= max_decisions:
        chosen = eligible
    else:
        count = len(eligible)
        candidate_positions = [
            0,
            1,
            max(0, (count - 1) // 2),
            min(count - 1, count // 2),
            max(0, count - 2),
            count - 1,
        ]
        chosen = []
        seen: set[int] = set()
        for position in candidate_positions:
            decision_index = int(eligible[position]["decision_index"])
            if decision_index in seen:
                continue
            chosen.append(eligible[position])
            seen.add(decision_index)
            if len(chosen) >= max_decisions:
                break
    output: list[dict[str, Any]] = []
    for row in chosen:
        payload = dict(row)
        payload["plan_sha256"] = _canonical_sha256(
            {
                "counterfactual_actuator_ids": payload["counterfactual_actuator_ids"],
                "hold_reference_settings": payload["hold_reference_settings"],
                "optimized_free_control_blocks": payload["optimized_free_control_blocks"],
            }
        )
        output.append(payload)
    return output


__all__ = [
    "DIRECT_TFV_COUNTERFACTUAL_MANIFEST_CONTRACT",
    "DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT",
    "select_counterfactual_decisions",
]
