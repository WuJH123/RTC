"""Diagnostic evidence helpers for Direct-TFV decision-level SWMM counterfactuals.

This module does not alter controller actions. It selects a small deterministic set of completed
non-HOLD decisions whose exact H120 free-control plan and H360 HOLD reference were recorded by the
authoritative runtime. These decisions can then be replayed from the same causal hydraulic prefix
in authoritative SWMM to test value magnitude/sign without confusing a local H360 counterfactual
with whole-event closed-loop benefit.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence


DIRECT_TFV_COUNTERFACTUAL_MANIFEST_CONTRACT = (
    "PROJECT7_DIRECT_TFV_DECISION_COUNTERFACTUAL_MANIFEST_V1"
)
DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT = (
    "PROJECT7_DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_V1"
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
        if str(row.get("source", "")) != "MPC_DIRECT_TFV_RECEDING":
            continue
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        if diagnostics.get("counterfactual_plan_telemetry_contract") != (
            DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT
        ):
            continue
        predicted = float(diagnostics.get("predicted_delta_tfv_m3", math.nan))
        blocks = diagnostics.get("optimized_free_control_blocks")
        hold = diagnostics.get("hold_reference_settings")
        if not math.isfinite(predicted) or predicted >= 0.0:
            continue
        if not (
            isinstance(blocks, list)
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
                "source": str(row.get("source", "")),
                "predicted_delta_tfv_m3": predicted,
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
    """Select strong/median/mild predictions while preserving a complete truth horizon.

    ``latest_elapsed_seconds`` should be set to ``simulation_end_elapsed - 21600`` for the current
    H360 value estimand. This prevents a truncated end-of-event SWMM branch from being compared with
    a full H360 Step3 prediction.
    """

    if max_decisions <= 0 or max_decisions > 6:
        raise ValueError("counterfactual diagnostic budget must lie in [1,6]")
    if latest_elapsed_seconds is not None and latest_elapsed_seconds < 0:
        raise ValueError("latest_elapsed_seconds must be non-negative")
    eligible = sorted(
        _eligible(rows, latest_elapsed_seconds=latest_elapsed_seconds),
        key=lambda row: (float(row["predicted_delta_tfv_m3"]), int(row["decision_index"])),
    )
    if len(eligible) <= max_decisions:
        chosen = eligible
    else:
        count = len(eligible)
        # Prioritise two strongest, two around the distribution median, and two mildest. For smaller
        # budgets the deterministic priority order still spans the prediction distribution.
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
