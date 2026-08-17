from __future__ import annotations

from rtc.direct_tfv_counterfactual import (
    DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT,
    select_counterfactual_decisions,
)


def _row(index: int, predicted: float, *, source: str = "MPC_DIRECT_TFV_RECEDING") -> dict:
    return {
        "elapsed_seconds": 3600 + index * 600,
        "source": source,
        "diagnostics": {
            "counterfactual_plan_telemetry_contract": (
                DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT
            ),
            "predicted_delta_tfv_m3": predicted,
            "best_screening_predicted_delta_tfv_m3": predicted / 2.0,
            "optimizer_gain_beyond_best_screening_m3": -predicted / 2.0,
            "predicted_beneficial_facility_count": 40,
            "active_facility_count": 22,
            "active_facility_ids": [f"A{i:03d}" for i in range(22)],
            "active_set_ceiling_binding": True,
            "first_move_changed_facility_count": 10,
            "hold_reference_settings": [0.5] * 109,
            "optimized_free_control_blocks": [[0.5 + index * 0.001] * 109 for _ in range(12)],
            "counterfactual_reference_semantics": "HOLD_ACTIVE_TARGET_H360",
            "counterfactual_candidate_semantics": (
                "EXACT_OPTIMIZED_H120_FREE_BLOCKS_THEN_TERMINAL_HOLD_H360"
            ),
        },
    }


def test_selector_spans_strong_median_and_mild_predictions() -> None:
    rows = [_row(i, -1000.0 + i * 50.0) for i in range(12)]
    selected = select_counterfactual_decisions(rows, max_decisions=6)
    assert len(selected) == 6
    indices = [int(row["decision_index"]) for row in selected]
    assert indices[:2] == [0, 1]
    assert 5 in indices and 6 in indices
    assert indices[-2:] == [10, 11]
    assert all(len(str(row["plan_sha256"])) == 64 for row in selected)


def test_selector_ignores_hold_and_missing_plan_telemetry() -> None:
    valid = _row(0, -100.0)
    hold = _row(1, -200.0, source="HOLD_DIRECT_TFV_NO_PREDICTED_BENEFIT")
    missing = _row(2, -300.0)
    del missing["diagnostics"]["optimized_free_control_blocks"]
    selected = select_counterfactual_decisions([valid, hold, missing], max_decisions=6)
    assert [row["decision_index"] for row in selected] == [0]


def test_selector_excludes_decisions_without_complete_truth_horizon() -> None:
    rows = [_row(i, -1000.0 + i * 50.0) for i in range(8)]
    selected = select_counterfactual_decisions(
        rows,
        max_decisions=6,
        latest_elapsed_seconds=3600 + 3 * 600,
    )
    assert len(selected) == 4
    assert all(int(row["elapsed_seconds"]) <= 5400 for row in selected)
    assert {int(row["decision_index"]) for row in selected} == {0, 1, 2, 3}


def test_selector_rejects_unbounded_diagnostic_budget() -> None:
    try:
        select_counterfactual_decisions([_row(0, -100.0)], max_decisions=7)
    except ValueError as exc:
        assert "[1,6]" in str(exc)
    else:
        raise AssertionError("counterfactual selector accepted an unbounded SWMM diagnostic budget")
