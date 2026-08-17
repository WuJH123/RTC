from __future__ import annotations

from rtc.direct_tfv_counterfactual import (
    DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT,
    select_counterfactual_decisions,
)


def _row(index: int, predicted: float, *, admitted: bool = True) -> dict:
    source = "MPC_DIRECT_TFV_RECEDING" if admitted else "HOLD_DIRECT_TFV_CALIBRATED_OR_NO_BENEFIT"
    upper = predicted + (100.0 if admitted else abs(predicted) + 100.0)
    return {
        "elapsed_seconds": 3600 + index * 600,
        "source": source,
        "diagnostics": {
            "direct_tfv_selected_source": (
                "DIRECT_TFV_RECEDING_LBFGSB"
                if admitted
                else "HOLD_CALIBRATED_TFV_UPPER_BOUND_NONNEGATIVE"
            ),
            "counterfactual_plan_telemetry_contract": (
                DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_CONTRACT
            ),
            "raw_optimized_predicted_delta_tfv_m3": predicted,
            "predicted_delta_tfv_m3": predicted if admitted else 0.0,
            "admission_margin_m3": upper - predicted,
            "admission_upper_bound_m3": upper,
            "admission_margin_kind": "dense",
            "admission_passed": admitted,
            "best_screening_predicted_delta_tfv_m3": predicted / 2.0,
            "optimizer_gain_beyond_best_screening_m3": -predicted / 2.0,
            "predicted_beneficial_facility_count": 40,
            "active_facility_count": 23,
            "active_facility_ids": [f"A{i:03d}" for i in range(23)],
            "active_set_ceiling_binding": True,
            "first_move_changed_facility_count": 10,
            "counterfactual_actuator_ids": [f"A{i:03d}" for i in range(109)],
            "hold_reference_settings": [0.5] * 109,
            "optimized_free_control_blocks": [[0.5 + index * 0.001] * 109 for _ in range(12)],
            "counterfactual_reference_semantics": "HOLD_ACTIVE_TARGET_H360",
            "counterfactual_candidate_semantics": (
                "RAW_OPTIMIZED_H120_FREE_BLOCKS_THEN_TERMINAL_HOLD_H360"
            ),
        },
    }


def test_selector_spans_strong_median_and_mild_raw_predictions() -> None:
    rows = [_row(i, -1000.0 + i * 50.0, admitted=i < 6) for i in range(12)]
    selected = select_counterfactual_decisions(rows, max_decisions=6)
    assert len(selected) == 6
    indices = [int(row["decision_index"]) for row in selected]
    assert indices[:2] == [0, 1]
    assert 5 in indices and 6 in indices
    assert indices[-2:] == [10, 11]
    assert all(len(str(row["plan_sha256"])) == 64 for row in selected)
    assert all(len(row["counterfactual_actuator_ids"]) == 109 for row in selected)


def test_selector_keeps_rejected_raw_optimizer_plan_for_scientific_audit() -> None:
    accepted = _row(0, -1000.0, admitted=True)
    rejected = _row(1, -100.0, admitted=False)
    missing = _row(2, -300.0, admitted=False)
    del missing["diagnostics"]["optimized_free_control_blocks"]
    selected = select_counterfactual_decisions([accepted, rejected, missing], max_decisions=6)
    assert [row["decision_index"] for row in selected] == [0, 1]
    by_index = {int(row["decision_index"]): row for row in selected}
    assert by_index[0]["admission_passed"] is True
    assert by_index[1]["admission_passed"] is False
    assert by_index[1]["executed_source"].startswith("HOLD_DIRECT_TFV_")


def test_selector_requires_unique_ordered_actuator_ids() -> None:
    invalid = _row(0, -100.0)
    invalid["diagnostics"]["counterfactual_actuator_ids"][-1] = "A000"
    assert select_counterfactual_decisions([invalid], max_decisions=6) == []


def test_plan_hash_binds_actuator_order() -> None:
    first = _row(0, -100.0)
    second = _row(0, -100.0)
    ids = second["diagnostics"]["counterfactual_actuator_ids"]
    ids[0], ids[1] = ids[1], ids[0]
    first_selected = select_counterfactual_decisions([first], max_decisions=6)
    second_selected = select_counterfactual_decisions([second], max_decisions=6)
    assert first_selected[0]["plan_sha256"] != second_selected[0]["plan_sha256"]


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
