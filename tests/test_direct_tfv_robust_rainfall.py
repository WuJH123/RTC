from __future__ import annotations

import pytest

from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS
from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT,
    derive_first_move_admission,
)
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DirectTFVScenarioMeanMPCV10,
)


def test_scenario_mean_policy_rejects_single_scenario_v11_admission_before_model_load() -> None:
    policy = {"contract": DIRECT_TFV_POLICY_ADMISSION_CONTRACT}
    old_first_move = {
        "contract": DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
        "development_only": True,
        "execution_estimand": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
        "query_step3_contract": DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
        "generic_d3_floor_controls_execution": False,
        "calibration_rainfall_group_count": 24,
        "minimum_calibration_rainfall_groups": 24,
    }
    with pytest.raises(ValueError, match="cannot reuse a single-scenario V11"):
        DirectTFVScenarioMeanMPCV10(
            model=None,
            graph=None,
            normalization=None,
            action_support={},
            policy_admission_calibration=policy,
            first_move_admission_calibration=old_first_move,
            sequence_support={},
        )


def test_v12_admission_must_record_scenario_query_and_rainfall_contract() -> None:
    rows = [
        {
            "rainfall_group": f"g{i:02d}",
            "plan_sha256": f"plan{i:02d}",
            "predicted_refined_delta_tfv_m3": -100.0,
            "true_refined_delta_tfv_m3": -80.0 + i,
            "first_move_changed_facility_count": 4,
        }
        for i in range(24)
    ]
    result = derive_first_move_admission(
        panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT,
        panel_records=rows,
        expected_rainfall_groups=[f"g{i:02d}" for i in range(24)],
        rainfall_scenario_contract=DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    )
    assert result["query_step3_contract"] == DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT
    assert result["rainfall_scenario_contract"] == DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT


def test_v12_admission_fails_without_rainfall_scenario_lineage() -> None:
    rows = [
        {
            "rainfall_group": f"g{i:02d}",
            "plan_sha256": f"plan{i:02d}",
            "predicted_refined_delta_tfv_m3": -1.0,
            "true_refined_delta_tfv_m3": -1.0,
            "first_move_changed_facility_count": 1,
        }
        for i in range(24)
    ]
    with pytest.raises(ValueError, match="requires rainfall scenario lineage"):
        derive_first_move_admission(
            panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
            panel_step3_contract=DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT,
            panel_records=rows,
            expected_rainfall_groups=[f"g{i:02d}" for i in range(24)],
        )
