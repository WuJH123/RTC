from __future__ import annotations

import pytest

from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS
from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
)
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.step3_tfv_value_mpc_v10 import DirectTFVScenarioMeanMPCV10


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
