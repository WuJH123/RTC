from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rtc.controller_v123 import V123_CONTROLLER_CONTRACT, V123TorchMPCController
from rtc.step2_policy_v123 import FirstMoveTFVPFVResultV123


def test_v123_result_exposes_controller_diagnostic_aliases() -> None:
    result = FirstMoveTFVPFVResultV123(
        settings=np.zeros((2, 1)),
        candidate_valid=True,
        selected_candidate_index=3,
        raw_candidate_count=169,
        first_move_group_count=75,
        tail_only_noop_candidate_count=58,
        scenario_count=1,
        predicted_delta_tfv_m3=-12000.0,
        predicted_delta_pfv_m3=2.0,
        tfv_risk_m3=-12000.0,
        pfv_risk_m3=2.0,
        pfv_soft_excess_m3=0.0,
        pfv_penalty_m3_equivalent=0.0,
        objective_score_m3_equivalent=-11998.0,
        false_benefit_margin_m3=1000.0,
        scoring_projection_max=0.0,
    )
    assert result.candidate_count == 169
    assert result.selected_group_score_m3 == -11998.0


def test_v123_controller_contract_is_distinct_from_v122() -> None:
    assert V123_CONTROLLER_CONTRACT.startswith("PROJECT7_V123_")
    assert issubclass(V123TorchMPCController, object)
