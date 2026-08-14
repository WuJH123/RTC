from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_policy_v123 import FirstMoveTFVPFVResultV123
from rtc.step3_calibration_v123 import fit_one_sided_value_calibration_v123
from rtc.step3_candidates_v123 import (
    FirstMoveCandidateDesignV123,
    candidate_coefficients_v123,
    unique_executable_first_moves_v123,
)
from rtc.step3_objective_v123 import TFVPFVObjectiveV123, tfv_pfv_score_v123


class _Basis:
    def __init__(self):
        self.temporal_basis_count = 3
        self.group_count = 3
        self.temporal_basis = np.asarray(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        self.grouping = SimpleNamespace(
            zone_id_by_actuator=np.asarray([0, 0, 1], dtype=np.int64),
            group_id_by_actuator=np.asarray([0, 1, 2], dtype=np.int64),
        )

    def validate(self):
        return None


def _objective(**kwargs) -> TFVPFVObjectiveV123:
    values = {
        "pfv_soft_margin_m3": 100.0,
        "pfv_scale_m3": 1000.0,
        "tfv_scale_m3": 10000.0,
        "pfv_penalty_weight": 0.5,
    }
    values.update(kwargs)
    return TFVPFVObjectiveV123(**values)


def test_pfv_improvement_does_not_buy_worse_tfv() -> None:
    tfv = torch.tensor([[1000.0, -500.0]])
    pfv = torch.tensor([[-10000.0, 0.0]])
    out = tfv_pfv_score_v123(tfv, pfv, movement=None, contract=_objective())
    assert out["score_m3_equivalent"][0] == 1000.0
    assert out["score_m3_equivalent"][1] == -500.0


def test_pfv_deterioration_is_soft_not_hard() -> None:
    tfv = torch.tensor([[-10000.0, -2000.0]])
    pfv = torch.tensor([[1100.0, 0.0]])
    out = tfv_pfv_score_v123(tfv, pfv, movement=None, contract=_objective())
    assert out["pfv_penalty_m3_equivalent"][0] == 5000.0
    assert out["score_m3_equivalent"][0] == -5000.0
    assert out["score_m3_equivalent"][0] < out["score_m3_equivalent"][1]


def test_pfv_below_margin_has_no_penalty() -> None:
    tfv = torch.tensor([[-1000.0]])
    pfv = torch.tensor([[99.0]])
    out = tfv_pfv_score_v123(tfv, pfv, movement=None, contract=_objective())
    assert out["pfv_penalty_m3_equivalent"].item() == 0.0
    assert out["score_m3_equivalent"].item() == -1000.0


def test_calibrated_pfv_error_budget_applies_only_to_active_move() -> None:
    tfv = torch.tensor([[0.0, -3000.0]])
    pfv = torch.tensor([[0.0, 20.0]])
    movement = torch.tensor([0.0, 0.25])
    out = tfv_pfv_score_v123(
        tfv,
        pfv,
        movement=movement,
        contract=_objective(pfv_model_error_margin_m3=200.0),
    )
    # PASSIVE must remain exact zero despite a positive calibrated error budget.
    assert out["pfv_risk_m3"][0].item() == 0.0
    assert out["score_m3_equivalent"][0].item() == 0.0
    # Active move is conservatively evaluated as 20 + 200 m3 PFV risk.
    assert out["pfv_risk_m3"][1].item() == 220.0
    assert out["pfv_soft_excess_m3"][1].item() == 120.0


def test_calibration_exposes_separate_tfv_and_pfv_one_sided_margins() -> None:
    fitted = fit_one_sided_value_calibration_v123(
        tfv_truth_m3=[0.0, 20.0, 100.0, 120.0],
        tfv_prediction_m3=[0.0, 0.0, 60.0, 60.0],
        pfv_truth_m3=[0.0, 10.0, 50.0, 80.0],
        pfv_prediction_m3=[0.0, 0.0, 20.0, 20.0],
        rainfall_groups=["a", "a", "b", "b"],
        quantile=0.75,
    )
    payload = fitted.as_payload()
    assert fitted.tfv_false_benefit_margin_m3 >= 0.0
    assert fitted.pfv_false_safety_margin_m3 >= 0.0
    assert payload["pfv_false_safety_margin_m3"] == fitted.pfv_false_safety_margin_m3


def test_v123_result_exposes_v122_runtime_compatibility_aliases() -> None:
    result = FirstMoveTFVPFVResultV123(
        settings=torch.zeros((4, 2)),
        candidate_valid=True,
        selected_candidate_index=3,
        raw_candidate_count=169,
        first_move_group_count=41,
        tail_only_noop_candidate_count=0,
        scenario_count=3,
        predicted_delta_tfv_m3=-5000.0,
        predicted_delta_pfv_m3=50.0,
        tfv_risk_m3=-4000.0,
        pfv_risk_m3=120.0,
        pfv_soft_excess_m3=20.0,
        pfv_penalty_m3_equivalent=100.0,
        objective_score_m3_equivalent=-3900.0,
        selected_group_score_m3=-3900.0,
        false_benefit_margin_m3=1000.0,
        scoring_projection_max=0.5,
    )
    assert result.candidate_count == 169
    assert result.selected_group_score_m3 == -3900.0
    assert result.controller_source == "MPC_V123"


def test_candidate_design_has_no_tail_only_nonhold_coefficients() -> None:
    coeff = candidate_coefficients_v123(
        _Basis(),
        design=FirstMoveCandidateDesignV123(max_cross_zone_pairs=2),
    )
    assert np.allclose(coeff[0], 0.0)
    assert np.all(np.max(np.abs(coeff[1:, 0, :]), axis=1) > 0.0)


def test_executable_first_move_dedup_reports_tail_only_collisions() -> None:
    reference = torch.zeros((4, 1))
    candidates = torch.tensor(
        [
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.5], [0.5], [0.5], [0.5]],
        ]
    )
    kept, report = unique_executable_first_moves_v123(
        candidates, reference, control_block_steps=2
    )
    assert kept.shape[0] == 2
    assert report["raw_candidate_count"] == 3
    assert report["unique_first_move_count"] == 2
    assert report["tail_only_or_passive_like_count"] == 1
