from __future__ import annotations

import torch

from rtc.step3_objective_v123 import TFVPFVObjectiveV123
from rtc.step3_objective_v125 import tfv_pfv_score_v125


def _contract() -> TFVPFVObjectiveV123:
    return TFVPFVObjectiveV123(
        pfv_soft_margin_m3=50.0,
        pfv_scale_m3=100.0,
        tfv_scale_m3=100.0,
        pfv_penalty_weight=1.0,
        pfv_model_error_margin_m3=30.0,
        movement_penalty_m3=10.0,
    )


def test_anchor_reference_stays_exact_zero_with_nonzero_pfv_model_error() -> None:
    result = tfv_pfv_score_v125(
        torch.tensor([[0.0, -100.0]]),
        torch.tensor([[0.0, 0.0]]),
        movement_from_anchor=torch.tensor([0.0, 0.2]),
        contract=_contract(),
    )
    assert float(result["tfv_risk_m3"][0]) == 0.0
    assert float(result["pfv_model_error_applied_m3"][0]) == 0.0
    assert float(result["pfv_risk_m3"][0]) == 0.0
    assert float(result["score_m3_equivalent"][0]) == 0.0


def test_changed_candidate_receives_pfv_error_margin_and_movement_penalty() -> None:
    result = tfv_pfv_score_v125(
        torch.tensor([[-100.0]]),
        torch.tensor([[25.0]]),
        movement_from_anchor=torch.tensor([0.2]),
        contract=_contract(),
    )
    # Direct PFV prediction 25 + one-sided model error 30 = risk 55, hence 5 m3
    # above the soft margin. Movement contributes 2 m3 equivalent.
    assert float(result["pfv_model_error_applied_m3"][0]) == 30.0
    assert float(result["pfv_risk_m3"][0]) == 55.0
    assert float(result["pfv_soft_excess_m3"][0]) == 5.0
    assert float(result["movement_penalty_m3_equivalent"][0]) == 2.0
    assert float(result["score_m3_equivalent"][0]) == -93.0
