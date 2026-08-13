"""Final simplified Project7 Step2 contract.

Step2 is a control-oriented value learner, not a full hydraulic digital twin.
It predicts the signed authoritative change in whole-system cumulative TFV for
joint actuator sequences. Nodewise counterfactual hydraulic effects remain
optional diagnostics and never gate the production policy.
"""
from __future__ import annotations

from dataclasses import dataclass

V120_CONTRACT = "PROJECT7_STEP2_V120_TFV_VALUE_ONLY_V1"
V120_BUNDLE_CONTRACT = "PROJECT7_STEP2_V120_TFV_VALUE_ONLY_BUNDLE_V1"


@dataclass(frozen=True)
class Step2V120Contract:
    actuator_count: int = 109
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    value_horizon_minutes: int = 360
    control_blocks: int = 36
    candidates_per_bank: int = 24
    candidate_banks: int = 4
    seed: int = 42
    primary_objective: str = "whole_system_cumulative_TFV_m3"
    step2_target: str = "direct_signed_authoritative_delta_TFV_m3"
    hydraulic_model_required_online: bool = False
    nodewise_action_effect_required: bool = False
    sum_d2_effects_for_joint_action: bool = False
    execute_first_move_only: bool = True

    def validate(self) -> None:
        if self.actuator_count != 109:
            raise ValueError("V120 requires the frozen 109-actuator testbed")
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V120 frozen clock is 300-s model / 600-s control")
        if self.value_horizon_minutes != 360 or self.control_blocks != 36:
            raise ValueError("V120 direct value horizon must remain 360 min / 36 blocks")
        if self.candidates_per_bank != 24 or self.candidate_banks < 1:
            raise ValueError("V120 candidate policy must retain D3-v2 24-candidate banks")
        if self.seed != 42:
            raise ValueError("V120 scientific seed drift")
        if self.primary_objective != "whole_system_cumulative_TFV_m3":
            raise ValueError("V120 primary objective drift")
        if self.step2_target != "direct_signed_authoritative_delta_TFV_m3":
            raise ValueError("V120 Step2 target drift")
        if self.hydraulic_model_required_online or self.nodewise_action_effect_required:
            raise ValueError("V120 must not re-introduce nodewise Hydraulic as a production gate")
        if self.sum_d2_effects_for_joint_action:
            raise ValueError("V120 forbids SUM-D2 joint-action reconstruction")
        if not self.execute_first_move_only:
            raise ValueError("V120 must remain receding-horizon first-move control")


__all__ = ["Step2V120Contract", "V120_BUNDLE_CONTRACT", "V120_CONTRACT"]
