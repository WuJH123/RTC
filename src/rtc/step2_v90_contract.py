"""Frozen development contract for Project7 Step2 V9 decision experiments."""
from __future__ import annotations

from dataclasses import dataclass

from .step2_v80_contract import DirectHydraulicEffectLossContractV80

V90_CONTRACT = "PROJECT7_STEP2_V90_TRAJECTORY_CONDITIONED_SIGNED_EFFECT_V1"

LEVEL_A = "A_BOUNDARY"
LEVEL_B = "B_PREDICTED_REFERENCE_TRAJECTORY"
LEVEL_C = "C_ORACLE_REFERENCE_TRAJECTORY"
STATE_SUFFICIENCY_LEVELS = (LEVEL_A, LEVEL_B, LEVEL_C)


@dataclass(frozen=True)
class DirectHydraulicEffectLossContractV90(DirectHydraulicEffectLossContractV80):
    """No sweep: the requested fixed V9 development schedule and mixture."""

    d2_pretrain_epochs: int = 4
    d3_joint_epochs: int = 10
    d3_weight: float = 0.75
    d2_anchor_weight: float = 0.25

    def validate(self) -> None:
        super().validate()
        if self.d2_pretrain_epochs != 4 or self.d3_joint_epochs != 10:
            raise ValueError("V9 epoch schedule is frozen at D2=4 and D3=10")
        if abs(self.d3_weight - 0.75) > 1e-12 or abs(self.d2_anchor_weight - 0.25) > 1e-12:
            raise ValueError("V9 D3/D2 mixture is frozen at 0.75/0.25")


def validate_conditioning_level_v90(level: str) -> str:
    value = str(level).upper()
    if value not in STATE_SUFFICIENCY_LEVELS:
        raise ValueError(f"unknown V9 state-sufficiency level: {level}")
    return value


__all__ = [
    "DirectHydraulicEffectLossContractV90",
    "LEVEL_A",
    "LEVEL_B",
    "LEVEL_C",
    "STATE_SUFFICIENCY_LEVELS",
    "V90_CONTRACT",
    "validate_conditioning_level_v90",
]
