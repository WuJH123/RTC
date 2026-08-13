"""Project7 Step2 V7.0 contracts: direct control-objective value + effect-aware hydraulics."""
from __future__ import annotations

from dataclasses import dataclass

V70_CONTRACT = "PROJECT7_STEP2_V70_DIRECT_CONTROL_OBJECTIVE_V1"
V70_VALUE_CONTRACT = "PROJECT7_V70_DIRECT_DELTA_TFV_VALUE_V1"
V70_HYDRAULIC_CONTRACT = "PROJECT7_V70_EFFECT_AWARE_HYDRAULIC_V1"


@dataclass(frozen=True)
class DirectValueLossContractV70:
    """Magnitude is primary; ordering is auxiliary and scale-aware."""
    transformed_magnitude_weight: float = 1.5
    physical_magnitude_weight: float = 1.0
    pairwise_difference_weight: float = 0.75
    pairwise_sign_weight: float = 0.15
    pair_min_effect_fraction: float = 0.02
    d2_pretrain_epochs: int = 4
    joint_epochs: int = 12
    joint_d3_weight: float = 0.75
    joint_d2_weight: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    transformed_limit: float = 5.0

    def validate(self) -> None:
        weights = (
            self.transformed_magnitude_weight,
            self.physical_magnitude_weight,
            self.pairwise_difference_weight,
            self.pairwise_sign_weight,
        )
        if any(v < 0 for v in weights):
            raise ValueError("V7 value loss weights must be non-negative")
        if not 0.0 <= self.pair_min_effect_fraction < 1.0:
            raise ValueError("invalid V7 pair threshold")
        if abs(self.joint_d3_weight + self.joint_d2_weight - 1.0) > 1e-9:
            raise ValueError("V7 joint D2/D3 weights must sum to one")
        if min(self.d2_pretrain_epochs, self.joint_epochs) <= 0:
            raise ValueError("V7 epochs must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.transformed_limit <= 0:
            raise ValueError("invalid V7 optimization contract")


@dataclass(frozen=True)
class HydraulicEffectLossContractV70:
    """Train the reference trajectory and, separately, the control-induced delta."""
    absolute_state_weight: float = 0.50
    delta_state_weight: float = 1.00
    absolute_flow_weight: float = 0.35
    delta_flow_weight: float = 0.75
    onset_transition_weight: float = 0.50
    onset_gamma: float = 2.0
    onset_max_pos_weight: float = 50.0
    onset_epsilon_m3s: float = 1e-7
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    def validate(self) -> None:
        values = (
            self.absolute_state_weight,
            self.delta_state_weight,
            self.absolute_flow_weight,
            self.delta_flow_weight,
            self.onset_transition_weight,
        )
        if any(v < 0 for v in values):
            raise ValueError("V7 hydraulic weights must be non-negative")
        if self.onset_gamma < 0 or self.onset_max_pos_weight < 1:
            raise ValueError("invalid onset imbalance contract")
        if self.onset_epsilon_m3s < 0 or self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("invalid V7 hydraulic optimization contract")


__all__ = [
    "DirectValueLossContractV70",
    "HydraulicEffectLossContractV70",
    "V70_CONTRACT",
    "V70_HYDRAULIC_CONTRACT",
    "V70_VALUE_CONTRACT",
]
