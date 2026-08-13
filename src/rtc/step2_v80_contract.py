"""Frozen development contract for Project7 Step2 V8.0.

V8 keeps the successful V7 direct Delta-TFV value model unchanged and replaces only
Hydraulic candidate-minus-reference learning.  The primary hydraulic target is the
signed counterfactual effect, not a difference between two independently decoded
absolute trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass

V80_CONTRACT = "PROJECT7_STEP2_V80_DIRECT_HYDRAULIC_EFFECT_V1"
V80_HYDRAULIC_CONTRACT = "PROJECT7_V80_DIRECT_COUNTERFACTUAL_HYDRAULIC_SURROGATE_V1"


@dataclass(frozen=True)
class DirectHydraulicEffectLossContractV80:
    # Stage semantics: first learn single-actuator spatial response, then multi-actuator response.
    d2_pretrain_epochs: int = 3
    d3_joint_epochs: int = 8
    d3_weight: float = 0.75
    d2_anchor_weight: float = 0.25

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    # Sparse-effect objective. Dense loss is only a regularizer; active-effect cells are primary.
    dense_state_weight: float = 0.25
    active_state_weight: float = 1.0
    magnitude_state_weight: float = 0.50
    sign_state_weight: float = 0.10
    dense_flow_weight: float = 0.25
    active_flow_weight: float = 0.75
    magnitude_flow_weight: float = 0.35
    sign_flow_weight: float = 0.05
    onset_weight: float = 0.20

    # An element is informative when its true effect reaches this fraction of its TrainFit-only
    # RMS delta scale. This is fixed by contract, not selected on Holdout.
    active_effect_fraction: float = 0.25
    smooth_l1_beta: float = 0.5

    onset_epsilon_m3s: float = 1e-7
    onset_gamma: float = 2.0
    onset_max_sqrt_pos_weight: float = 20.0
    onset_threshold_bins: int = 401
    onset_logit_min: float = -12.0
    onset_logit_max: float = 12.0

    graph_blocks: int = 4
    hidden_dim: int = 64
    time_embedding_dim: int = 12

    def validate(self) -> None:
        if self.d2_pretrain_epochs <= 0 or self.d3_joint_epochs <= 0:
            raise ValueError("V8 hydraulic stages require positive epochs")
        if abs(self.d3_weight + self.d2_anchor_weight - 1.0) > 1e-9:
            raise ValueError("V8 D2/D3 source weights must sum to one")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid V8 optimizer contract")
        weights = (
            self.dense_state_weight,
            self.active_state_weight,
            self.magnitude_state_weight,
            self.sign_state_weight,
            self.dense_flow_weight,
            self.active_flow_weight,
            self.magnitude_flow_weight,
            self.sign_flow_weight,
            self.onset_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("V8 loss weights must be non-negative")
        if not 0 < self.active_effect_fraction < 1:
            raise ValueError("active_effect_fraction must lie in (0,1)")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if self.onset_epsilon_m3s < 0 or self.onset_gamma < 0:
            raise ValueError("invalid onset contract")
        if self.onset_max_sqrt_pos_weight < 1:
            raise ValueError("onset weight cap must be >= 1")
        if self.onset_threshold_bins < 21:
            raise ValueError("onset threshold grid is too small")
        if self.onset_logit_max <= self.onset_logit_min:
            raise ValueError("invalid onset logit range")
        if self.graph_blocks <= 0 or self.hidden_dim <= 0 or self.time_embedding_dim <= 0:
            raise ValueError("invalid V8 architecture dimensions")


__all__ = [
    "DirectHydraulicEffectLossContractV80",
    "V80_CONTRACT",
    "V80_HYDRAULIC_CONTRACT",
]
