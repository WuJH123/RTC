"""Canonical Project7 V127 scientific development surface.

V127 learns differentiable hydraulic dynamics under continuous actions, validates the
resulting control gradients against authoritative SWMM evidence, then optimizes all 109
writable targets with bounded continuous MPC. Sparse-RBC is only a warm start / safety
fallback and never the Step2 reference or action-space ceiling.
"""
from __future__ import annotations

from .d5_gradient_v127 import V127_D5_CONTRACT
from .step2_differentiable_v127 import (
    ControlOrientedDifferentiableSurrogateV127,
    V127_STEP2_CONTRACT,
)
from .step3_mpc_v127 import DifferentiableRollingMPCV127, V127_STEP3_CONTRACT

CURRENT_PROJECT7_CONTRACT = "PROJECT7_V127_CONTINUOUS_DIFFERENTIABLE_MPC_CORRECTNESS_V2"
CURRENT_STEP2_CONTRACT = V127_STEP2_CONTRACT
CURRENT_STEP3_CONTRACT = V127_STEP3_CONTRACT
CURRENT_D5_CONTRACT = V127_D5_CONTRACT
CURRENT_STEP2_CLASS = ControlOrientedDifferentiableSurrogateV127
CURRENT_POLICY_CLASS = DifferentiableRollingMPCV127
HYDRAULIC_MODEL_REQUIRED_ONLINE = True
CONTINUOUS_MPC_SCIENTIFIC_TARGET = True
# Runtime requires finite, causal, same-checkpoint scientific evidence. Empirical quality
# scores are reported/interpreted; no author-chosen rank/cosine threshold disables the
# continuous method.
CONTINUOUS_MPC_RUNTIME_REQUIRES_GATE = True
RBC_ROLE = "WARM_START_AND_SAFETY_FALLBACK_ONLY"
RBC_IS_VALUE_REFERENCE = False
RBC_IS_ACTION_SPACE_CEILING = False
CONTINUOUS_MPC_ENABLED = True

__all__ = [
    "CONTINUOUS_MPC_ENABLED",
    "CONTINUOUS_MPC_RUNTIME_REQUIRES_GATE",
    "CONTINUOUS_MPC_SCIENTIFIC_TARGET",
    "CURRENT_D5_CONTRACT",
    "CURRENT_POLICY_CLASS",
    "CURRENT_PROJECT7_CONTRACT",
    "CURRENT_STEP2_CLASS",
    "CURRENT_STEP2_CONTRACT",
    "CURRENT_STEP3_CONTRACT",
    "HYDRAULIC_MODEL_REQUIRED_ONLINE",
    "RBC_IS_ACTION_SPACE_CEILING",
    "RBC_IS_VALUE_REFERENCE",
    "RBC_ROLE",
]
