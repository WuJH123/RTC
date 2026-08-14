"""Canonical Project7 Step2/Step3 development surface after the V125 D4 diagnosis.

V126 changes training semantics, not the accepted online engineering shell. Broad D2 and
targeted-D3 data learn the causal action representation first; D4-FIT then receives an
isolated anchor-advantage fine-tune. The runtime selector remains the corrected V125
Sparse-RBC-anchor policy until V126 development evidence is accepted.
"""
from __future__ import annotations

from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_curriculum_v126 import V126_CURRICULUM_CONTRACT
from .step2_d4_action_support_v125 import D4_ACTION_SUPPORT_CONTRACT_V125
from .step2_policy_v125 import AnchorOverridePolicyV125, V125_POLICY_CONTRACT

VALUE_MODEL_VERSION = "V124_INTERACTION_BACKBONE_V126_SOURCE_AWARE_TRAINING"
HYDRAULIC_MODEL_VERSION = "HISTORICAL_V8_TO_V113_ONLY"
HYDRAULIC_MODEL_REQUIRED_ONLINE = False
CURRENT_STEP2_CONTRACT = V126_CURRICULUM_CONTRACT
CURRENT_RUNTIME_POLICY_CONTRACT = V125_POLICY_CONTRACT
CURRENT_D4_CONTRACT = D4_ACTION_SUPPORT_CONTRACT_V125
CURRENT_POLICY_CLASS = AnchorOverridePolicyV125
CONTINUOUS_MPC_ENABLED = False

__all__ = [
    "AnchorOverridePolicyV125",
    "ControlValueSurrogateV70",
    "CONTINUOUS_MPC_ENABLED",
    "CURRENT_D4_CONTRACT",
    "CURRENT_POLICY_CLASS",
    "CURRENT_RUNTIME_POLICY_CONTRACT",
    "CURRENT_STEP2_CONTRACT",
    "HYDRAULIC_MODEL_REQUIRED_ONLINE",
    "HYDRAULIC_MODEL_VERSION",
    "VALUE_MODEL_VERSION",
]
