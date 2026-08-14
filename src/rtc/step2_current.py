"""Canonical Project7 Step2/Step3 surface after the V125 action-support redesign.

Historical V7/V120/V123 classes remain importable for reproducibility.  The current
production *decision* contract is V125: causal sparse state, finite signed TFV/PFV Value,
Sparse-RBC anchor by default, evidence-gated learned override every 10 minutes.
"""
from __future__ import annotations

from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_d4_action_support_v125 import D4_ACTION_SUPPORT_CONTRACT_V125
from .step2_policy_v125 import AnchorOverridePolicyV125, V125_POLICY_CONTRACT

VALUE_MODEL_VERSION = "V124_OR_ACCEPTED_V70_CAUSAL"
HYDRAULIC_MODEL_VERSION = "HISTORICAL_V8_TO_V113_ONLY"
HYDRAULIC_MODEL_REQUIRED_ONLINE = False
CURRENT_STEP2_CONTRACT = V125_POLICY_CONTRACT
CURRENT_D4_CONTRACT = D4_ACTION_SUPPORT_CONTRACT_V125
CURRENT_POLICY_CLASS = AnchorOverridePolicyV125
CONTINUOUS_MPC_ENABLED = False

__all__ = [
    "AnchorOverridePolicyV125",
    "ControlValueSurrogateV70",
    "CONTINUOUS_MPC_ENABLED",
    "CURRENT_D4_CONTRACT",
    "CURRENT_POLICY_CLASS",
    "CURRENT_STEP2_CONTRACT",
    "HYDRAULIC_MODEL_REQUIRED_ONLINE",
    "HYDRAULIC_MODEL_VERSION",
    "VALUE_MODEL_VERSION",
]
