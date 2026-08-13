"""Single canonical Step2 import surface for Project7.

The final simplified Step2 is control-oriented: it predicts signed authoritative
whole-system Delta-TFV for joint actions. Historical Hydraulic-effect models are
kept in git for research audit only and are not part of the production contract.
"""
from __future__ import annotations

from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_v120_contract import V120_CONTRACT

VALUE_MODEL_VERSION = "V7_DIRECT_VALUE"
HYDRAULIC_MODEL_VERSION = "NOT_REQUIRED_ONLINE"
CURRENT_STEP2_CONTRACT = V120_CONTRACT

__all__ = [
    "ControlValueSurrogateV70",
    "CURRENT_STEP2_CONTRACT",
    "HYDRAULIC_MODEL_VERSION",
    "VALUE_MODEL_VERSION",
]
