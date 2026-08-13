"""Canonical Step2 surface for the simplified Project7 V120 policy."""
from __future__ import annotations

from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_v120_contract import V120_CONTRACT

VALUE_MODEL_VERSION = "V7"
HYDRAULIC_MODEL_VERSION = "V11"
CURRENT_STEP2_CONTRACT = V120_CONTRACT

__all__ = [
    "ControlValueSurrogateV70",
    "CURRENT_STEP2_CONTRACT",
    "HYDRAULIC_MODEL_VERSION",
    "VALUE_MODEL_VERSION",
]
