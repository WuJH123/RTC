"""Frozen V11.1 zero-anchored signed hydraulic-effect contract."""
from __future__ import annotations

from dataclasses import dataclass

from .step2_v110_contract import ActuatorSetHydraulicContractV110, HydraulicHorizonV110

V111_CONTRACT = "PROJECT7_STEP2_V111_ZERO_ANCHORED_SIGNED_EFFECT_V1"
V111_CURRENT_STEP2_CONTRACT = "PROJECT7_STEP2_CURRENT_V111_V1"


@dataclass(frozen=True)
class V111LossContract:
    """Preregistered V11.1 objective coefficients.

    The direct signed effect is the primary objective.  The three-head
    decomposition remains diagnostic/auxiliary and cannot dominate it.
    """

    direct_weight: float = 1.0
    temporal_weight: float = 0.10
    active_aux_weight: float = 0.10
    sign_aux_weight: float = 0.05
    magnitude_aux_weight: float = 0.05
    local_support_min: int = 8
    tiny_steps: int = 100
    micro_groups: int = 12
    canonical_max_epochs: int = 12
    canonical_patience: int = 3
    # V111 uses the already-established FP32 AdamW scale from the supported
    # V7 training contract.  This is a single preregistered value, not a
    # holdout-tuned sweep; the gradient audit showed 2e-4 left the direct head
    # effectively at its zero anchor over the fixed mechanism schedule.
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        expected = {
            "direct_weight": 1.0,
            "temporal_weight": 0.10,
            "active_aux_weight": 0.10,
            "sign_aux_weight": 0.05,
            "magnitude_aux_weight": 0.05,
            "local_support_min": 8,
            "tiny_steps": 100,
            "micro_groups": 12,
            "canonical_max_epochs": 12,
            "canonical_patience": 3,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "grad_clip": 1.0,
            "seed": 42,
        }
        bad = [key for key, value in expected.items() if getattr(self, key) != value]
        if bad:
            raise ValueError(f"V111 frozen loss/schedule values changed: {bad}")


__all__ = [
    "ActuatorSetHydraulicContractV110",
    "HydraulicHorizonV110",
    "V111LossContract",
    "V111_CONTRACT",
    "V111_CURRENT_STEP2_CONTRACT",
]
