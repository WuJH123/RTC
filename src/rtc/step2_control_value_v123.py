"""Dual-volume Value interface for Project7 V12.3.

The first V12.3 implementation intentionally reuses the scientifically understood V7
scalar Value architecture for each target rather than inventing a larger multi-task
network before PFV identifiability is measured.  TFV and PFV models can therefore be
trained/evaluated separately on the same D2/D3 groups and only combined by Step3 after
both evidence reports are available.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import ControlValueSurrogateV70

V123_DUAL_VALUE_CONTRACT = "PROJECT7_V123_SEPARATE_TFV_PFV_DIRECT_VALUE_V1"


@dataclass(frozen=True)
class DualVolumeValueOutputV123:
    delta_tfv_m3: torch.Tensor
    delta_pfv_m3: torch.Tensor


class DualVolumeValueV123(nn.Module):
    """Expose separately trained TFV/PFV scalar surrogates through one runtime API."""

    def __init__(
        self,
        *,
        tfv_model: ControlValueSurrogateV70,
        pfv_model: ControlValueSurrogateV70,
    ) -> None:
        super().__init__()
        if tfv_model is pfv_model:
            raise ValueError("V123 TFV and PFV models must be separately parameterised")
        self.tfv_model = tfv_model
        self.pfv_model = pfv_model

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV60,
    ) -> DualVolumeValueOutputV123:
        tfv = self.tfv_model(
            initial_state,
            rainfall,
            reference_settings,
            candidate_settings,
            previous_actuator_flow,
            prepared,
        ).delta_tfv_m3
        pfv = self.pfv_model(
            initial_state,
            rainfall,
            reference_settings,
            candidate_settings,
            previous_actuator_flow,
            prepared,
        ).delta_tfv_m3
        if tfv.shape != pfv.shape:
            raise RuntimeError("V123 TFV/PFV Value outputs do not align")
        if not bool(torch.isfinite(tfv).all()) or not bool(torch.isfinite(pfv).all()):
            raise RuntimeError("V123 TFV/PFV Value output is non-finite")
        return DualVolumeValueOutputV123(delta_tfv_m3=tfv, delta_pfv_m3=pfv)


__all__ = [
    "DualVolumeValueOutputV123",
    "DualVolumeValueV123",
    "V123_DUAL_VALUE_CONTRACT",
]
