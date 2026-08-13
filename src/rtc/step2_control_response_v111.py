"""V11.1 direct signed hydraulic-effect decoder.

The V11 relation encoder, causal action prefix, frozen V7 reference and
nonlocal attention are intentionally reused.  Only the physical decoder is
revised: the primary response is a zero-anchored direct signed head rather
than an active*sign*magnitude product.  The latter heads remain available for
interpretation and auxiliary supervision.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import (
    ActuatorSetHydraulicResponseV110,
    HydraulicEffectOutputV110,
)
from .step2_v110_contract import ActuatorSetHydraulicContractV110


def zero_anchored_direct_head(head: nn.Linear, *, weight_value: float = 1.0e-4) -> None:
    """Initialize a direct head near the exact zero-effect baseline.

    A tiny non-zero weight preserves an action gradient at initialization;
    zero bias prevents a domain-wide offset on inactive cells.
    """
    if not isinstance(head, nn.Linear):
        raise TypeError("V111 direct head must be nn.Linear")
    if weight_value <= 0.0 or weight_value > 1.0e-2:
        raise ValueError("V111 direct-head initialization is outside the frozen range")
    with torch.no_grad():
        head.weight.fill_(float(weight_value))
        if head.bias is not None:
            head.bias.zero_()


class ActuatorSetHydraulicResponseV111(ActuatorSetHydraulicResponseV110):
    """V11 model with direct, signed, zero-anchored physical outputs."""

    def __init__(
        self,
        *,
        reference_model: nn.Module,
        state_magnitude_scale: torch.Tensor | np.ndarray,
        flow_magnitude_scale: torch.Tensor | np.ndarray,
        node_static_dim: int,
        physics_dim: int,
        rainfall_dim: int,
        actuator_count: int,
        node_count: int,
        relations,
        horizon=None,
        contract: ActuatorSetHydraulicContractV110 = ActuatorSetHydraulicContractV110(),
    ) -> None:
        super().__init__(
            reference_model=reference_model,
            state_magnitude_scale=state_magnitude_scale,
            flow_magnitude_scale=flow_magnitude_scale,
            node_static_dim=node_static_dim,
            physics_dim=physics_dim,
            rainfall_dim=rainfall_dim,
            actuator_count=actuator_count,
            node_count=node_count,
            relations=relations,
            horizon=horizon if horizon is not None else __import__(
                "rtc.step2_v110_contract", fromlist=["HydraulicHorizonV110"]
            ).HydraulicHorizonV110(),
            contract=contract,
        )
        self.state_direct_effect_head = nn.Linear(self.hidden_dim, 5)
        self.flow_direct_effect_head = nn.Linear(self.hidden_dim, 1)
        zero_anchored_direct_head(self.state_direct_effect_head)
        zero_anchored_direct_head(self.flow_direct_effect_head)
        self._last_node_hidden: torch.Tensor | None = None
        self._last_flow_hidden: torch.Tensor | None = None
        self._node_hook = self.node_decoder.register_forward_hook(self._capture_node_hidden)
        self._flow_hook = self.flow_decoder.register_forward_hook(self._capture_flow_hidden)

    def _capture_node_hidden(self, _module, _inputs, output):
        self._last_node_hidden = output

    def _capture_flow_hidden(self, _module, _inputs, output):
        self._last_flow_hidden = output

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV60,
    ) -> HydraulicEffectOutputV110:
        self._last_node_hidden = None
        self._last_flow_hidden = None
        base = super().forward(
            initial_state,
            rainfall,
            reference_settings,
            candidate_settings,
            previous_actuator_flow,
            prepared,
        )
        if self._last_node_hidden is None or self._last_flow_hidden is None:
            raise RuntimeError("V111 decoder hooks did not capture hidden representations")

        state_norm = self.state_direct_effect_head(self._last_node_hidden)
        flow_norm = self.flow_direct_effect_head(self._last_flow_hidden).squeeze(-1)
        state_scale = self.state_magnitude_scale.to(state_norm)[None, None, None]
        flow_scale = self.flow_magnitude_scale.to(flow_norm)[None, None, None]
        unique_delta = state_norm * state_scale
        flow_delta = flow_norm * flow_scale
        any_active = base.changed_actuator_mask.any(dim=-1)
        unique_delta = torch.where(
            any_active[..., None, None], unique_delta, torch.zeros_like(unique_delta)
        )
        flow_delta = torch.where(
            any_active[..., None], flow_delta, torch.zeros_like(flow_delta)
        )
        depth = unique_delta[..., 0]
        raw_states = torch.stack(
            (depth, depth, unique_delta[..., 1], unique_delta[..., 2],
             unique_delta[..., 3], unique_delta[..., 4]), dim=-1
        )
        reference = base.reference_states_physical[:, None].expand_as(raw_states)
        candidate_raw = reference + raw_states
        invert = prepared.invert_elevation_m.to(candidate_raw)
        while invert.ndim < candidate_raw[..., 0].ndim:
            invert = invert.unsqueeze(0)
        projected_depth = candidate_raw[..., 0].clamp_min(0.0)
        projected = torch.stack(
            (projected_depth, projected_depth + invert,
             candidate_raw[..., 2].clamp_min(0.0),
             candidate_raw[..., 3].clamp_min(0.0),
             candidate_raw[..., 4].clamp_min(0.0),
             candidate_raw[..., 5].clamp_min(0.0)), dim=-1
        )
        ref_flow = base.reference_flows_physical[:, None].expand_as(flow_delta)
        return replace(
            base,
            raw_delta_states_physical=raw_states,
            raw_delta_flows_physical=flow_delta,
            candidate_states_projected_physical=projected,
            candidate_flows_projected_physical=ref_flow + flow_delta,
        )


HydraulicEffectOutputV111 = HydraulicEffectOutputV110


__all__ = [
    "ActuatorSetHydraulicResponseV111",
    "HydraulicEffectOutputV111",
    "zero_anchored_direct_head",
]
