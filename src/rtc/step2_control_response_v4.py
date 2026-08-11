"""Isolated V4 counterfactual response model.

The model is intentionally not wired into ``rtc-train-step2-large``.  It separates an
absolute reference trajectory from a structurally zero-at-zero action-effect operator.
The action effect is a causal multi-horizon response: a setting change at step ``t`` can
affect ``t`` and later outputs, but never an earlier output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .step2_action_pathway_audit_v4 import direct_pair_delta_tfv


@dataclass(frozen=True)
class CounterfactualResponseV4:
    reference_states: torch.Tensor
    candidate_states: torch.Tensor
    delta_states: torch.Tensor
    reference_flows: torch.Tensor
    candidate_flows: torch.Tensor
    delta_flows: torch.Tensor
    reference_flood_rate: torch.Tensor
    candidate_flood_rate: torch.Tensor


def _scatter_directed(
    values: torch.Tensor,
    upstream: torch.Tensor,
    downstream: torch.Tensor,
    node_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter [B,H,A] actuator values to directed upstream/downstream node channels."""

    b, h, a = values.shape
    flat = values.reshape(b * h, a, 1)
    out = torch.zeros(b * h, node_count, 1, device=values.device, dtype=values.dtype)
    inc = torch.zeros_like(out)
    out = out.index_add(1, upstream, flat)
    inc = inc.index_add(1, downstream, flat)
    return out.reshape(b, h, node_count, 1), inc.reshape(b, h, node_count, 1)


def expand_control_blocks(block_settings: torch.Tensor, *, model_steps_per_block: int = 2) -> torch.Tensor:
    """Expand each 10-minute command into two 5-minute model-step commands."""

    if block_settings.dim() != 3:
        raise ValueError("block_settings must be [B,blocks,actuators]")
    if model_steps_per_block <= 0:
        raise ValueError("model_steps_per_block must be positive")
    return block_settings.repeat_interleave(int(model_steps_per_block), dim=1)


class DifferentiableCounterfactualResponseModelV4(nn.Module):
    """Reference-plus-effect model with actuator identity and directed scatter."""

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        node_static_dim: int,
        actuator_physics_dim: int,
        hidden_dim: int = 160,
        actuator_count: int = 109,
        actuator_embedding_dim: int = 16,
        state_effect_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.rainfall_dim = int(rainfall_dim)
        self.node_static_dim = int(node_static_dim)
        self.actuator_physics_dim = int(actuator_physics_dim)
        self.actuator_count = int(actuator_count)
        self.actuator_embedding_dim = int(actuator_embedding_dim)
        scale = torch.ones(self.state_dim) if state_effect_scale is None else torch.as_tensor(state_effect_scale, dtype=torch.float32)
        if scale.numel() != self.state_dim:
            raise ValueError("state_effect_scale must contain one value per state channel")
        self.register_buffer("state_effect_scale", scale.reshape(self.state_dim).clamp_min(1e-6))
        self.actuator_identity = nn.Embedding(self.actuator_count, self.actuator_embedding_dim)

        reference_node_in = self.state_dim + self.rainfall_dim + self.node_static_dim + 2
        self.reference_state_encoder = nn.Sequential(
            nn.Linear(reference_node_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.reference_state_head = nn.Linear(hidden_dim, self.state_dim)

        token_in = 3 + self.actuator_physics_dim + self.actuator_embedding_dim
        self.reference_flow_encoder = nn.Sequential(
            nn.Linear(1 + self.actuator_physics_dim + self.actuator_embedding_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.reference_flow_head = nn.Linear(hidden_dim, 1)
        self.effect_flow_encoder = nn.Sequential(
            nn.Linear(token_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.effect_flow_head = nn.Linear(hidden_dim, 1)

        effect_node_in = self.node_static_dim + 3
        self.effect_state_encoder = nn.Sequential(
            nn.Linear(effect_node_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.effect_state_head = nn.Linear(hidden_dim, 2 * self.state_dim)

    def _tokens(
        self,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        actuator_physics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, a = candidate_settings.shape
        if a != self.actuator_count:
            raise ValueError("candidate settings actuator dimension does not match model")
        ids = torch.arange(a, device=candidate_settings.device)
        identity = self.actuator_identity(ids).reshape(1, 1, a, -1).expand(b, h, -1, -1)
        physics = actuator_physics.to(dtype=candidate_settings.dtype).reshape(1, 1, a, -1).expand(b, h, -1, -1)
        ref = reference_settings[..., None]
        cand = candidate_settings[..., None]
        delta = (candidate_settings - reference_settings)[..., None]
        effect_input = torch.cat([ref, cand, delta, physics, identity], dim=-1)
        ref_input = torch.cat([ref, physics, identity], dim=-1)
        return self.reference_flow_encoder(ref_input), self.effect_flow_encoder(effect_input), delta.squeeze(-1)

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        static_node_features: torch.Tensor,
        actuator_physics: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> CounterfactualResponseV4:
        if edge_index is None or edge_index.shape[0] != 2:
            raise ValueError("V4 requires a [2,E] graph edge_index for directed response propagation")
        if rainfall.dim() != 4:
            raise ValueError("rainfall must be [B,H,N,R]")
        b, h, n, _ = rainfall.shape
        if reference_settings.shape != candidate_settings.shape:
            raise ValueError("reference/candidate setting shapes differ")
        if initial_state.shape[:2] != (b, n):
            raise ValueError("initial state does not match rainfall batch/node dimensions")
        up = actuator_upstream.long().to(device=initial_state.device)
        down = actuator_downstream.long().to(device=initial_state.device)
        static = static_node_features.to(device=initial_state.device, dtype=initial_state.dtype)
        physics = actuator_physics.to(device=initial_state.device, dtype=initial_state.dtype)

        ref_out, ref_in = _scatter_directed(reference_settings, up, down, n)
        reference_action = torch.cat([ref_out, ref_in], dim=-1)
        static_h = static.reshape(1, 1, n, -1).expand(b, h, -1, -1)
        initial_h = initial_state.reshape(b, 1, n, -1).expand(-1, h, -1, -1)
        rain = rainfall.to(dtype=initial_state.dtype)
        ref_input = torch.cat([initial_h, rain, static_h, reference_action], dim=-1)
        base_hidden = self.reference_state_encoder(ref_input)
        base_step = torch.tanh(self.reference_state_head(base_hidden)) * self.state_effect_scale.to(initial_state.dtype)
        reference_states = initial_h + torch.cumsum(base_step, dim=1)

        ref_token, effect_token, delta_u = self._tokens(reference_settings, candidate_settings, physics)
        reference_flow_step = torch.tanh(self.reference_flow_head(ref_token).squeeze(-1))
        reference_flows = previous_actuator_flow.reshape(b, 1, -1) + torch.cumsum(reference_flow_step, dim=1)

        local_effect = delta_u * self.effect_flow_head(effect_token).squeeze(-1)
        delta_flows = torch.cumsum(local_effect, dim=1)
        candidate_flows = reference_flows + delta_flows

        effect_out, effect_in = _scatter_directed(delta_flows, up, down, n)
        edge_src = edge_index[0].long().to(device=initial_state.device)
        edge_dst = edge_index[1].long().to(device=initial_state.device)
        flat_signal = (effect_out - effect_in).reshape(b * h, n, 1)
        propagated = torch.zeros_like(flat_signal).index_add(1, edge_dst, flat_signal[:, edge_src])
        degree = torch.bincount(edge_dst, minlength=n).clamp_min(1).to(dtype=flat_signal.dtype, device=flat_signal.device)
        propagated = (propagated / degree.reshape(1, n, 1)).reshape(b, h, n, 1)
        effect_node_input = torch.cat([effect_out, effect_in, propagated, static_h], dim=-1)
        effect_gain = torch.tanh(self.effect_state_head(self.effect_state_encoder(effect_node_input)))
        effect_gain = effect_gain.reshape(b, h, n, 2, self.state_dim)
        delta_states = effect_out * effect_gain[:, :, :, 0, :] + effect_in * effect_gain[:, :, :, 1, :]
        candidate_states = reference_states + delta_states
        return CounterfactualResponseV4(
            reference_states=reference_states,
            candidate_states=candidate_states,
            delta_states=delta_states,
            reference_flows=reference_flows,
            candidate_flows=candidate_flows,
            delta_flows=delta_flows,
            reference_flood_rate=reference_states[..., 2],
            candidate_flood_rate=candidate_states[..., 2],
        )


__all__ = [
    "CounterfactualResponseV4",
    "DifferentiableCounterfactualResponseModelV4",
    "direct_pair_delta_tfv",
    "expand_control_blocks",
]
