from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class GraphMessageBlock(nn.Module):
    """Small dependency-free directed message-passing block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], edge_index: [2, E]
        src, dst = edge_index.long()
        msg = self.message(torch.cat([x[:, src], x[:, dst]], dim=-1))
        agg = torch.zeros_like(x).index_add(1, dst, msg)
        degree = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        degree = degree.index_add(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0).view(1, -1, 1)
        delta = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(x + delta)


class SparseStateEstimator(nn.Module):
    """Step 1: reconstruct network state from causal sparse histories.

    Missing observations are represented explicitly by a mask; the model never receives
    future truth. A GRU extracts temporal information independently at every node before
    topology-aware message passing reconstructs the network-wide state.
    """

    def __init__(
        self,
        observed_dim: int,
        static_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        graph_layers: int = 3,
    ):
        super().__init__()
        self.temporal = nn.GRU(observed_dim * 2 + static_dim, hidden_dim, batch_first=True)
        self.graph = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(graph_layers))
        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        observed_history: torch.Tensor,
        observation_mask: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # histories: [B, T, N, F]; static: [N, S] or [B, N, S]
        b, t, n, _ = observed_history.shape
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(b, -1, -1)
        static = static_node_features[:, None].expand(-1, t, -1, -1)
        temporal_input = torch.cat(
            [observed_history * observation_mask, observation_mask, static], dim=-1
        )
        # GRU over time for each node.
        node_sequences = temporal_input.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, hidden = self.temporal(node_sequences)
        x = hidden[-1].reshape(b, n, -1)
        for block in self.graph:
            x = block(x, edge_index)
        return self.head(x)


class ActuatorFlowModel(nn.Module):
    """Step 2A: continuous setting + local hydraulics -> actuator flow.

    The responsiveness probability is a soft hurdle, not a hard actuator-selection gate.
    """

    def __init__(self, state_dim: int, physics_dim: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = state_dim * 2 + physics_dim + 2  # upstream/downstream, setting, previous flow
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.response_logit = nn.Linear(hidden_dim, 1)
        self.flow_delta = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        upstream_state: torch.Tensor,
        downstream_state: torch.Tensor,
        setting: torch.Tensor,
        previous_flow: torch.Tensor,
        physics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat(
            [upstream_state, downstream_state, setting[..., None], previous_flow[..., None], physics],
            dim=-1,
        )
        z = self.encoder(z)
        responsiveness = torch.sigmoid(self.response_logit(z)).squeeze(-1)
        delta = self.flow_delta(z).squeeze(-1)
        flow = previous_flow + responsiveness * delta
        return flow, responsiveness


class HydraulicTransition(nn.Module):
    """Step 2B: node state + rainfall + actuator flow injection -> next node state."""

    def __init__(
        self,
        state_dim: int,
        rainfall_dim: int,
        static_dim: int,
        hidden_dim: int = 160,
        graph_layers: int = 3,
    ):
        super().__init__()
        self.input = nn.Linear(state_dim + rainfall_dim + static_dim + 1, hidden_dim)
        self.graph = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(graph_layers))
        self.residual = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, state_dim))

    def forward(
        self,
        state: torch.Tensor,
        rainfall: torch.Tensor,
        static_node_features: torch.Tensor,
        actuator_injection: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        b = state.shape[0]
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(b, -1, -1)
        x = self.input(torch.cat([state, rainfall, static_node_features, actuator_injection], dim=-1))
        for block in self.graph:
            x = block(x, edge_index)
        return state + self.residual(x)


@dataclass
class Rollout:
    states: torch.Tensor
    actuator_flows: torch.Tensor
    responsiveness: torch.Tensor


class DifferentiableHydraulicWorldModel(nn.Module):
    """Step 2C: recursively couple actuator flow and graph hydraulic dynamics."""

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        node_static_dim: int,
        actuator_physics_dim: int,
        hidden_dim: int = 160,
    ):
        super().__init__()
        self.actuator = ActuatorFlowModel(state_dim, actuator_physics_dim, hidden_dim)
        self.transition = HydraulicTransition(
            state_dim, rainfall_dim, node_static_dim, hidden_dim=hidden_dim
        )

    def rollout(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Rollout:
        # rainfall [B,H,N,R], settings [B,H,A]
        state = initial_state
        flow = previous_actuator_flow
        states, flows, responses = [], [], []
        up = actuator_upstream.long()
        down = actuator_downstream.long()
        for k in range(settings.shape[1]):
            q, response = self.actuator(
                state[:, up], state[:, down], settings[:, k], flow, actuator_physics
            )
            injection = torch.zeros(
                state.shape[0], state.shape[1], 1, device=state.device, dtype=state.dtype
            )
            injection = injection.index_add(1, up, -q[..., None])
            injection = injection.index_add(1, down, q[..., None])
            state = self.transition(
                state, rainfall[:, k], static_node_features, injection, edge_index
            )
            states.append(state)
            flows.append(q)
            responses.append(response)
            flow = q
        return Rollout(
            states=torch.stack(states, dim=1),
            actuator_flows=torch.stack(flows, dim=1),
            responsiveness=torch.stack(responses, dim=1),
        )
