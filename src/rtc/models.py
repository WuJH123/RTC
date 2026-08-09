from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class GraphMessageBlock(nn.Module):
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
        src, dst = edge_index.long()
        msg = self.message(torch.cat([x[:, src], x[:, dst]], dim=-1))
        agg = torch.zeros_like(x).index_add(1, dst, msg)
        degree = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        degree = degree.index_add(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0).view(1, -1, 1)
        delta = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(x + delta)


class SparseStateEstimator(nn.Module):
    """Step 1: reconstruct current network state from strictly causal sparse histories."""

    def __init__(
        self,
        observed_dim: int,
        static_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        context_dim: int = 0,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        if self.context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        in_dim = observed_dim * 2 + static_dim + self.context_dim
        self.temporal = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.graph = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(graph_layers))
        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        observed_history: torch.Tensor,
        observation_mask: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        context_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observed_history.shape != observation_mask.shape:
            raise ValueError("observed_history and observation_mask must have identical shape")
        b, t, n, _ = observed_history.shape
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(b, -1, -1)
        if static_node_features.shape[:2] != (b, n):
            raise ValueError("static node features must align with batch/node dimensions")
        static = static_node_features[:, None].expand(-1, t, -1, -1)

        parts = [observed_history * observation_mask, observation_mask, static]
        if self.context_dim:
            if context_history is None:
                raise ValueError("context_history is required when context_dim > 0")
            if context_history.dim() == 3:
                if context_history.shape[:2] != (b, t) or context_history.shape[-1] != self.context_dim:
                    raise ValueError("global context must have shape [B,T,context_dim]")
                context = context_history[:, :, None, :].expand(-1, -1, n, -1)
            elif context_history.dim() == 4:
                if context_history.shape != (b, t, n, self.context_dim):
                    raise ValueError("node context must have shape [B,T,N,context_dim]")
                context = context_history
            else:
                raise ValueError("context_history must be rank 3 or 4")
            parts.append(context)
        elif context_history is not None:
            raise ValueError("context_history was supplied but model context_dim is zero")

        temporal_input = torch.cat(parts, dim=-1)
        node_sequences = temporal_input.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, hidden = self.temporal(node_sequences)
        x = hidden[-1].reshape(b, n, -1)
        for block in self.graph:
            x = block(x, edge_index)
        return self.head(x)


class ActuatorFlowModel(nn.Module):
    """Continuous setting + local hydraulics + physical/identity features -> actuator flow."""

    def __init__(
        self,
        state_dim: int,
        physics_dim: int,
        hidden_dim: int = 128,
        *,
        actuator_count: int = 0,
        actuator_embedding_dim: int = 16,
    ):
        super().__init__()
        self.actuator_count = int(actuator_count)
        self.actuator_embedding_dim = int(actuator_embedding_dim if actuator_count > 0 else 0)
        self.identity = (
            nn.Embedding(self.actuator_count, self.actuator_embedding_dim)
            if self.actuator_count > 0
            else None
        )
        in_dim = state_dim * 2 + physics_dim + 2 + self.actuator_embedding_dim
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
        parts = [
            upstream_state,
            downstream_state,
            setting[..., None],
            previous_flow[..., None],
            physics,
        ]
        if self.identity is not None:
            actuator_n = int(setting.shape[-1])
            if actuator_n != self.actuator_count:
                raise ValueError(
                    f"actuator count/order differs from frozen model: {actuator_n} != {self.actuator_count}"
                )
            ids = torch.arange(actuator_n, device=setting.device)
            emb = self.identity(ids).unsqueeze(0).expand(setting.shape[0], -1, -1)
            parts.append(emb)
        z = self.encoder(torch.cat(parts, dim=-1))
        responsiveness = torch.sigmoid(self.response_logit(z)).squeeze(-1)
        delta = self.flow_delta(z).squeeze(-1)
        flow = previous_flow + responsiveness * delta
        return flow, responsiveness


class HydraulicTransition(nn.Module):
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
    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        node_static_dim: int,
        actuator_physics_dim: int,
        hidden_dim: int = 160,
        actuator_count: int = 0,
        actuator_embedding_dim: int = 16,
    ):
        super().__init__()
        self.actuator = ActuatorFlowModel(
            state_dim,
            actuator_physics_dim,
            hidden_dim,
            actuator_count=actuator_count,
            actuator_embedding_dim=actuator_embedding_dim,
        )
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
        if rainfall.shape[0] != settings.shape[0] or rainfall.shape[1] != settings.shape[1]:
            raise ValueError("rainfall/settings batch and horizon dimensions must match")
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
