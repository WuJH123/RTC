from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.clamp_min(1e-6)


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
        # AMP can produce half-precision messages while the node activation remains float32.
        # Accumulate in the activation dtype so index_add receives matching tensors and the
        # graph reduction retains float32 numerical stability.
        agg = torch.zeros_like(x).index_add(1, dst, msg.to(dtype=x.dtype))
        degree = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        degree = degree.index_add(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0).view(1, -1, 1)
        delta = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(x + delta)


class SparseStateEstimator(nn.Module):
    """Step1 with physical-SI external interface and checkpointed internal normalization."""

    def __init__(
        self,
        observed_dim: int,
        static_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        context_dim: int = 0,
        **runtime_metadata,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.runtime_metadata = dict(runtime_metadata)
        self.temporal = nn.GRU(observed_dim * 2 + static_dim + self.context_dim, hidden_dim, batch_first=True)
        self.graph = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(graph_layers))
        self.head = nn.Linear(hidden_dim, state_dim)
        self.register_buffer("observed_mean", torch.zeros(observed_dim))
        self.register_buffer("observed_std", torch.ones(observed_dim))
        self.register_buffer("static_mean", torch.zeros(static_dim))
        self.register_buffer("static_std", torch.ones(static_dim))
        self.register_buffer("context_mean", torch.zeros(self.context_dim))
        self.register_buffer("context_std", torch.ones(self.context_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        observed_mean: torch.Tensor,
        observed_std: torch.Tensor,
        static_mean: torch.Tensor,
        static_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        context_mean: torch.Tensor | None = None,
        context_std: torch.Tensor | None = None,
    ) -> None:
        self.observed_mean.copy_(observed_mean.reshape_as(self.observed_mean))
        self.observed_std.copy_(_safe_std(observed_std.reshape_as(self.observed_std)))
        self.static_mean.copy_(static_mean.reshape_as(self.static_mean))
        self.static_std.copy_(_safe_std(static_std.reshape_as(self.static_std)))
        self.state_mean.copy_(state_mean.reshape_as(self.state_mean))
        self.state_std.copy_(_safe_std(state_std.reshape_as(self.state_std)))
        if self.context_dim:
            if context_mean is None or context_std is None:
                raise ValueError("context normalization is required")
            self.context_mean.copy_(context_mean.reshape_as(self.context_mean))
            self.context_std.copy_(_safe_std(context_std.reshape_as(self.context_std)))

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
        observed_norm = (observed_history - self.observed_mean) / self.observed_std
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(b, -1, -1)
        static_norm = (static_node_features - self.static_mean) / self.static_std
        static = static_norm[:, None].expand(-1, t, -1, -1)
        parts = [observed_norm * observation_mask, observation_mask, static]
        if self.context_dim:
            if context_history is None:
                raise ValueError("context_history is required when context_dim > 0")
            context_norm = (context_history - self.context_mean) / self.context_std
            if context_norm.dim() == 3:
                context = context_norm[:, :, None, :].expand(-1, -1, n, -1)
            elif context_norm.dim() == 4:
                if context_norm.shape[2] != n:
                    raise ValueError("node-local context has wrong node count")
                context = context_norm
            else:
                raise ValueError("context_history must be rank 3 or 4")
            parts.append(context)
        elif context_history is not None:
            raise ValueError("context_history supplied to context-free model")
        temporal_input = torch.cat(parts, dim=-1)
        node_sequences = temporal_input.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, hidden = self.temporal(node_sequences)
        x = hidden[-1].reshape(b, n, -1)
        for block in self.graph:
            x = block(x, edge_index)
        return self.head(x) * self.state_std + self.state_mean


class ActuatorFlowModel(nn.Module):
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
        self.identity = nn.Embedding(self.actuator_count, self.actuator_embedding_dim) if self.actuator_count > 0 else None
        in_dim = state_dim * 2 + physics_dim + 2 + self.actuator_embedding_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.response_logit = nn.Linear(hidden_dim, 1)
        self.flow_delta = nn.Linear(hidden_dim, 1)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("physics_mean", torch.zeros(physics_dim))
        self.register_buffer("physics_std", torch.ones(physics_dim))
        self.register_buffer("flow_std", torch.ones(1))

    @torch.no_grad()
    def set_normalization(self, *, state_mean, state_std, physics_mean, physics_std, flow_std) -> None:
        self.state_mean.copy_(state_mean.reshape_as(self.state_mean))
        self.state_std.copy_(_safe_std(state_std.reshape_as(self.state_std)))
        self.physics_mean.copy_(physics_mean.reshape_as(self.physics_mean))
        self.physics_std.copy_(_safe_std(physics_std.reshape_as(self.physics_std)))
        self.flow_std.copy_(_safe_std(flow_std.reshape_as(self.flow_std)))

    def forward(self, upstream_state, downstream_state, setting, previous_flow, physics):
        parts = [
            (upstream_state - self.state_mean) / self.state_std,
            (downstream_state - self.state_mean) / self.state_std,
            setting[..., None],
            previous_flow[..., None] / self.flow_std,
            (physics - self.physics_mean) / self.physics_std,
        ]
        if self.identity is not None:
            actuator_n = int(setting.shape[-1])
            if actuator_n != self.actuator_count:
                raise ValueError(f"actuator count/order differs from frozen model: {actuator_n} != {self.actuator_count}")
            emb = self.identity(torch.arange(actuator_n, device=setting.device)).unsqueeze(0).expand(setting.shape[0], -1, -1)
            parts.append(emb)
        z = self.encoder(torch.cat(parts, dim=-1))
        responsiveness = torch.sigmoid(self.response_logit(z)).squeeze(-1)
        delta = self.flow_delta(z).squeeze(-1) * self.flow_std
        return previous_flow + responsiveness * delta, responsiveness


class HydraulicTransition(nn.Module):
    def __init__(self, state_dim, rainfall_dim, static_dim, hidden_dim=160, graph_layers=3):
        super().__init__()
        self.input = nn.Linear(state_dim + rainfall_dim + static_dim + 1, hidden_dim)
        self.graph = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(graph_layers))
        self.residual = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, state_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("rain_mean", torch.zeros(rainfall_dim))
        self.register_buffer("rain_std", torch.ones(rainfall_dim))
        self.register_buffer("static_mean", torch.zeros(static_dim))
        self.register_buffer("static_std", torch.ones(static_dim))
        self.register_buffer("injection_std", torch.ones(1))

    @torch.no_grad()
    def set_normalization(self, *, state_mean, state_std, rain_mean, rain_std, static_mean, static_std, injection_std):
        self.state_mean.copy_(state_mean.reshape_as(self.state_mean))
        self.state_std.copy_(_safe_std(state_std.reshape_as(self.state_std)))
        self.rain_mean.copy_(rain_mean.reshape_as(self.rain_mean))
        self.rain_std.copy_(_safe_std(rain_std.reshape_as(self.rain_std)))
        self.static_mean.copy_(static_mean.reshape_as(self.static_mean))
        self.static_std.copy_(_safe_std(static_std.reshape_as(self.static_std)))
        self.injection_std.copy_(_safe_std(injection_std.reshape_as(self.injection_std)))

    def forward(self, state, rainfall, static_node_features, actuator_injection, edge_index):
        b = state.shape[0]
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(b, -1, -1)
        x = self.input(torch.cat([
            (state - self.state_mean) / self.state_std,
            (rainfall - self.rain_mean) / self.rain_std,
            (static_node_features - self.static_mean) / self.static_std,
            actuator_injection / self.injection_std,
        ], dim=-1))
        for block in self.graph:
            x = block(x, edge_index)
        return state + self.residual(x) * self.state_std


@dataclass
class Rollout:
    states: torch.Tensor
    actuator_flows: torch.Tensor
    responsiveness: torch.Tensor


class DifferentiableHydraulicWorldModel(nn.Module):
    def __init__(self, *, state_dim, rainfall_dim, node_static_dim, actuator_physics_dim, hidden_dim=160, actuator_count=0, actuator_embedding_dim=16):
        super().__init__()
        self.actuator = ActuatorFlowModel(
            state_dim, actuator_physics_dim, hidden_dim,
            actuator_count=actuator_count, actuator_embedding_dim=actuator_embedding_dim,
        )
        self.transition = HydraulicTransition(state_dim, rainfall_dim, node_static_dim, hidden_dim=hidden_dim)

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        state_mean,
        state_std,
        rain_mean,
        rain_std,
        static_mean,
        static_std,
        physics_mean,
        physics_std,
        flow_std,
    ) -> None:
        self.actuator.set_normalization(
            state_mean=state_mean, state_std=state_std,
            physics_mean=physics_mean, physics_std=physics_std, flow_std=flow_std,
        )
        self.transition.set_normalization(
            state_mean=state_mean, state_std=state_std,
            rain_mean=rain_mean, rain_std=rain_std,
            static_mean=static_mean, static_std=static_std,
            injection_std=flow_std,
        )

    def rollout(self, initial_state, rainfall, settings, previous_actuator_flow, actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index):
        if rainfall.shape[0] != settings.shape[0] or rainfall.shape[1] != settings.shape[1]:
            raise ValueError("rainfall/settings batch and horizon dimensions must match")
        state, flow = initial_state, previous_actuator_flow
        states, flows, responses = [], [], []
        up, down = actuator_upstream.long(), actuator_downstream.long()
        for k in range(settings.shape[1]):
            q, response = self.actuator(state[:, up], state[:, down], settings[:, k], flow, actuator_physics)
            injection = torch.zeros(state.shape[0], state.shape[1], 1, device=state.device, dtype=state.dtype)
            injection = injection.index_add(1, up, -q[..., None])
            injection = injection.index_add(1, down, q[..., None])
            state = self.transition(state, rainfall[:, k], static_node_features, injection, edge_index)
            states.append(state)
            flows.append(q)
            responses.append(response)
            flow = q
        return Rollout(
            states=torch.stack(states, dim=1),
            actuator_flows=torch.stack(flows, dim=1),
            responsiveness=torch.stack(responses, dim=1),
        )
