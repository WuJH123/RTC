from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.clamp_min(1e-6)


def _inverse_degree(edge_index: torch.Tensor, node_count: int, *, dtype: torch.dtype) -> torch.Tensor:
    dst = edge_index[1]
    degree = torch.zeros(node_count, device=edge_index.device, dtype=dtype)
    degree = degree.index_add(0, dst, torch.ones_like(dst, dtype=dtype))
    return degree.clamp_min(1.0).reciprocal()


class GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        inverse_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_index = edge_index.long()
        src, dst = edge_index
        msg = self.message(torch.cat([x[:, src], x[:, dst]], dim=-1))
        # AMP can produce half-precision messages while the node activation remains float32.
        # Accumulate in the activation dtype so index_add receives matching tensors and the
        # graph reduction retains float32 numerical stability.
        agg = torch.zeros_like(x).index_add(1, dst, msg.to(dtype=x.dtype))
        inv = inverse_degree
        if inv is None:
            inv = _inverse_degree(edge_index, x.shape[1], dtype=x.dtype)
        else:
            inv = inv.to(device=x.device, dtype=x.dtype)
        agg = agg * inv.view(1, -1, 1)
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
        self.temporal = nn.GRU(
            observed_dim * 2 + static_dim + self.context_dim,
            hidden_dim,
            batch_first=True,
        )
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
        edge_index = edge_index.long()
        inv_degree = _inverse_degree(edge_index, n, dtype=x.dtype)
        for block in self.graph:
            x = block(x, edge_index, inv_degree)
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
        bounded_flow_residual: bool = False,
        delta_flow_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.actuator_count = int(actuator_count)
        self.bounded_flow_residual = bool(bounded_flow_residual)
        self.actuator_embedding_dim = int(
            actuator_embedding_dim if actuator_count > 0 else 0
        )
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
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("physics_mean", torch.zeros(physics_dim))
        self.register_buffer("physics_std", torch.ones(physics_dim))
        self.register_buffer("flow_std", torch.ones(1))
        flow_scale = (
            torch.ones(self.actuator_count, dtype=torch.float32)
            if delta_flow_scale is None
            else torch.as_tensor(delta_flow_scale, dtype=torch.float32).reshape(-1)
        )
        if flow_scale.numel() != self.actuator_count:
            raise ValueError(
                "delta_flow_scale must contain one value per actuator; "
                f"got {flow_scale.numel()} for {self.actuator_count} actuators"
            )
        self.register_buffer("delta_flow_scale", _safe_std(flow_scale))

    @torch.no_grad()
    def set_normalization(
        self, *, state_mean, state_std, physics_mean, physics_std, flow_std
    ) -> None:
        self.state_mean.copy_(state_mean.reshape_as(self.state_mean))
        self.state_std.copy_(_safe_std(state_std.reshape_as(self.state_std)))
        self.physics_mean.copy_(physics_mean.reshape_as(self.physics_mean))
        self.physics_std.copy_(_safe_std(physics_std.reshape_as(self.physics_std)))
        self.flow_std.copy_(_safe_std(flow_std.reshape_as(self.flow_std)))

    @torch.no_grad()
    def set_delta_flow_scale(self, delta_flow_scale: torch.Tensor) -> None:
        scale = torch.as_tensor(delta_flow_scale, device=self.delta_flow_scale.device)
        self.delta_flow_scale.copy_(_safe_std(scale.reshape_as(self.delta_flow_scale)))

    def prepare_static(
        self, physics: torch.Tensor, *, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if physics.dim() == 2:
            physics = physics.unsqueeze(0).expand(batch_size, -1, -1)
        physics_norm = (physics - self.physics_mean) / self.physics_std
        identity = None
        if self.identity is not None:
            actuator_n = int(physics_norm.shape[1])
            if actuator_n != self.actuator_count:
                raise ValueError(
                    f"actuator count/order differs from frozen model: "
                    f"{actuator_n} != {self.actuator_count}"
                )
            ids = torch.arange(actuator_n, device=physics_norm.device)
            identity = self.identity(ids).unsqueeze(0).expand(batch_size, -1, -1)
        return physics_norm, identity

    def forward_prepared(
        self,
        upstream_state: torch.Tensor,
        downstream_state: torch.Tensor,
        setting: torch.Tensor,
        previous_flow: torch.Tensor,
        physics_norm: torch.Tensor,
        identity_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [
            (upstream_state - self.state_mean) / self.state_std,
            (downstream_state - self.state_mean) / self.state_std,
            setting[..., None],
            previous_flow[..., None] / self.flow_std,
            physics_norm,
        ]
        if identity_embedding is not None:
            parts.append(identity_embedding)
        z = self.encoder(torch.cat(parts, dim=-1))
        responsiveness = torch.sigmoid(self.response_logit(z)).squeeze(-1)
        raw_delta = self.flow_delta(z).squeeze(-1)
        if self.bounded_flow_residual:
            delta = torch.tanh(raw_delta) * self.delta_flow_scale.to(dtype=raw_delta.dtype)
        else:
            delta = raw_delta * self.flow_std
        return previous_flow + responsiveness * delta, responsiveness

    def forward(
        self, upstream_state, downstream_state, setting, previous_flow, physics
    ):
        physics_norm, identity = self.prepare_static(
            physics, batch_size=int(setting.shape[0])
        )
        return self.forward_prepared(
            upstream_state,
            downstream_state,
            setting,
            previous_flow,
            physics_norm,
            identity,
        )


class HydraulicTransition(nn.Module):
    def __init__(
        self,
        state_dim,
        rainfall_dim,
        static_dim,
        hidden_dim=160,
        graph_layers=3,
        action_context_dim: int = 0,
        bounded_state_residual: bool = False,
        delta_state_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.action_context_dim = int(action_context_dim)
        self.bounded_state_residual = bool(bounded_state_residual)
        self.input = nn.Linear(
            state_dim + rainfall_dim + static_dim + 1 + self.action_context_dim,
            hidden_dim,
        )
        self.graph = nn.ModuleList(
            GraphMessageBlock(hidden_dim) for _ in range(graph_layers)
        )
        self.residual = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, state_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("rain_mean", torch.zeros(rainfall_dim))
        self.register_buffer("rain_std", torch.ones(rainfall_dim))
        self.register_buffer("static_mean", torch.zeros(static_dim))
        self.register_buffer("static_std", torch.ones(static_dim))
        self.register_buffer("injection_std", torch.ones(1))
        state_scale = (
            torch.ones(state_dim, dtype=torch.float32)
            if delta_state_scale is None
            else torch.as_tensor(delta_state_scale, dtype=torch.float32).reshape(-1)
        )
        if state_scale.numel() != int(state_dim):
            raise ValueError(
                "delta_state_scale must contain one value per state channel; "
                f"got {state_scale.numel()} for {state_dim} channels"
            )
        self.register_buffer("delta_state_scale", _safe_std(state_scale))

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
        injection_std,
    ):
        self.state_mean.copy_(state_mean.reshape_as(self.state_mean))
        self.state_std.copy_(_safe_std(state_std.reshape_as(self.state_std)))
        self.rain_mean.copy_(rain_mean.reshape_as(self.rain_mean))
        self.rain_std.copy_(_safe_std(rain_std.reshape_as(self.rain_std)))
        self.static_mean.copy_(static_mean.reshape_as(self.static_mean))
        self.static_std.copy_(_safe_std(static_std.reshape_as(self.static_std)))
        self.injection_std.copy_(_safe_std(injection_std.reshape_as(self.injection_std)))

    @torch.no_grad()
    def set_delta_state_scale(self, delta_state_scale: torch.Tensor) -> None:
        scale = torch.as_tensor(delta_state_scale, device=self.delta_state_scale.device)
        self.delta_state_scale.copy_(_safe_std(scale.reshape_as(self.delta_state_scale)))

    def prepare_static(
        self,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        static_norm = (static_node_features - self.static_mean) / self.static_std
        edge_index = edge_index.long()
        inv_degree = _inverse_degree(
            edge_index, static_norm.shape[1], dtype=dtype
        )
        return static_norm, edge_index, inv_degree

    def forward_prepared(
        self,
        state: torch.Tensor,
        rainfall: torch.Tensor,
        static_norm: torch.Tensor,
        actuator_injection: torch.Tensor,
        edge_index: torch.Tensor,
        inverse_degree: torch.Tensor,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            (state - self.state_mean) / self.state_std,
            (rainfall - self.rain_mean) / self.rain_std,
            static_norm,
            actuator_injection / self.injection_std,
        ]
        if self.action_context_dim:
            if action_context is None:
                raise ValueError(
                    "action_context is required when direct action context is enabled"
                )
            if action_context.shape[-1] != self.action_context_dim:
                raise ValueError("action_context has wrong feature dimension")
            parts.append(action_context)
        x = self.input(torch.cat(parts, dim=-1))
        for block in self.graph:
            x = block(x, edge_index, inverse_degree)
        raw_delta = self.residual(x)
        if self.bounded_state_residual:
            delta = torch.tanh(raw_delta) * self.delta_state_scale.to(dtype=raw_delta.dtype)
        else:
            delta = raw_delta * self.state_std
        return state + delta

    def forward(
        self,
        state,
        rainfall,
        static_node_features,
        actuator_injection,
        edge_index,
        action_context: torch.Tensor | None = None,
    ):
        static_norm, edge_index, inv_degree = self.prepare_static(
            static_node_features,
            edge_index,
            batch_size=int(state.shape[0]),
            dtype=state.dtype,
        )
        return self.forward_prepared(
            state,
            rainfall,
            static_norm,
            actuator_injection,
            edge_index,
            inv_degree,
            action_context,
        )


@dataclass
class Rollout:
    states: torch.Tensor
    actuator_flows: torch.Tensor
    responsiveness: torch.Tensor


class DifferentiableHydraulicWorldModel(nn.Module):
    def __init__(
        self,
        *,
        state_dim,
        rainfall_dim,
        node_static_dim,
        actuator_physics_dim,
        hidden_dim=160,
        actuator_count=0,
        actuator_embedding_dim=16,
        direct_action_context: bool = False,
        bounded_state_residual: bool = False,
        bounded_flow_residual: bool = False,
        delta_state_scale: torch.Tensor | None = None,
        delta_flow_scale: torch.Tensor | None = None,
        **runtime_metadata,
    ):
        super().__init__()
        self.runtime_metadata = dict(runtime_metadata)
        self.direct_action_context = bool(direct_action_context)
        self.bounded_state_residual = bool(bounded_state_residual)
        self.bounded_flow_residual = bool(bounded_flow_residual)
        self.actuator = ActuatorFlowModel(
            state_dim,
            actuator_physics_dim,
            hidden_dim,
            actuator_count=actuator_count,
            actuator_embedding_dim=actuator_embedding_dim,
            bounded_flow_residual=self.bounded_flow_residual,
            delta_flow_scale=delta_flow_scale,
        )
        self.transition = HydraulicTransition(
            state_dim,
            rainfall_dim,
            node_static_dim,
            hidden_dim=hidden_dim,
            action_context_dim=2 if self.direct_action_context else 0,
            bounded_state_residual=self.bounded_state_residual,
            delta_state_scale=delta_state_scale,
        )

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
            state_mean=state_mean,
            state_std=state_std,
            physics_mean=physics_mean,
            physics_std=physics_std,
            flow_std=flow_std,
        )
        self.transition.set_normalization(
            state_mean=state_mean,
            state_std=state_std,
            rain_mean=rain_mean,
            rain_std=rain_std,
            static_mean=static_mean,
            static_std=static_std,
            injection_std=flow_std,
        )

    @staticmethod
    def _setting_context(
        setting: torch.Tensor,
        upstream: torch.Tensor,
        downstream: torch.Tensor,
        *,
        node_count: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = setting.to(dtype=dtype)[..., None]
        outgoing = torch.zeros(
            setting.shape[0], node_count, 1, device=setting.device, dtype=dtype
        ).index_add(1, upstream, values)
        incoming = torch.zeros(
            setting.shape[0], node_count, 1, device=setting.device, dtype=dtype
        ).index_add(1, downstream, values)
        return torch.cat([outgoing, incoming], dim=-1)

    def rollout(
        self,
        initial_state,
        rainfall,
        settings,
        previous_actuator_flow,
        actuator_upstream,
        actuator_downstream,
        actuator_physics,
        static_node_features,
        edge_index,
    ):
        if (
            rainfall.shape[0] != settings.shape[0]
            or rainfall.shape[1] != settings.shape[1]
        ):
            raise ValueError("rainfall/settings batch and horizon dimensions must match")
        state, flow = initial_state, previous_actuator_flow
        states, flows, responses = [], [], []
        up, down = actuator_upstream.long(), actuator_downstream.long()
        batch = int(state.shape[0])
        physics_norm, identity = self.actuator.prepare_static(
            actuator_physics, batch_size=batch
        )
        static_norm, edge_index, inv_degree = self.transition.prepare_static(
            static_node_features,
            edge_index,
            batch_size=batch,
            dtype=state.dtype,
        )
        for k in range(settings.shape[1]):
            q, response = self.actuator.forward_prepared(
                state[:, up],
                state[:, down],
                settings[:, k],
                flow,
                physics_norm,
                identity,
            )
            injection = torch.zeros(
                state.shape[0],
                state.shape[1],
                1,
                device=state.device,
                dtype=state.dtype,
            )
            injection = injection.index_add(1, up, -q[..., None])
            injection = injection.index_add(1, down, q[..., None])
            action_context = None
            if self.direct_action_context:
                action_context = self._setting_context(
                    settings[:, k],
                    up,
                    down,
                    node_count=state.shape[1],
                    dtype=state.dtype,
                )
            state = self.transition.forward_prepared(
                state,
                rainfall[:, k],
                static_norm,
                injection,
                edge_index,
                inv_degree,
                action_context,
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
