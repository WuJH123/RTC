"""Development-only edge-aware extension of the current V128 typed actuator surrogate.

It preserves V128 actuator identity/physics/flow messages and the frozen time/action contract,
but makes ordinary graph propagation edge-aware.  Each message sees current source/destination
hidden state, frozen SWMM edge physics, current head difference, and a length-normalized head
gradient.  This class is restricted to smoke/dev ablation until held-out spatial/ranking evidence
supports promotion; the current full checkpoint contract remains the baseline V128 model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .edge_physics_current_v128 import EdgePhysicsArtifactV128
from .models import HydraulicTransition, Rollout, _inverse_degree
from .step2_differentiable_v128 import TypedActuatorMessageSurrogateV128, V128SurrogateDesign

V128_EDGE_AWARE_CONTRACT = "PROJECT7_V128_EDGE_AWARE_TYPED_SURROGATE_DEVELOPMENT_V1"


class EdgeAwareGraphMessageBlockV128(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 2, hidden_dim),
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
        inverse_degree: torch.Tensor,
        edge_static: torch.Tensor,
        dynamic_edge: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index.long()
        if edge_static.shape[0] != src.numel() or dynamic_edge.shape[:2] != (x.shape[0], src.numel()):
            raise ValueError("edge-aware V128 message inputs do not align")
        static = edge_static.to(device=x.device, dtype=x.dtype)[None].expand(x.shape[0], -1, -1)
        msg = self.message(
            torch.cat((x[:, src], x[:, dst], static, dynamic_edge.to(dtype=x.dtype)), dim=-1)
        )
        agg = torch.zeros_like(x).index_add(1, dst, msg.to(dtype=x.dtype))
        agg = agg * inverse_degree.to(device=x.device, dtype=x.dtype).view(1, -1, 1)
        delta = self.update(torch.cat((x, agg), dim=-1))
        return self.norm(x + delta)


class EdgeAwareHydraulicTransitionV128(HydraulicTransition):
    def __init__(
        self,
        state_dim: int,
        rainfall_dim: int,
        static_dim: int,
        *,
        hidden_dim: int,
        action_context_dim: int,
        delta_state_scale: torch.Tensor,
        artifact: EdgePhysicsArtifactV128,
        head_index: int = 1,
    ) -> None:
        super().__init__(
            state_dim,
            rainfall_dim,
            static_dim,
            hidden_dim=hidden_dim,
            graph_layers=0,
            action_context_dim=action_context_dim,
            bounded_state_residual=True,
            delta_state_scale=delta_state_scale,
        )
        self.head_index = int(head_index)
        edge_static = torch.as_tensor(artifact.edge_static_features, dtype=torch.float32)
        lengths = torch.as_tensor(artifact.effective_length_m, dtype=torch.float32)
        self.register_buffer("edge_static_features", edge_static)
        self.register_buffer("edge_effective_length_m", lengths.clamp_min(1.0))
        self.register_buffer("artifact_edge_index", torch.as_tensor(artifact.edge_index, dtype=torch.long))
        self.graph = nn.ModuleList(
            EdgeAwareGraphMessageBlockV128(hidden_dim, int(edge_static.shape[1])) for _ in range(3)
        )

    def prepare_static(
        self,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        static_norm, edge, inv = super().prepare_static(
            static_node_features, edge_index, batch_size=batch_size, dtype=dtype
        )
        expected = self.artifact_edge_index.to(device=edge.device)
        if edge.shape != expected.shape or not torch.equal(edge, expected):
            raise ValueError("runtime graph edge_index differs from frozen edge-physics artifact")
        return static_norm, edge, inv

    def _dynamic_edge_features(self, state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if not 0 <= self.head_index < state.shape[-1]:
            raise ValueError("edge-aware head channel is outside state tensor")
        src, dst = edge_index.long()
        head = state[..., self.head_index]
        delta = head[:, src] - head[:, dst]
        head_scale = self.state_std[self.head_index].to(device=state.device, dtype=state.dtype).clamp_min(1.0e-5)
        delta_norm = delta / head_scale
        length = self.edge_effective_length_m.to(device=state.device, dtype=state.dtype)
        raw_gradient = delta / length[None]
        median_length = torch.median(length).clamp_min(1.0)
        gradient_scale = (head_scale / median_length).clamp_min(1.0e-6)
        gradient_norm = torch.sign(raw_gradient) * torch.log1p(torch.abs(raw_gradient) / gradient_scale)
        return torch.stack((delta_norm, gradient_norm), dim=-1)

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
            if action_context is None or action_context.shape[-1] != self.action_context_dim:
                raise ValueError("edge-aware V128 requires the frozen typed action context")
            parts.append(action_context)
        x = self.input(torch.cat(parts, dim=-1))
        dynamic = self._dynamic_edge_features(state, edge_index)
        for block in self.graph:
            x = block(
                x,
                edge_index,
                inverse_degree,
                self.edge_static_features,
                dynamic,
            )
        raw_delta = self.residual(x)
        delta = torch.tanh(raw_delta) * self.delta_state_scale.to(dtype=raw_delta.dtype)
        return state + delta


class EdgeAwareTypedActuatorSurrogateV128(TypedActuatorMessageSurrogateV128):
    contract = V128_EDGE_AWARE_CONTRACT

    def __init__(self, *, edge_artifact: EdgePhysicsArtifactV128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        state_dim = int(self.transition.state_mean.numel())
        rainfall_dim = int(self.transition.rain_mean.numel())
        static_dim = int(self.transition.static_mean.numel())
        hidden_dim = int(self.transition.input.out_features)
        delta_state_scale = self.transition.delta_state_scale.detach().clone()
        self.transition = EdgeAwareHydraulicTransitionV128(
            state_dim,
            rainfall_dim,
            static_dim,
            hidden_dim=hidden_dim,
            action_context_dim=2 * self.action_message_dim,
            delta_state_scale=delta_state_scale,
            artifact=edge_artifact,
        )
        self.v128_contract = V128_EDGE_AWARE_CONTRACT
        self.runtime_metadata.update(
            {
                "development_only_edge_aware": True,
                "edge_aware_contract": V128_EDGE_AWARE_CONTRACT,
                "full_promotion_allowed": False,
            }
        )


def build_v128_edge_aware_model_from_graph(
    graph: Any,
    *,
    edge_artifact: EdgePhysicsArtifactV128,
    state_dim: int,
    rainfall_dim: int,
    delta_state_scale: torch.Tensor | np.ndarray,
    delta_flow_scale: torch.Tensor | np.ndarray,
    design: V128SurrogateDesign = V128SurrogateDesign(),
) -> EdgeAwareTypedActuatorSurrogateV128:
    edge_artifact.validate(graph)
    design.validate()
    return EdgeAwareTypedActuatorSurrogateV128(
        edge_artifact=edge_artifact,
        state_dim=int(state_dim),
        rainfall_dim=int(rainfall_dim),
        node_static_dim=int(np.asarray(graph.static_node_features).shape[1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=int(design.hidden_dim),
        actuator_embedding_dim=int(design.actuator_embedding_dim),
        action_message_dim=int(design.action_message_dim),
        delta_state_scale=torch.as_tensor(delta_state_scale, dtype=torch.float32),
        delta_flow_scale=torch.as_tensor(delta_flow_scale, dtype=torch.float32),
        model_step_seconds=300,
        horizon_steps=72,
        control_update_seconds=600,
        free_control_horizon_steps=24,
        time_contract="PROJECT7_V128_300S_MODEL_600S_RECEDING_CONTROL_V1",
    )


__all__ = [
    "EdgeAwareGraphMessageBlockV128",
    "EdgeAwareHydraulicTransitionV128",
    "EdgeAwareTypedActuatorSurrogateV128",
    "V128_EDGE_AWARE_CONTRACT",
    "build_v128_edge_aware_model_from_graph",
]
