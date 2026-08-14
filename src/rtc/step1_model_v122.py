"""Project7 V12.2 sparse-state reconstructor.

The legacy Step1 encodes each node history independently and then propagates sensor
information through a small number of local graph-message layers.  With sparse sensors
in a 932-node network, dynamic information can therefore be structurally distant from
many unobserved nodes.  V12.2 retains the causal 13-frame history but adds one masked
sensor-to-all-node attention block before local hydraulic graph refinement.
"""
from __future__ import annotations

import torch
from torch import nn

from .models import GraphMessageBlock, _inverse_degree, _safe_std

STEP1_V122_CONTRACT = "PROJECT7_STEP1_V122_CAUSAL_SENSOR_CROSS_ATTENTION_V1"


class SparseStateEstimatorV122(nn.Module):
    """Causal temporal encoder + global sparse-sensor fusion + local graph refinement."""

    def __init__(
        self,
        observed_dim: int,
        static_dim: int,
        state_dim: int,
        hidden_dim: int = 160,
        graph_layers: int = 4,
        context_dim: int = 0,
        attention_heads: int = 4,
        hard_sensor_consistency: bool = True,
        **runtime_metadata: object,
    ) -> None:
        super().__init__()
        if observed_dim != 2:
            raise ValueError("V122 Step1 expects sparse depth/head observations")
        if state_dim < 2:
            raise ValueError("V122 Step1 state requires depth/head channels")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.context_dim = int(context_dim)
        self.hard_sensor_consistency = bool(hard_sensor_consistency)
        self.runtime_metadata = {
            **dict(runtime_metadata),
            "step1_contract": STEP1_V122_CONTRACT,
            "attention_heads": int(attention_heads),
            "hard_sensor_consistency": bool(hard_sensor_consistency),
        }
        self.temporal = nn.GRU(
            observed_dim * 2 + static_dim + self.context_dim,
            hidden_dim,
            batch_first=True,
        )
        self.sensor_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.sensor_norm = nn.LayerNorm(hidden_dim)
        self.graph = nn.ModuleList(
            GraphMessageBlock(hidden_dim) for _ in range(int(graph_layers))
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
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
                raise ValueError("V122 Step1 requires context normalization")
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
            raise ValueError("observed_history and observation_mask must align")
        batch, steps, nodes, _ = observed_history.shape
        observed_norm = (observed_history - self.observed_mean) / self.observed_std
        if static_node_features.dim() == 2:
            static_node_features = static_node_features.unsqueeze(0).expand(batch, -1, -1)
        static_norm = (static_node_features - self.static_mean) / self.static_std
        static = static_norm[:, None].expand(-1, steps, -1, -1)
        parts = [observed_norm * observation_mask, observation_mask, static]
        if self.context_dim:
            if context_history is None:
                raise ValueError("context_history is required")
            context_norm = (context_history - self.context_mean) / self.context_std
            if context_norm.dim() == 3:
                context_norm = context_norm[:, :, None, :].expand(-1, -1, nodes, -1)
            if context_norm.dim() != 4 or context_norm.shape[2] != nodes:
                raise ValueError("V122 Step1 context has incompatible node dimension")
            parts.append(context_norm)
        elif context_history is not None:
            raise ValueError("context supplied to context-free V122 Step1")

        temporal_input = torch.cat(parts, dim=-1)
        node_sequences = temporal_input.permute(0, 2, 1, 3).reshape(
            batch * nodes, steps, -1
        )
        _, hidden = self.temporal(node_sequences)
        x = hidden[-1].reshape(batch, nodes, -1)

        # A node is a valid sensor key only when at least one hydraulic observation was
        # present in the causal history.  Queries remain all nodes.  No future state is used.
        sensor_present = observation_mask.amax(dim=(1, 3)) > 0
        if bool(torch.any(sensor_present.sum(dim=1) == 0)):
            raise ValueError("V122 Step1 requires at least one observed sensor per sample")
        attended, _ = self.sensor_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=~sensor_present,
            need_weights=False,
        )
        x = self.sensor_norm(x + attended)

        edge_index = edge_index.long()
        inverse_degree = _inverse_degree(edge_index, nodes, dtype=x.dtype)
        for block in self.graph:
            x = block(x, edge_index, inverse_degree)
        prediction = self.head(x) * self.state_std + self.state_mean

        # The first two SWMM state channels are depth/head.  At currently observed
        # locations these are measurements, not quantities that need reconstruction.
        if self.hard_sensor_consistency:
            current = observed_history[:, -1]
            current_mask = observation_mask[:, -1].bool()
            first_two = torch.where(current_mask, current, prediction[..., :2])
            prediction = torch.cat((first_two, prediction[..., 2:]), dim=-1)
        return prediction


__all__ = ["STEP1_V122_CONTRACT", "SparseStateEstimatorV122"]
