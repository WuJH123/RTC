"""Project7 Step2 V8.0 direct counterfactual hydraulic-effect surrogate.

The successful V7 direct Delta-TFV value model is intentionally untouched.  V8 fixes
only the remaining hydraulic failure mode: V7 obtained candidate effects by
subtracting two large absolute predictions, while sparse action effects were averaged
over the whole 932-node network.  V8 freezes the V7 reference hydraulic model and
learns candidate-minus-reference state/flow effects directly with an exact-zero,
causal, topology-aware branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .step2_control_response_v60 import (
    HydraulicOutputV60,
    PreparedStaticV60,
    prepare_static_v60,
)
from .step2_control_response_v70 import HydraulicResponseSurrogateV70
from .step2_v80_contract import DirectHydraulicEffectLossContractV80


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _zero_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=False),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim, bias=False),
    )


@dataclass(frozen=True)
class PreparedStaticV80:
    base: PreparedStaticV60
    edge_index: torch.Tensor
    node_degree: torch.Tensor


class CausalPrefixActionProjectorV80(nn.Module):
    """Project only actions that can causally affect each retained hydraulic time."""

    def __init__(self, temporal_basis: np.ndarray, *, control_block_steps: int = 2) -> None:
        super().__init__()
        basis = torch.as_tensor(np.asarray(temporal_basis), dtype=torch.float32)
        if basis.ndim != 2:
            raise ValueError("V8 temporal basis must be [control_blocks,K]")
        if control_block_steps <= 0:
            raise ValueError("V8 control_block_steps must be positive")
        weights = basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("weights", weights)
        self.control_block_steps = int(control_block_steps)

    @property
    def feature_count(self) -> int:
        return int(self.weights.shape[1])

    def forward(self, delta_settings: torch.Tensor, retained_indices: torch.Tensor) -> torch.Tensor:
        if delta_settings.ndim != 4:
            raise ValueError("V8 delta settings must be [B,C,H,A]")
        expected = int(self.weights.shape[0]) * self.control_block_steps
        if delta_settings.shape[2] != expected:
            raise ValueError(f"V8 action horizon {delta_settings.shape[2]} != {expected}")
        blocks = delta_settings[:, :, :: self.control_block_steps]
        prefix_rows: list[torch.Tensor] = []
        for raw in retained_indices.detach().cpu().tolist():
            last_block = min(int(raw) // self.control_block_steps, self.weights.shape[0] - 1)
            active = self.weights.clone()
            if last_block + 1 < active.shape[0]:
                active[last_block + 1 :] = 0.0
            active = active / active.sum(dim=0, keepdim=True).clamp_min(1e-6)
            prefix_rows.append(active.transpose(0, 1))  # [K,T]
        prefix = torch.stack(prefix_rows, dim=0).to(blocks)  # [J,K,T]
        return torch.einsum("jkt,bcta->bcjak", prefix, blocks)


def prepare_static_v80(graph: Any, device: torch.device | str = "cpu") -> PreparedStaticV80:
    target = torch.device(device)
    base = prepare_static_v60(graph, target)
    edge_index = torch.as_tensor(np.asarray(graph.edge_index), dtype=torch.long, device=target)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("V8 graph edge_index must be [2,E]")
    node_count = int(base.node_static.shape[0])
    degree = torch.zeros(node_count, dtype=torch.float32, device=target)
    degree.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=target))
    return PreparedStaticV80(base=base, edge_index=edge_index, node_degree=degree.clamp_min(1.0))


def _scatter_actuators_to_nodes(
    values: torch.Tensor,
    upstream: torch.Tensor,
    downstream: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    if values.shape[-2] != upstream.numel() or upstream.shape != downstream.shape:
        raise ValueError("V8 actuator/node alignment failed")
    leading, hidden = values.shape[:-2], values.shape[-1]
    flat = values.reshape(-1, values.shape[-2], hidden)
    output = values.new_zeros(flat.shape[0], node_count, hidden)
    output.index_add_(1, upstream, 0.5 * flat)
    output.index_add_(1, downstream, 0.5 * flat)
    return output.reshape(*leading, node_count, hidden)


class ZeroPreservingGraphBlockV80(nn.Module):
    """Mean-neighbour propagation with no bias: zero action remains exactly zero."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbour_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        node_count, hidden = values.shape[-2:]
        flat = values.reshape(-1, node_count, hidden)
        src, dst = edge_index[0], edge_index[1]
        neighbours = torch.zeros_like(flat)
        neighbours.index_add_(1, dst, flat.index_select(1, src))
        neighbours = neighbours / degree.reshape(1, -1, 1).to(flat)
        update = F.silu(
            self.self_projection(flat) + self.neighbour_projection(neighbours)
        )
        update = self.output_projection(update)
        return (flat + 0.5 * update).reshape_as(values)


class DirectHydraulicEffectSurrogateV80(nn.Module):
    """Frozen reference trajectory + direct sparse topology-aware action effect."""

    def __init__(
        self,
        *,
        reference_model: HydraulicResponseSurrogateV70,
        temporal_basis: np.ndarray,
        control_block_steps: int,
        state_delta_scale: Sequence[float] | torch.Tensor,
        flow_delta_scale: Sequence[float] | torch.Tensor,
        physics_dim: int,
        node_static_dim: int,
        actuator_count: int,
        contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
    ) -> None:
        super().__init__()
        contract.validate()
        self.reference_model = reference_model
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

        self.contract = contract
        self.actuator_count = int(actuator_count)
        self.hidden_dim = int(contract.hidden_dim)
        context_dim = int(reference_model.hidden_dim)
        self.prefix = CausalPrefixActionProjectorV80(
            temporal_basis, control_block_steps=control_block_steps
        )
        self.time_embedding = nn.Embedding(
            len(reference_model.horizon_contract.indices()), contract.time_embedding_dim
        )
        k = self.prefix.feature_count
        base_dim = (
            2 * context_dim
            + int(physics_dim)
            + 1  # current/previous actuator flow
            + 1  # reference setting at the retained time
            + contract.time_embedding_dim
        )
        effect_dim = k + 1  # causal prefix basis + instantaneous setting delta
        self.actuator_effect_encoder = _mlp(
            base_dim + effect_dim, self.hidden_dim, self.hidden_dim
        )
        self.graph_blocks = nn.ModuleList(
            ZeroPreservingGraphBlockV80(self.hidden_dim)
            for _ in range(contract.graph_blocks)
        )
        self.context_gate = nn.Sequential(
            nn.Linear(context_dim + int(node_static_dim), self.hidden_dim),
            nn.Sigmoid(),
        )
        self.node_delta_head = _zero_head(self.hidden_dim, self.hidden_dim, 5)
        self.flow_node_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.flow_delta_head = _zero_head(self.hidden_dim, self.hidden_dim, 1)
        self.onset_delta_head = _zero_head(self.hidden_dim, self.hidden_dim, 1)
        self.onset_log_temperature = nn.Parameter(torch.zeros(()))
        self.onset_bias = nn.Parameter(torch.zeros(()))

        state_scale = torch.as_tensor(state_delta_scale, dtype=torch.float32).reshape(-1)
        flow_scale = torch.as_tensor(flow_delta_scale, dtype=torch.float32).reshape(-1)
        if state_scale.numel() != 6 or flow_scale.numel() != self.actuator_count:
            raise ValueError("V8 direct effect scales do not match state/actuator dimensions")
        self.register_buffer("state_delta_scale", state_scale.clamp_min(1e-6))
        self.register_buffer("flow_delta_scale", flow_scale.clamp_min(1e-6))

    def train(self, mode: bool = True):
        super().train(mode)
        # The absolute-reference model is immutable even while the V8 effect branch trains.
        self.reference_model.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _reference_and_context(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_count: int,
        prepared: PreparedStaticV80,
    ) -> tuple[HydraulicOutputV60, torch.Tensor]:
        # Run the expensive V7 absolute model for one reference candidate only.  This avoids
        # the historical 24-candidate absolute decoder cost and prevents candidate/reference
        # subtraction from re-entering the V8 primary effect path.
        with torch.no_grad():
            one_reference = reference_settings[:, None]
            base_output = self.reference_model(
                initial_state,
                rainfall,
                reference_settings,
                one_reference,
                prepared.base,
            )
            node_context, _ = self.reference_model.boundary(
                initial_state, rainfall, reference_settings, prepared.base
            )
        return base_output, node_context

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV80,
    ) -> HydraulicOutputV60:
        if candidate_settings.ndim != 4:
            raise ValueError("V8 candidate settings must be [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count:
            raise ValueError("V8 actuator count mismatch")
        if previous_actuator_flow.shape != (batch, actuators):
            raise ValueError("V8 previous actuator flow must be [B,A]")
        if reference_settings.shape != (batch, horizon, actuators):
            raise ValueError("V8 reference setting shape mismatch")

        base_output, node_context = self._reference_and_context(
            initial_state,
            rainfall,
            reference_settings,
            candidates,
            prepared,
        )
        indices = base_output.horizon_indices
        retained_context = node_context.index_select(1, indices)
        retained_count, node_count = retained_context.shape[1], retained_context.shape[2]

        reference_expanded = reference_settings[:, None].expand_as(candidate_settings)
        action_delta = candidate_settings - reference_expanded
        prefix_delta = self.prefix(action_delta, indices)
        current_delta = action_delta.index_select(2, indices)[..., None]

        up = retained_context[:, :, prepared.base.actuator_upstream]
        down = retained_context[:, :, prepared.base.actuator_downstream]
        physics = prepared.base.actuator_physics[None, None].expand(
            batch, retained_count, -1, -1
        )
        previous_flow = previous_actuator_flow[:, None, :, None].expand(
            batch, retained_count, -1, -1
        )
        reference_current = reference_settings.index_select(1, indices)[..., None]
        time = self.time_embedding(
            torch.arange(retained_count, device=initial_state.device)
        )[None, :, None].expand(batch, -1, actuators, -1)
        base = torch.cat(
            (up, down, physics, previous_flow, reference_current, time), dim=-1
        )
        base = base[:, None].expand(batch, candidates, -1, -1, -1)
        effect_features = torch.cat((prefix_delta, current_delta), dim=-1)
        zeros = torch.zeros_like(effect_features)
        token = self.actuator_effect_encoder(
            torch.cat((base, effect_features), dim=-1)
        ) - self.actuator_effect_encoder(torch.cat((base, zeros), dim=-1))

        node_effect = _scatter_actuators_to_nodes(
            token,
            prepared.base.actuator_upstream,
            prepared.base.actuator_downstream,
            node_count,
        )
        for block in self.graph_blocks:
            node_effect = block(node_effect, prepared.edge_index, prepared.node_degree)

        static = prepared.base.node_static[None, None].expand(
            batch, retained_count, -1, -1
        )
        gate = self.context_gate(torch.cat((retained_context, static), dim=-1))
        node_effect = node_effect * gate[:, None]

        raw_state = self.node_delta_head(node_effect)
        scale = self.state_delta_scale.to(raw_state)
        delta_depth = raw_state[..., 0] * scale[0]
        delta_flood = raw_state[..., 1] * scale[2]
        delta_volume = raw_state[..., 2] * scale[3]
        delta_inflow = raw_state[..., 3] * scale[4]
        delta_outflow = raw_state[..., 4] * scale[5]
        raw_delta_state = torch.stack(
            (
                delta_depth,
                delta_depth,
                delta_flood,
                delta_volume,
                delta_inflow,
                delta_outflow,
            ),
            dim=-1,
        )

        reference_states = base_output.reference_states_physical.expand(
            batch, candidates, retained_count, node_count, 6
        )
        depth = (reference_states[..., 0] + raw_delta_state[..., 0]).clamp_min(0.0)
        invert = prepared.base.invert_elevation_m.to(depth)
        candidate_states = torch.stack(
            (
                depth,
                depth + invert.reshape(1, 1, 1, -1),
                (reference_states[..., 2] + raw_delta_state[..., 2]).clamp_min(0.0),
                (reference_states[..., 3] + raw_delta_state[..., 3]).clamp_min(0.0),
                (reference_states[..., 4] + raw_delta_state[..., 4]).clamp_min(0.0),
                (reference_states[..., 5] + raw_delta_state[..., 5]).clamp_min(0.0),
            ),
            dim=-1,
        )
        delta_states = candidate_states - reference_states

        endpoint_effect = 0.5 * (
            node_effect[..., prepared.base.actuator_upstream, :]
            + node_effect[..., prepared.base.actuator_downstream, :]
        )
        flow_hidden = token + self.flow_node_projection(endpoint_effect)
        raw_flow_delta = self.flow_delta_head(flow_hidden).squeeze(-1)
        raw_flow_delta = raw_flow_delta * self.flow_delta_scale.to(raw_flow_delta).reshape(
            1, 1, 1, -1
        )
        reference_flows = base_output.reference_flows_physical.expand(
            batch, candidates, retained_count, actuators
        )
        candidate_flows = (reference_flows + raw_flow_delta).clamp_min(0.0)
        delta_flows = candidate_flows - reference_flows

        temperature = torch.exp(self.onset_log_temperature).clamp(0.25, 4.0)
        reference_logits = (
            base_output.reference_flood_onset_logits.expand(
                batch, candidates, retained_count, node_count
            )
            / temperature
            + self.onset_bias
        )
        candidate_logits = reference_logits + self.onset_delta_head(node_effect).squeeze(-1)

        same_action = torch.all(candidate_settings == reference_expanded, dim=(2, 3))
        state_mask = same_action[..., None, None, None]
        flow_mask = same_action[..., None, None]
        logit_mask = same_action[..., None, None]
        candidate_states = torch.where(state_mask, reference_states, candidate_states)
        delta_states = torch.where(state_mask, torch.zeros_like(delta_states), delta_states)
        candidate_flows = torch.where(flow_mask, reference_flows, candidate_flows)
        delta_flows = torch.where(flow_mask, torch.zeros_like(delta_flows), delta_flows)
        candidate_logits = torch.where(logit_mask, reference_logits, candidate_logits)
        token = torch.where(same_action[..., None, None, None], torch.zeros_like(token), token)

        return HydraulicOutputV60(
            horizon_indices=indices,
            reference_states_physical=reference_states,
            candidate_states_physical=candidate_states,
            delta_states_physical=delta_states,
            reference_flows_physical=reference_flows,
            candidate_flows_physical=candidate_flows,
            delta_flows_physical=delta_flows,
            reference_flood_onset_logits=reference_logits,
            candidate_flood_onset_logits=candidate_logits,
            joint_context_before_scatter=token,
        )


__all__ = [
    "CausalPrefixActionProjectorV80",
    "DirectHydraulicEffectSurrogateV80",
    "PreparedStaticV80",
    "ZeroPreservingGraphBlockV80",
    "prepare_static_v80",
]
