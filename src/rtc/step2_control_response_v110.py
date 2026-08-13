"""Actuator-set, dual-timescale Hydraulic Effect surrogate for Project7 Step2 V11.

The model is intentionally direct-response rather than recurrent free-run.  For
each retained response time it receives only the action prefix that has already
occurred, so future action changes cannot leak backward in time.  Actuators that
have changed by that response time form a variable-size set; self-attention
learns joint interactions and global relation-biased cross-attention maps the
set to every node without a finite-hop cutoff.

The long-horizon 0-360 min control objective remains the frozen V7 direct
Delta-TFV model.  This module only learns 0-120 min hydraulic consequences.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import HydraulicResponseSurrogateV70
from .step2_v110_contract import ActuatorSetHydraulicContractV110, HydraulicHorizonV110

STATE_EFFECT_CHANNELS_V110 = (0, 2, 3, 4, 5)
ACTION_PREFIX_FEATURE_NAMES_V110 = (
    "reference_setting_now",
    "candidate_setting_now",
    "delta_setting_now",
    "prefix_mean_delta",
    "prefix_mean_abs_delta",
    "prefix_max_abs_delta",
    "recent_10min_mean_delta",
    "recent_30min_mean_delta",
    "recent_60min_mean_delta",
    "age_since_first_change_fraction",
    "age_since_last_change_fraction",
    "changed_ever",
)
PAIR_FEATURE_NAMES_V110 = (
    "is_upstream_endpoint",
    "is_downstream_endpoint",
    "undirected_upstream_proximity",
    "undirected_downstream_proximity",
    "forward_upstream_proximity",
    "forward_downstream_proximity",
    "reverse_upstream_proximity",
    "reverse_downstream_proximity",
    "same_hydraulic_component",
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
    return values.sign() * torch.log1p(values.abs())


def _bfs(adjacency: Sequence[Sequence[int]], start: int) -> np.ndarray:
    distance = np.full(len(adjacency), -1, dtype=np.int32)
    distance[int(start)] = 0
    queue: deque[int] = deque([int(start)])
    while queue:
        node = queue.popleft()
        nxt_distance = int(distance[node]) + 1
        for nxt in adjacency[node]:
            if distance[nxt] < 0:
                distance[nxt] = nxt_distance
                queue.append(int(nxt))
    return distance


def _proximity(distance: np.ndarray) -> np.ndarray:
    result = np.zeros(distance.shape, dtype=np.float32)
    reachable = distance >= 0
    result[reachable] = 1.0 / (1.0 + distance[reachable].astype(np.float32))
    return result


@dataclass(frozen=True)
class ActuatorNodeRelationsV110:
    pair_features: torch.Tensor  # [A,N,F]
    pair_feature_names: tuple[str, ...]
    node_count: int
    actuator_count: int
    finite_hop_cutoff: bool = False
    uses_future_truth: bool = False
    uses_online_link_flow: bool = False


def build_actuator_node_relations_v110(graph: Any) -> ActuatorNodeRelationsV110:
    """Build all-range static actuator-node relations from the frozen graph schema."""
    node_count = len(graph.node_ids)
    actuator_count = len(graph.actuator_ids)
    edge_index = np.asarray(graph.edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("V11 graph.edge_index must be [2,E]")
    forward: list[list[int]] = [[] for _ in range(node_count)]
    reverse: list[list[int]] = [[] for _ in range(node_count)]
    undirected: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in edge_index.T.tolist():
        if not (0 <= src < node_count and 0 <= dst < node_count):
            raise ValueError("V11 graph edge outside node range")
        forward[src].append(dst)
        reverse[dst].append(src)
        undirected[src].append(dst)
        undirected[dst].append(src)

    upstream = np.asarray(graph.actuator_upstream, dtype=np.int64).reshape(-1)
    downstream = np.asarray(graph.actuator_downstream, dtype=np.int64).reshape(-1)
    if upstream.shape != (actuator_count,) or downstream.shape != (actuator_count,):
        raise ValueError("V11 actuator endpoints are misaligned")

    features = np.zeros(
        (actuator_count, node_count, len(PAIR_FEATURE_NAMES_V110)), dtype=np.float32
    )
    for actuator, (up, down) in enumerate(zip(upstream.tolist(), downstream.tolist())):
        und_up, und_down = _bfs(undirected, up), _bfs(undirected, down)
        fwd_up, fwd_down = _bfs(forward, up), _bfs(forward, down)
        rev_up, rev_down = _bfs(reverse, up), _bfs(reverse, down)
        features[actuator, up, 0] = 1.0
        features[actuator, down, 1] = 1.0
        features[actuator, :, 2] = _proximity(und_up)
        features[actuator, :, 3] = _proximity(und_down)
        features[actuator, :, 4] = _proximity(fwd_up)
        features[actuator, :, 5] = _proximity(fwd_down)
        features[actuator, :, 6] = _proximity(rev_up)
        features[actuator, :, 7] = _proximity(rev_down)
        features[actuator, :, 8] = np.logical_or(und_up >= 0, und_down >= 0).astype(
            np.float32
        )
    if not np.isfinite(features).all():
        raise RuntimeError("V11 relation features contain non-finite values")
    return ActuatorNodeRelationsV110(
        pair_features=torch.from_numpy(features),
        pair_feature_names=PAIR_FEATURE_NAMES_V110,
        node_count=node_count,
        actuator_count=actuator_count,
    )


def response_time_features_v110(
    indices: torch.Tensor, *, horizon_steps: int, dtype: torch.dtype
) -> torch.Tensor:
    x = (indices.to(dtype=dtype) + 1.0) / float(horizon_steps)
    return torch.stack((x, x.square(), torch.sin(torch.pi * x), torch.cos(torch.pi * x)), dim=-1)


def action_prefix_features_v110(
    reference_settings: torch.Tensor,
    candidate_settings: torch.Tensor,
    retained_indices: torch.Tensor,
    *,
    change_epsilon: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal action exposure at each response time.

    Returns [B,C,T,A,F] features and [B,C,T,A] changed-ever masks.  Every
    feature for response index r is computed only from action steps <= r.
    """
    if reference_settings.ndim != 3 or candidate_settings.ndim != 4:
        raise ValueError("V11 settings must be [B,H,A] and [B,C,H,A]")
    batch, candidates, horizon, actuators = candidate_settings.shape
    if reference_settings.shape != (batch, horizon, actuators):
        raise ValueError("V11 candidate/reference settings are misaligned")
    reference = reference_settings[:, None].expand_as(candidate_settings)
    delta = candidate_settings - reference
    records: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for raw_index in retained_indices.detach().cpu().tolist():
        index = int(raw_index)
        if index < 0 or index >= horizon:
            raise ValueError("V11 retained action index outside candidate horizon")
        prefix = delta[:, :, : index + 1]
        changed = prefix.abs() > float(change_epsilon)
        changed_ever = changed.any(dim=2)
        current = delta[:, :, index]
        prefix_mean = prefix.mean(dim=2)
        prefix_abs_mean = prefix.abs().mean(dim=2)
        prefix_max_abs = prefix.abs().amax(dim=2)
        recent10 = delta[:, :, max(index - 1, 0) : index + 1].mean(dim=2)
        recent30 = delta[:, :, max(index - 5, 0) : index + 1].mean(dim=2)
        recent60 = delta[:, :, max(index - 11, 0) : index + 1].mean(dim=2)

        changed_float = changed.to(delta.dtype)
        first = changed_float.argmax(dim=2)
        reversed_changed = torch.flip(changed_float, dims=(2,))
        last_from_end = reversed_changed.argmax(dim=2)
        last = index - last_from_end
        denominator = float(index + 1)
        first_age = (index - first).to(delta.dtype) / denominator
        last_age = (index - last).to(delta.dtype) / denominator
        first_age = torch.where(changed_ever, first_age, torch.zeros_like(first_age))
        last_age = torch.where(changed_ever, last_age, torch.zeros_like(last_age))

        record = torch.stack(
            (
                reference[:, :, index],
                candidate_settings[:, :, index],
                current,
                prefix_mean,
                prefix_abs_mean,
                prefix_max_abs,
                recent10,
                recent30,
                recent60,
                first_age,
                last_age,
                changed_ever.to(delta.dtype),
            ),
            dim=-1,
        )
        records.append(record)
        masks.append(changed_ever)
    return torch.stack(records, dim=2), torch.stack(masks, dim=2)


@dataclass(frozen=True)
class HydraulicEffectOutputV110:
    horizon_indices: torch.Tensor
    response_minutes: torch.Tensor
    reference_states_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    active_state_logits: torch.Tensor
    sign_state_logits: torch.Tensor
    magnitude_state_normalized: torch.Tensor
    active_flow_logits: torch.Tensor
    sign_flow_logits: torch.Tensor
    magnitude_flow_normalized: torch.Tensor
    raw_delta_states_physical: torch.Tensor
    raw_delta_flows_physical: torch.Tensor
    candidate_states_projected_physical: torch.Tensor
    candidate_flows_projected_physical: torch.Tensor
    changed_actuator_mask: torch.Tensor


class ActuatorSetHydraulicResponseV110(nn.Module):
    """Variable-set, nonlocal, causal 0-120 min counterfactual response model."""

    def __init__(
        self,
        *,
        reference_model: HydraulicResponseSurrogateV70,
        state_magnitude_scale: torch.Tensor | np.ndarray,
        flow_magnitude_scale: torch.Tensor | np.ndarray,
        node_static_dim: int,
        physics_dim: int,
        rainfall_dim: int,
        actuator_count: int,
        node_count: int,
        relations: ActuatorNodeRelationsV110,
        horizon: HydraulicHorizonV110 = HydraulicHorizonV110(),
        contract: ActuatorSetHydraulicContractV110 = ActuatorSetHydraulicContractV110(),
    ) -> None:
        super().__init__()
        horizon.validate()
        contract.validate()
        if relations.node_count != node_count or relations.actuator_count != actuator_count:
            raise ValueError("V11 relation dimensions do not match graph")
        if relations.finite_hop_cutoff or relations.uses_future_truth or relations.uses_online_link_flow:
            raise ValueError("V11 relations violate nonlocal/causal contract")
        self.reference_model = reference_model
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

        self.horizon = horizon
        self.contract = contract
        self.node_count = int(node_count)
        self.actuator_count = int(actuator_count)
        self.rainfall_dim = int(rainfall_dim)
        self.physics_dim = int(physics_dim)
        self.node_static_dim = int(node_static_dim)
        self.hidden_dim = int(contract.hidden_dim)
        self.attention_dim = int(contract.attention_dim)

        state_scale = torch.as_tensor(state_magnitude_scale, dtype=torch.float32)
        flow_scale = torch.as_tensor(flow_magnitude_scale, dtype=torch.float32).reshape(-1)
        if state_scale.shape != (node_count, 5):
            raise ValueError("V11 state magnitude scale must be [N,5]")
        if flow_scale.shape != (actuator_count,):
            raise ValueError("V11 flow magnitude scale must be [A]")
        self.register_buffer("state_magnitude_scale", state_scale.clamp_min(1.0e-8))
        self.register_buffer("flow_magnitude_scale", flow_scale.clamp_min(1.0e-8))
        self.register_buffer("pair_features", relations.pair_features.float())

        self.actuator_identity = nn.Embedding(actuator_count, contract.actuator_identity_dim)
        self.node_identity = nn.Embedding(node_count, 8)

        action_dim = len(ACTION_PREFIX_FEATURE_NAMES_V110)
        source_input_dim = (
            action_dim
            + 12  # predicted reference upstream/downstream six-state vectors
            + 2   # current previous flow + predicted reference future flow
            + physics_dim
            + contract.actuator_identity_dim
            + rainfall_dim
            + contract.time_feature_dim
        )
        self.source_encoder = _mlp(source_input_dim, self.hidden_dim, self.attention_dim)
        self.set_attention = nn.MultiheadAttention(
            self.attention_dim, contract.attention_heads, batch_first=True
        )
        self.set_norm = nn.LayerNorm(self.attention_dim)
        self.set_ff = _mlp(self.attention_dim, self.hidden_dim, self.attention_dim)

        node_query_dim = (
            6
            + 6  # predicted reference state + normalized current state
            + node_static_dim
            + 8
            + rainfall_dim
            + contract.time_feature_dim
        )
        self.node_query_encoder = _mlp(node_query_dim, self.hidden_dim, self.attention_dim)
        flow_query_dim = (
            12
            + 1
            + physics_dim
            + contract.actuator_identity_dim
            + rainfall_dim
            + contract.time_feature_dim
        )
        self.flow_query_encoder = _mlp(flow_query_dim, self.hidden_dim, self.attention_dim)
        self.source_key = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.source_value = nn.Linear(self.attention_dim, self.hidden_dim, bias=False)
        self.node_query_projection = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.flow_query_projection = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.pair_bias = _mlp(len(PAIR_FEATURE_NAMES_V110), contract.pair_hidden_dim, 1)

        self.node_decoder = _mlp(
            self.hidden_dim + self.attention_dim, self.hidden_dim, self.hidden_dim
        )
        self.flow_decoder = _mlp(
            self.hidden_dim + self.attention_dim, self.hidden_dim, self.hidden_dim
        )
        self.state_active_head = nn.Linear(self.hidden_dim, 5)
        self.state_sign_head = nn.Linear(self.hidden_dim, 5)
        self.state_magnitude_head = nn.Linear(self.hidden_dim, 5)
        self.flow_active_head = nn.Linear(self.hidden_dim, 1)
        self.flow_sign_head = nn.Linear(self.hidden_dim, 1)
        self.flow_magnitude_head = nn.Linear(self.hidden_dim, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference_model.eval()
        return self

    def _reference_short(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        prepared: PreparedStaticV60,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            reference_output = self.reference_model(
                initial_state,
                rainfall,
                reference_settings,
                reference_settings[:, None],
                prepared,
            )
        desired = torch.as_tensor(
            self.horizon.indices(), dtype=torch.long, device=initial_state.device
        )
        available = reference_output.horizon_indices
        positions: list[int] = []
        for target in desired.detach().cpu().tolist():
            match = torch.nonzero(available == int(target), as_tuple=False).reshape(-1)
            if match.numel() != 1:
                raise ValueError(
                    f"V11 short-horizon index {target} absent/duplicated in frozen V7 reference"
                )
            positions.append(int(match.item()))
        pos = torch.as_tensor(positions, dtype=torch.long, device=initial_state.device)
        states = reference_output.reference_states_physical[:, 0].index_select(1, pos)
        flows = reference_output.reference_flows_physical[:, 0].index_select(1, pos)
        return desired, states, flows

    def _rain_context(
        self, rainfall: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        records = []
        for raw in indices.detach().cpu().tolist():
            index = int(raw)
            records.append(rainfall[:, : index + 1].mean(dim=(1, 2)))
        return torch.stack(records, dim=1)

    def _set_context(
        self, source: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, candidates, retained, actuators, hidden = source.shape
        flat = source.reshape(batch * candidates * retained, actuators, hidden)
        active = active_mask.reshape(batch * candidates * retained, actuators)
        safe_mask = ~active
        no_active = ~active.any(dim=1)
        if no_active.any():
            safe_mask = safe_mask.clone()
            safe_mask[no_active, 0] = False
            flat = flat.clone()
            flat[no_active, 0] = 0.0
        attended, _ = self.set_attention(
            flat, flat, flat, key_padding_mask=safe_mask, need_weights=False
        )
        contextual = self.set_norm(flat + attended)
        contextual = contextual + self.set_ff(contextual)
        contextual = contextual.reshape(batch, candidates, retained, actuators, hidden)
        return torch.where(
            active_mask[..., None],
            contextual,
            torch.zeros_like(contextual),
        )

    def _cross_attention(
        self,
        *,
        query: torch.Tensor,
        source: torch.Tensor,
        active_mask: torch.Tensor,
        relation_bias: torch.Tensor,
        query_projection: nn.Linear,
    ) -> torch.Tensor:
        q = query_projection(query)
        k = self.source_key(source)
        v = self.source_value(source)
        scores = torch.einsum("btqd,bctad->bctqa", q, k) / sqrt(float(self.attention_dim))
        scores = scores + relation_bias.to(scores)[None, None, None]
        no_active = ~active_mask.any(dim=-1)
        safe_mask = active_mask.clone()
        if no_active.any():
            safe_mask = safe_mask.clone()
            safe_mask[..., 0] = safe_mask[..., 0] | no_active
        scores = scores.masked_fill(~safe_mask[..., None, :], -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bctqa,bctah->bctqh", weights, v)
        return torch.where(
            no_active[..., None, None],
            torch.zeros_like(context),
            context,
        )

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV60,
    ) -> HydraulicEffectOutputV110:
        if initial_state.ndim != 3 or initial_state.shape[-1] != 6:
            raise ValueError("V11 initial state must be [B,N,6]")
        if rainfall.ndim != 4:
            raise ValueError("V11 rainfall must be [B,H,N,R]")
        if candidate_settings.ndim != 4:
            raise ValueError("V11 candidate settings must be [B,C,H,A]")
        batch, candidates, full_horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count or initial_state.shape[1] != self.node_count:
            raise ValueError("V11 node/actuator dimensions do not match model")
        if reference_settings.shape != (batch, full_horizon, actuators):
            raise ValueError("V11 reference action shape mismatch")
        if previous_actuator_flow.shape != (batch, actuators):
            raise ValueError("V11 previous actuator flow shape mismatch")

        indices, reference_states, reference_flows = self._reference_short(
            initial_state, rainfall, reference_settings, prepared
        )
        retained = indices.numel()
        action_features, active_mask = action_prefix_features_v110(
            reference_settings, candidate_settings, indices
        )
        rain_context = self._rain_context(rainfall, indices)
        time_features = response_time_features_v110(
            indices, horizon_steps=self.horizon.horizon_steps, dtype=initial_state.dtype
        )
        time_bt = time_features[None].expand(batch, -1, -1)

        up = prepared.actuator_upstream
        down = prepared.actuator_downstream
        ref_up = _signed_log1p(reference_states[:, :, up])
        ref_down = _signed_log1p(reference_states[:, :, down])
        ref_flow = _signed_log1p(reference_flows)
        prev_flow = previous_actuator_flow[:, None, :].expand(batch, retained, -1)
        physics = prepared.actuator_physics[None, None].expand(batch, retained, -1, -1)
        ids = torch.arange(self.actuator_count, device=initial_state.device)
        identity = self.actuator_identity(ids)[None, None].expand(batch, retained, -1, -1)
        rain_act = rain_context[:, :, None].expand(batch, retained, actuators, -1)
        time_act = time_bt[:, :, None].expand(batch, retained, actuators, -1)

        base_source = torch.cat(
            (
                ref_up,
                ref_down,
                prev_flow[..., None],
                ref_flow[..., None],
                physics,
                identity,
                rain_act,
                time_act,
            ),
            dim=-1,
        )
        base_source = base_source[:, None].expand(batch, candidates, retained, -1, -1)
        source = self.source_encoder(torch.cat((action_features, base_source), dim=-1))
        source = self._set_context(source, active_mask)

        static = prepared.node_static[None, None].expand(batch, retained, -1, -1)
        node_ids = torch.arange(self.node_count, device=initial_state.device)
        node_identity = self.node_identity(node_ids)[None, None].expand(batch, retained, -1, -1)
        current = initial_state[:, None].expand(batch, retained, -1, -1)
        rain_node = rain_context[:, :, None].expand(batch, retained, self.node_count, -1)
        time_node = time_bt[:, :, None].expand(batch, retained, self.node_count, -1)
        node_query = self.node_query_encoder(
            torch.cat(
                (
                    _signed_log1p(reference_states),
                    current,
                    static,
                    node_identity,
                    rain_node,
                    time_node,
                ),
                dim=-1,
            )
        )

        pair_bias_an = self.pair_bias(self.pair_features.to(initial_state)).squeeze(-1)
        node_relation_bias = pair_bias_an.transpose(0, 1)
        node_context = self._cross_attention(
            query=node_query,
            source=source,
            active_mask=active_mask,
            relation_bias=node_relation_bias,
            query_projection=self.node_query_projection,
        )
        node_query_expanded = node_query[:, None].expand(
            batch, candidates, retained, self.node_count, -1
        )
        node_hidden = self.node_decoder(torch.cat((node_context, node_query_expanded), dim=-1))

        flow_up = _signed_log1p(reference_states[:, :, up])
        flow_down = _signed_log1p(reference_states[:, :, down])
        flow_physics = prepared.actuator_physics[None, None].expand(
            batch, retained, -1, -1
        )
        flow_identity = self.actuator_identity(ids)[None, None].expand(
            batch, retained, -1, -1
        )
        flow_rain = rain_context[:, :, None].expand(batch, retained, actuators, -1)
        flow_time = time_bt[:, :, None].expand(batch, retained, actuators, -1)
        flow_query = self.flow_query_encoder(
            torch.cat(
                (
                    flow_up,
                    flow_down,
                    _signed_log1p(reference_flows)[..., None],
                    flow_physics,
                    flow_identity,
                    flow_rain,
                    flow_time,
                ),
                dim=-1,
            )
        )
        source_to_target = 0.5 * (
            pair_bias_an[:, up] + pair_bias_an[:, down]
        ).transpose(0, 1)
        flow_context = self._cross_attention(
            query=flow_query,
            source=source,
            active_mask=active_mask,
            relation_bias=source_to_target,
            query_projection=self.flow_query_projection,
        )
        flow_query_expanded = flow_query[:, None].expand(
            batch, candidates, retained, actuators, -1
        )
        flow_hidden = self.flow_decoder(
            torch.cat((flow_context, flow_query_expanded), dim=-1)
        )

        active_state_logits = self.state_active_head(node_hidden)
        sign_state_logits = self.state_sign_head(node_hidden)
        magnitude_state = F.softplus(self.state_magnitude_head(node_hidden))
        active_flow_logits = self.flow_active_head(flow_hidden).squeeze(-1)
        sign_flow_logits = self.flow_sign_head(flow_hidden).squeeze(-1)
        magnitude_flow = F.softplus(self.flow_magnitude_head(flow_hidden)).squeeze(-1)

        state_scale = self.state_magnitude_scale.to(node_hidden)[None, None, None]
        flow_scale = self.flow_magnitude_scale.to(flow_hidden)[None, None, None]
        unique_delta = (
            torch.sigmoid(active_state_logits)
            * torch.tanh(sign_state_logits)
            * magnitude_state
            * state_scale
        )
        flow_delta = (
            torch.sigmoid(active_flow_logits)
            * torch.tanh(sign_flow_logits)
            * magnitude_flow
            * flow_scale
        )
        any_active = active_mask.any(dim=-1)
        unique_delta = torch.where(
            any_active[..., None, None], unique_delta, torch.zeros_like(unique_delta)
        )
        flow_delta = torch.where(
            any_active[..., None], flow_delta, torch.zeros_like(flow_delta)
        )
        depth = unique_delta[..., 0]
        raw_delta_states = torch.stack(
            (
                depth,
                depth,
                unique_delta[..., 1],
                unique_delta[..., 2],
                unique_delta[..., 3],
                unique_delta[..., 4],
            ),
            dim=-1,
        )

        ref = reference_states[:, None].expand(
            batch, candidates, retained, self.node_count, 6
        )
        candidate_raw = ref + raw_delta_states
        projected_depth = candidate_raw[..., 0].clamp_min(0.0)
        invert = prepared.invert_elevation_m.to(candidate_raw)
        while invert.ndim < projected_depth.ndim:
            invert = invert.unsqueeze(0)
        candidate_projected = torch.stack(
            (
                projected_depth,
                projected_depth + invert,
                candidate_raw[..., 2].clamp_min(0.0),
                candidate_raw[..., 3].clamp_min(0.0),
                candidate_raw[..., 4].clamp_min(0.0),
                candidate_raw[..., 5].clamp_min(0.0),
            ),
            dim=-1,
        )
        ref_flow_expanded = reference_flows[:, None].expand(
            batch, candidates, retained, actuators
        )
        candidate_flows = ref_flow_expanded + flow_delta

        response_minutes = (
            (indices.to(initial_state.dtype) + 1.0)
            * float(self.horizon.model_step_seconds)
            / 60.0
        )
        return HydraulicEffectOutputV110(
            horizon_indices=indices,
            response_minutes=response_minutes,
            reference_states_physical=reference_states,
            reference_flows_physical=reference_flows,
            active_state_logits=active_state_logits,
            sign_state_logits=sign_state_logits,
            magnitude_state_normalized=magnitude_state,
            active_flow_logits=active_flow_logits,
            sign_flow_logits=sign_flow_logits,
            magnitude_flow_normalized=magnitude_flow,
            raw_delta_states_physical=raw_delta_states,
            raw_delta_flows_physical=flow_delta,
            candidate_states_projected_physical=candidate_projected,
            candidate_flows_projected_physical=candidate_flows,
            changed_actuator_mask=active_mask,
        )


__all__ = [
    "ACTION_PREFIX_FEATURE_NAMES_V110",
    "ActuatorNodeRelationsV110",
    "ActuatorSetHydraulicResponseV110",
    "HydraulicEffectOutputV110",
    "PAIR_FEATURE_NAMES_V110",
    "STATE_EFFECT_CHANNELS_V110",
    "action_prefix_features_v110",
    "build_actuator_node_relations_v110",
    "response_time_features_v110",
]
