"""Train-only V4.1 grouped counterfactual response model.

V4.1 keeps the absolute reference trajectory separate from the control response,
evaluates the shared reference exactly once per counterfactual group, and decomposes
the response into additive single-actuator effects plus a D3-only interaction
residual.  Every response head is structurally centred at zero action.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PreparedStaticV41:
    """Immutable graph tensors reused by every candidate in a group."""

    static_node_features: torch.Tensor
    actuator_physics: torch.Tensor
    actuator_upstream: torch.Tensor
    actuator_downstream: torch.Tensor
    edge_index: torch.Tensor
    invert_elevation_m: torch.Tensor


@dataclass(frozen=True)
class ReferenceEncodingV41:
    """One shared reference computation for all candidates in a group."""

    node_context: torch.Tensor
    global_context: torch.Tensor
    reference_states_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    reference_flood_latent: torch.Tensor
    node_static_context: torch.Tensor
    actuator_static: torch.Tensor
    actuator_upstream: torch.Tensor
    actuator_downstream: torch.Tensor
    invert_elevation_m: torch.Tensor


@dataclass(frozen=True)
class CounterfactualGroupResponseV41:
    """Physical outputs and explicit single/interaction contributions."""

    reference_states_physical: torch.Tensor
    candidate_states_physical: torch.Tensor
    single_delta_states_physical: torch.Tensor
    interaction_delta_states_physical: torch.Tensor
    delta_states_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    candidate_flows_physical: torch.Tensor
    single_delta_flows_physical: torch.Tensor
    interaction_delta_flows_physical: torch.Tensor
    delta_flows_physical: torch.Tensor
    direct_single_delta_tfv_m3: torch.Tensor
    direct_interaction_delta_tfv_m3: torch.Tensor
    direct_delta_tfv_m3: torch.Tensor
    trajectory_delta_tfv_m3: torch.Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _directed_scatter(
    values: torch.Tensor,
    upstream: torch.Tensor,
    downstream: torch.Tensor,
    node_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum actuator values into upstream and downstream nodes.

    ``values`` may contain arbitrary leading dimensions and must end in
    ``[actuator, channel]``.
    """

    leading = values.shape[:-2]
    actuator_count, channels = values.shape[-2:]
    flat = values.reshape(-1, actuator_count, channels)
    outgoing = values.new_zeros(flat.shape[0], node_count, channels)
    incoming = values.new_zeros(flat.shape[0], node_count, channels)
    outgoing = outgoing.index_add(1, upstream, flat)
    incoming = incoming.index_add(1, downstream, flat)
    return (
        outgoing.reshape(*leading, node_count, channels),
        incoming.reshape(*leading, node_count, channels),
    )


def _causal_accumulate(value: torch.Tensor) -> torch.Tensor:
    """Stable causal response with a fixed decaying memory kernel."""

    horizon = value.shape[-3]
    response: list[torch.Tensor] = []
    memory = torch.zeros_like(value[..., 0, :, :])
    for step in range(horizon):
        memory = 0.65 * memory + value[..., step, :, :]
        response.append(memory)
    return torch.stack(response, dim=-3)


def _trapezoid_delta_tfv(delta_flood_rate: torch.Tensor, elapsed_seconds: torch.Tensor) -> torch.Tensor:
    """Integrate a candidate-minus-reference flood-rate trajectory in m3."""

    if elapsed_seconds.shape[-1] != delta_flood_rate.shape[-2] + 1:
        raise ValueError("elapsed_seconds must contain the prefix time plus every forecast time")
    initial = torch.zeros_like(delta_flood_rate[..., :1, :])
    rate = torch.cat((initial, delta_flood_rate), dim=-2)
    dt = elapsed_seconds[..., 1:] - elapsed_seconds[..., :-1]
    while dt.dim() < rate.dim() - 1:
        dt = dt.unsqueeze(1)
    return (0.5 * (rate[..., 1:, :] + rate[..., :-1, :]) * dt.unsqueeze(-1)).sum(dim=(-2, -1))


class DifferentiableCounterfactualResponseModelV41(nn.Module):
    """Grouped reference + single-actuator + interaction-residual model."""

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
        temporal_embedding_dim: int = 12,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        flow_std: torch.Tensor,
        d2_state_scale: torch.Tensor,
        d3_state_scale: torch.Tensor,
        d2_flow_scale: torch.Tensor,
        d3_flow_scale: torch.Tensor,
        d2_tfv_scale: float,
        d3_tfv_scale: float,
        max_horizon_steps: int,
        effect_rank: int = 12,
        interaction_magnitude_features_enabled: bool = False,
    ) -> None:
        super().__init__()
        if state_dim < 3:
            raise ValueError("V4.1 requires depth, head, and flooding-rate state channels")
        self.state_dim = int(state_dim)
        self.rainfall_dim = int(rainfall_dim)
        self.node_static_dim = int(node_static_dim)
        self.actuator_physics_dim = int(actuator_physics_dim)
        self.hidden_dim = int(hidden_dim)
        self.actuator_count = int(actuator_count)
        self.actuator_embedding_dim = int(actuator_embedding_dim)
        self.temporal_embedding_dim = int(temporal_embedding_dim)
        self.max_horizon_steps = int(max_horizon_steps)
        self.effect_rank = int(effect_rank)
        self.interaction_magnitude_features_enabled = bool(interaction_magnitude_features_enabled)
        self.interaction_magnitude_feature_names = (
            "active_actuator_count",
            "delta_u_l1",
            "delta_u_l2",
            "delta_u_linf",
            "signed_delta_sum",
            "sum_delta_squared",
            "cumulative_l1_action_energy",
            "cumulative_l2_action_energy",
            "changed_control_blocks",
            "duration_since_first_action",
            "maximum_simultaneous_changed_actuators",
        )
        self.interaction_magnitude_dim = (
            len(self.interaction_magnitude_feature_names)
            if self.interaction_magnitude_features_enabled
            else 0
        )
        if self.effect_rank <= 0:
            raise ValueError("effect_rank must be positive")

        self._register_vector("state_mean", state_mean, self.state_dim)
        self._register_vector("state_std", state_std, self.state_dim, positive=True)
        flow_std_tensor = torch.as_tensor(flow_std, dtype=torch.float32).reshape(-1)
        if flow_std_tensor.numel() == 1:
            flow_std_tensor = flow_std_tensor.expand(self.actuator_count).clone()
        self._register_vector(
            "flow_std", flow_std_tensor, self.actuator_count, positive=True
        )
        self._register_vector("d2_state_scale", d2_state_scale, self.state_dim, positive=True)
        self._register_vector("d3_state_scale", d3_state_scale, self.state_dim, positive=True)
        self._register_vector("d2_flow_scale", d2_flow_scale, self.actuator_count, positive=True)
        self._register_vector("d3_flow_scale", d3_flow_scale, self.actuator_count, positive=True)
        self.register_buffer("d2_tfv_scale", torch.tensor(float(d2_tfv_scale)).clamp_min(1.0))
        self.register_buffer("d3_tfv_scale", torch.tensor(float(d3_tfv_scale)).clamp_min(1.0))

        self.actuator_identity = nn.Embedding(self.actuator_count, self.actuator_embedding_dim)
        self.temporal_identity = nn.Embedding(self.max_horizon_steps, self.temporal_embedding_dim)
        self.node_static_encoder = _mlp(self.node_static_dim, hidden_dim, hidden_dim)
        actuator_static_dim = self.actuator_physics_dim + self.actuator_embedding_dim + 2 * hidden_dim
        self.actuator_static_encoder = _mlp(actuator_static_dim, hidden_dim, hidden_dim)

        reference_node_dim = self.state_dim + self.rainfall_dim + hidden_dim + 2
        self.reference_encoder = _mlp(reference_node_dim, hidden_dim, hidden_dim)
        self.reference_state_head = nn.Linear(hidden_dim, self.state_dim)
        self.reference_flow_encoder = _mlp(hidden_dim + 2, hidden_dim, hidden_dim)
        self.reference_flow_head = nn.Linear(hidden_dim, 1)

        token_dim = hidden_dim + self.temporal_embedding_dim + 3
        self.single_effect_encoder = _mlp(token_dim, hidden_dim, hidden_dim)
        self.single_flow_head = nn.Linear(hidden_dim, 1)
        self.single_state_head = nn.Linear(hidden_dim, 2 * self.state_dim)
        self.single_network_coefficient_head = nn.Linear(
            hidden_dim, self.effect_rank * self.state_dim
        )
        self.single_node_basis_head = nn.Linear(hidden_dim, self.effect_rank)
        self.direct_single_tfv_head = _mlp(2 * hidden_dim, hidden_dim, 1)

        # Keep the V4.1 normalized interaction encoder shape unchanged.  V4.2
        # learns absolute action magnitude through a zero-start residual branch,
        # so loading an old V4.1 checkpoint preserves its initial response.
        self.interaction_encoder = _mlp(4 * hidden_dim + 3, hidden_dim, hidden_dim)
        if self.interaction_magnitude_features_enabled:
            self.interaction_magnitude_encoder = _mlp(
                self.interaction_magnitude_dim, hidden_dim, hidden_dim
            )
            self.interaction_magnitude_residual = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.interaction_magnitude_residual.weight)
            nn.init.zeros_(self.interaction_magnitude_residual.bias)
        self.interaction_flow_head = nn.Linear(2 * hidden_dim, 1)
        self.interaction_state_head = nn.Linear(
            hidden_dim, self.effect_rank * self.state_dim
        )
        self.direct_interaction_tfv_head = _mlp(2 * hidden_dim, hidden_dim, 1)

    def _register_vector(
        self,
        name: str,
        value: torch.Tensor,
        length: int,
        *,
        positive: bool = False,
    ) -> None:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() != length:
            raise ValueError(f"{name} must contain {length} values")
        if positive:
            tensor = tensor.clamp_min(1e-6)
        self.register_buffer(name, tensor)

    def prepare_static(
        self,
        *,
        static_node_features: torch.Tensor,
        actuator_physics: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        edge_index: torch.Tensor,
        invert_elevation_m: torch.Tensor,
    ) -> PreparedStaticV41:
        """Validate and retain graph/static tensors without candidate duplication."""

        node_count = int(static_node_features.shape[0])
        if static_node_features.shape != (node_count, self.node_static_dim):
            raise ValueError("static_node_features shape does not match model")
        if actuator_physics.shape != (self.actuator_count, self.actuator_physics_dim):
            raise ValueError("actuator_physics shape does not match model")
        if actuator_upstream.numel() != self.actuator_count or actuator_downstream.numel() != self.actuator_count:
            raise ValueError("actuator topology does not match actuator_count")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        if invert_elevation_m.numel() != node_count:
            raise ValueError("invert_elevation_m must contain one value per node")
        return PreparedStaticV41(
            static_node_features=static_node_features,
            actuator_physics=actuator_physics,
            actuator_upstream=actuator_upstream.long(),
            actuator_downstream=actuator_downstream.long(),
            edge_index=edge_index.long(),
            invert_elevation_m=invert_elevation_m.reshape(node_count),
        )

    def _static_encodings(
        self,
        prepared: PreparedStaticV41,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        static = prepared.static_node_features.to(device=device, dtype=dtype)
        physics = prepared.actuator_physics.to(device=device, dtype=dtype)
        upstream = prepared.actuator_upstream.to(device=device)
        downstream = prepared.actuator_downstream.to(device=device)
        node_static = self.node_static_encoder(static)
        ids = torch.arange(self.actuator_count, device=device)
        identity = self.actuator_identity(ids)
        endpoint = torch.cat((node_static[upstream], node_static[downstream]), dim=-1)
        actuator_static = self.actuator_static_encoder(torch.cat((physics, identity, endpoint), dim=-1))
        return node_static, actuator_static, upstream, downstream, prepared.invert_elevation_m.to(device=device, dtype=dtype)

    def encode_reference(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV41,
    ) -> ReferenceEncodingV41:
        """Encode the same-prefix reference once, independently of candidates."""

        if rainfall.dim() != 4:
            raise ValueError("rainfall must be [B,H,N,R]")
        batch, horizon, nodes, rain_channels = rainfall.shape
        if rain_channels != self.rainfall_dim:
            raise ValueError("rainfall channel count does not match model")
        if initial_state.shape != (batch, nodes, self.state_dim):
            raise ValueError("initial_state shape does not match rainfall")
        if reference_settings.shape != (batch, horizon, self.actuator_count):
            raise ValueError("reference_settings shape does not match group")
        if horizon > self.max_horizon_steps:
            raise ValueError("forecast horizon exceeds configured temporal embedding")

        dtype = initial_state.dtype
        device = initial_state.device
        node_static, actuator_static, upstream, downstream, invert = self._static_encodings(
            prepared, device=device, dtype=dtype
        )
        ref_action = reference_settings.unsqueeze(-1)
        ref_out, ref_in = _directed_scatter(ref_action, upstream, downstream, nodes)
        initial_h = initial_state[:, None].expand(-1, horizon, -1, -1)
        static_h = node_static[None, None].expand(batch, horizon, -1, -1)
        reference_input = torch.cat((initial_h, rainfall, static_h, ref_out, ref_in), dim=-1)
        node_context = self.reference_encoder(reference_input)
        raw_step = torch.tanh(self.reference_state_head(node_context))
        raw_state = initial_h * self.state_std + self.state_mean
        raw_state = raw_state + torch.cumsum(raw_step * self.state_std, dim=1)

        depth = F.softplus(raw_state[..., 0])
        flood_latent = raw_state[..., 2]
        flood = F.softplus(flood_latent)
        physical = torch.cat(
            (
                depth.unsqueeze(-1),
                (depth + invert.reshape(1, 1, nodes)).unsqueeze(-1),
                flood.unsqueeze(-1),
                raw_state[..., 3:],
            ),
            dim=-1,
        )

        ref_flow_input = torch.cat(
            (
                actuator_static[None, None].expand(batch, horizon, -1, -1),
                reference_settings.unsqueeze(-1),
                reference_settings.new_zeros(batch, horizon, self.actuator_count, 1),
            ),
            dim=-1,
        )
        flow_hidden = self.reference_flow_encoder(ref_flow_input)
        flow_step = torch.tanh(self.reference_flow_head(flow_hidden).squeeze(-1)) * self.flow_std
        reference_flows = previous_actuator_flow[:, None] * self.flow_std + torch.cumsum(flow_step, dim=1)
        return ReferenceEncodingV41(
            node_context=node_context,
            global_context=node_context.mean(dim=2),
            reference_states_physical=physical,
            reference_flows_physical=reference_flows,
            reference_flood_latent=flood_latent,
            node_static_context=node_static,
            actuator_static=actuator_static,
            actuator_upstream=upstream,
            actuator_downstream=downstream,
            invert_elevation_m=invert,
        )

    @staticmethod
    def _allows_interaction(source_kind: str) -> bool:
        source = source_kind.upper()
        if source == "D2":
            return False
        if source == "D3":
            return True
        raise ValueError("source_kind must be D2 or D3")

    def _topology_interaction(
        self,
        *,
        hidden_delta: torch.Tensor,
        delta_u: torch.Tensor,
        interaction_hidden: torch.Tensor,
        reference: ReferenceEncodingV41,
        prepared: PreparedStaticV41,
        interaction_time_gate: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Optional D3 spatial hook; V4.1/V4.2 deliberately use no graph path."""

        del hidden_delta, delta_u, interaction_hidden, reference, prepared, interaction_time_gate
        return None, None

    def _candidate_tokens(
        self,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        actuator_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, candidates, horizon, actuators = candidate_settings.shape
        delta = candidate_settings - reference_settings[:, None]
        temporal_ids = torch.arange(horizon, device=candidate_settings.device)
        temporal = self.temporal_identity(temporal_ids)[None, None, :, None]
        temporal = temporal.expand(batch, candidates, -1, actuators, -1)
        static = actuator_static[None, None, None].expand(batch, candidates, horizon, -1, -1)
        reference = reference_settings[:, None, :, :, None].expand(-1, candidates, -1, -1, -1)
        actual_input = torch.cat((static, temporal, reference, candidate_settings[..., None], delta[..., None]), dim=-1)
        zero_input = torch.cat((static, temporal, reference, reference, torch.zeros_like(delta[..., None])), dim=-1)
        return self.single_effect_encoder(actual_input), self.single_effect_encoder(zero_input), delta

    def interaction_magnitude_features(self, delta_u: torch.Tensor) -> torch.Tensor:
        """Build causal, absolute action descriptors for the D3 interaction branch."""

        if delta_u.dim() != 4:
            raise ValueError("delta_u must be [batch, candidate, horizon, actuator]")
        absolute = delta_u.abs()
        active = delta_u.detach().ne(0.0)
        active_count = active.sum(dim=-1).to(delta_u.dtype)
        l1 = absolute.sum(dim=-1)
        l2 = delta_u.square().sum(dim=-1).sqrt()
        linf = absolute.amax(dim=-1)
        signed_sum = delta_u.sum(dim=-1)
        square_sum = delta_u.square().sum(dim=-1)
        cumulative_l1 = l1.cumsum(dim=2)
        cumulative_l2 = square_sum.cumsum(dim=2).sqrt()
        step_has_action = active_count.gt(0.0)
        horizon = delta_u.shape[2]
        block_count = (horizon + 1) // 2
        padded = step_has_action
        if horizon % 2:
            padded = torch.cat((padded, torch.zeros_like(padded[..., :1])), dim=2)
        block_has_action = padded.reshape(*padded.shape[:2], block_count, 2).any(dim=-1)
        changed_blocks = block_has_action.cumsum(dim=2).repeat_interleave(2, dim=2)[..., :horizon]
        time = torch.arange(horizon, device=delta_u.device, dtype=delta_u.dtype)
        time = time.reshape(1, 1, horizon)
        first_candidates = torch.where(
            step_has_action,
            time.expand_as(step_has_action),
            torch.full_like(time.expand_as(step_has_action), float(horizon)),
        )
        first_action = torch.cummin(first_candidates, dim=2).values
        duration_since_first = (time - first_action).clamp_min(0.0)
        maximum_simultaneous = torch.cummax(active_count, dim=2).values
        actuator_scale = float(max(self.actuator_count, 1))
        step_scale = (time + 1.0).clamp_min(1.0)
        block_scale = float(max((self.max_horizon_steps + 1) // 2, 1))
        duration_scale = float(max(self.max_horizon_steps - 1, 1))
        return torch.stack(
            (
                active_count / actuator_scale,
                l1 / actuator_scale,
                l2 / actuator_scale**0.5,
                linf,
                signed_sum / actuator_scale,
                square_sum / actuator_scale,
                cumulative_l1 / (actuator_scale * step_scale),
                cumulative_l2 / (actuator_scale * step_scale).sqrt(),
                changed_blocks / block_scale,
                duration_since_first / duration_scale,
                maximum_simultaneous / actuator_scale,
            ),
            dim=-1,
        )

    def forward_group(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV41,
        elapsed_seconds: torch.Tensor,
        *,
        source_kind: str,
    ) -> CounterfactualGroupResponseV41:
        """Evaluate one or more complete candidate sets in parallel."""

        if candidate_settings.dim() != 4:
            raise ValueError("candidate_settings must be [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if reference_settings.shape != (batch, horizon, actuators):
            raise ValueError("reference and candidate setting shapes differ")
        if actuators != self.actuator_count:
            raise ValueError("candidate actuator count does not match model")
        reference = self.encode_reference(
            initial_state, rainfall, reference_settings, previous_actuator_flow, prepared
        )
        allow_interaction = self._allows_interaction(source_kind)
        actual_hidden, zero_hidden, delta_u = self._candidate_tokens(
            reference_settings, candidate_settings, reference.actuator_static
        )
        actuator_action_mask = delta_u.detach().ne(0.0).unsqueeze(-1)
        hidden_delta = torch.where(
            actuator_action_mask, actual_hidden - zero_hidden, torch.zeros_like(actual_hidden)
        )
        causal_action_mask = delta_u.detach().abs().sum(dim=-1).cumsum(dim=2).ne(0.0)
        candidate_action_mask = causal_action_mask.any(dim=2)

        single_flow_impulse = self.single_flow_head(actual_hidden) - self.single_flow_head(zero_hidden)
        single_flow_impulse = torch.where(
            actuator_action_mask, single_flow_impulse, torch.zeros_like(single_flow_impulse)
        )
        single_flow = (
            _causal_accumulate(single_flow_impulse).squeeze(-1) * self.d2_flow_scale
        )
        single_flow = torch.where(
            causal_action_mask.unsqueeze(-1), single_flow, torch.zeros_like(single_flow)
        )

        endpoint_actual = self.single_state_head(actual_hidden).reshape(
            batch, candidates, horizon, actuators, 2, self.state_dim
        )
        endpoint_zero = self.single_state_head(zero_hidden).reshape_as(endpoint_actual)
        endpoint_impulse = torch.where(
            actuator_action_mask.unsqueeze(-1),
            endpoint_actual - endpoint_zero,
            torch.zeros_like(endpoint_actual),
        )
        endpoint_effect = _causal_accumulate(
            endpoint_impulse.reshape(
                batch, candidates, horizon, actuators, 2 * self.state_dim
            )
        ).reshape(batch, candidates, horizon, actuators, 2, self.state_dim)
        outgoing, _ = _directed_scatter(
            endpoint_effect[..., 0, :],
            reference.actuator_upstream,
            reference.actuator_downstream,
            initial_state.shape[1],
        )
        _, incoming = _directed_scatter(
            endpoint_effect[..., 1, :],
            reference.actuator_upstream,
            reference.actuator_downstream,
            initial_state.shape[1],
        )
        network_actual = self.single_network_coefficient_head(actual_hidden)
        network_zero = self.single_network_coefficient_head(zero_hidden)
        network_impulse = torch.where(
            actuator_action_mask,
            network_actual - network_zero,
            torch.zeros_like(network_actual),
        )
        network_coefficients = _causal_accumulate(network_impulse).reshape(
            batch,
            candidates,
            horizon,
            actuators,
            self.effect_rank,
            self.state_dim,
        )
        network_coefficients = network_coefficients.sum(dim=3)
        node_basis = torch.tanh(self.single_node_basis_head(reference.node_context))
        network_state = torch.einsum(
            "bchrs,bhnr->bchns", network_coefficients, node_basis
        )
        single_state_raw = (
            outgoing + incoming + network_state
        ) * self.d2_state_scale
        single_state_raw = torch.where(
            causal_action_mask[..., None, None], single_state_raw, torch.zeros_like(single_state_raw)
        )

        active_now = delta_u.detach().ne(0.0).sum(dim=-1).clamp_min(1)
        active_so_far = delta_u.detach().abs().cumsum(dim=2).ne(0.0).sum(dim=-1)
        interaction_time_gate = active_so_far.ge(2).to(initial_state.dtype)
        if not allow_interaction:
            interaction_time_gate = torch.zeros_like(interaction_time_gate)
        interaction_gate = interaction_time_gate.amax(dim=2)

        active_normalizer = active_now.to(initial_state.dtype).sqrt().unsqueeze(-1)
        pooled_hidden = hidden_delta.sum(dim=3) / active_normalizer
        pair_denominator = (
            active_now.to(initial_state.dtype)
            * (active_now.to(initial_state.dtype) - 1.0).clamp_min(1.0)
        ).unsqueeze(-1)
        hidden_sum = hidden_delta.sum(dim=3)
        pair_moment = (
            hidden_sum.square() - hidden_delta.square().sum(dim=3)
        ) / pair_denominator
        identity_moment = (
            delta_u[..., None] * reference.actuator_static[None, None, None]
        ).sum(dim=3) / active_normalizer
        pooled_abs = delta_u.abs().sum(dim=3, keepdim=True) / active_normalizer
        pooled_signed = delta_u.sum(dim=3, keepdim=True) / active_normalizer
        pooled_square = delta_u.square().sum(dim=3, keepdim=True) / active_normalizer
        magnitude_features = self.interaction_magnitude_features(delta_u)
        context = reference.global_context[:, None].expand(-1, candidates, -1, -1)
        interaction_input = torch.cat(
            (
            pooled_hidden,
            pair_moment,
            identity_moment,
            context,
            pooled_abs,
            pooled_signed,
            pooled_square,
            ),
            dim=-1,
        )
        interaction_hidden = self.interaction_encoder(interaction_input)
        if self.interaction_magnitude_features_enabled:
            magnitude_hidden = self.interaction_magnitude_encoder(magnitude_features)
            interaction_hidden = interaction_hidden + self.interaction_magnitude_residual(
                magnitude_hidden
            )
        topology_state_raw, topology_hidden = self._topology_interaction(
            hidden_delta=hidden_delta,
            delta_u=delta_u,
            interaction_hidden=interaction_hidden,
            reference=reference,
            prepared=prepared,
            interaction_time_gate=interaction_time_gate,
        )
        if topology_hidden is not None:
            interaction_hidden = interaction_hidden + topology_hidden
        gate_h = interaction_time_gate[..., None]
        interaction_hidden = interaction_hidden * gate_h

        flow_context = interaction_hidden[:, :, :, None].expand(-1, -1, -1, actuators, -1)
        interaction_flow = self.interaction_flow_head(torch.cat((flow_context, hidden_delta), dim=-1)).squeeze(-1)
        interaction_flow = _causal_accumulate(interaction_flow.unsqueeze(-1)).squeeze(-1)
        interaction_flow = interaction_flow * self.d3_flow_scale * gate_h

        nodes = initial_state.shape[1]
        interaction_coefficients = self.interaction_state_head(interaction_hidden).reshape(
            batch, candidates, horizon, self.effect_rank, self.state_dim
        )
        interaction_state_raw = torch.einsum(
            "bchrs,bhnr->bchns", interaction_coefficients, node_basis
        )
        if topology_state_raw is not None:
            interaction_state_raw = interaction_state_raw + topology_state_raw
        interaction_state_raw = (
            interaction_state_raw * self.d3_state_scale * gate_h.unsqueeze(-1)
        )

        ref_flood_latent = reference.reference_flood_latent[:, None].expand(-1, candidates, -1, -1)
        single_flood_latent = single_state_raw[..., 2]
        interaction_flood_latent = interaction_state_raw[..., 2]
        reference_flood = F.softplus(ref_flood_latent)
        single_flood = F.softplus(ref_flood_latent + single_flood_latent) - reference_flood
        total_flood = F.softplus(
            ref_flood_latent + single_flood_latent + interaction_flood_latent
        ) - reference_flood
        interaction_flood = total_flood - single_flood

        single_state = torch.cat(
            (
                single_state_raw[..., 0:1],
                single_state_raw[..., 0:1],
                single_flood.unsqueeze(-1),
                single_state_raw[..., 3:],
            ),
            dim=-1,
        )
        single_state = torch.where(
            causal_action_mask[..., None, None], single_state, torch.zeros_like(single_state)
        )
        interaction_state = torch.cat(
            (
                interaction_state_raw[..., 0:1],
                interaction_state_raw[..., 0:1],
                interaction_flood.unsqueeze(-1),
                interaction_state_raw[..., 3:],
            ),
            dim=-1,
        )
        delta_state = single_state + interaction_state

        reference_states = reference.reference_states_physical
        candidate_depth = reference_states[:, None, ..., 0] + delta_state[..., 0]
        candidate_flood = F.softplus(
            ref_flood_latent + single_flood_latent + interaction_flood_latent
        )
        calculated_candidate_states = torch.cat(
            (
                candidate_depth.unsqueeze(-1),
                (
                    candidate_depth
                    + reference.invert_elevation_m.reshape(1, 1, 1, nodes)
                ).unsqueeze(-1),
                candidate_flood.unsqueeze(-1),
                reference_states[:, None, ..., 3:] + delta_state[..., 3:],
            ),
            dim=-1,
        )
        candidate_states = torch.where(
            causal_action_mask[..., None, None],
            calculated_candidate_states,
            reference_states[:, None].expand(-1, candidates, -1, -1, -1),
        )
        delta_state = torch.where(
            causal_action_mask[..., None, None],
            candidate_states - reference_states[:, None],
            torch.zeros_like(candidate_states),
        )
        interaction_state = torch.where(
            interaction_time_gate[..., None, None].ne(0.0),
            delta_state - single_state,
            torch.zeros_like(delta_state),
        )

        global_context = context.mean(dim=2)
        single_direct_actual = self.direct_single_tfv_head(
            torch.cat((actual_hidden, context[:, :, :, None].expand(-1, -1, -1, actuators, -1)), dim=-1)
        )
        single_direct_zero = self.direct_single_tfv_head(
            torch.cat((zero_hidden, context[:, :, :, None].expand(-1, -1, -1, actuators, -1)), dim=-1)
        )
        direct_single_effect = torch.where(
            actuator_action_mask,
            single_direct_actual - single_direct_zero,
            torch.zeros_like(single_direct_actual),
        )
        direct_single = direct_single_effect.mean(dim=2).sum(dim=2).squeeze(-1)
        direct_single = direct_single * self.d2_tfv_scale
        direct_single = torch.where(
            candidate_action_mask, direct_single, torch.zeros_like(direct_single)
        )
        direct_interaction = self.direct_interaction_tfv_head(
            torch.cat((interaction_hidden.mean(dim=2), global_context), dim=-1)
        ).squeeze(-1)
        zero_interaction = self.direct_interaction_tfv_head(
            torch.cat((torch.zeros_like(global_context), global_context), dim=-1)
        ).squeeze(-1)
        direct_interaction = (
            (direct_interaction - zero_interaction)
            * self.d3_tfv_scale
            * interaction_gate
        )
        direct_tfv = direct_single + direct_interaction

        delta_flow = single_flow + interaction_flow
        delta_flow = torch.where(
            causal_action_mask.unsqueeze(-1), delta_flow, torch.zeros_like(delta_flow)
        )
        candidate_flows = reference.reference_flows_physical[:, None] + delta_flow
        trajectory_tfv = _trapezoid_delta_tfv(delta_state[..., 2], elapsed_seconds)
        return CounterfactualGroupResponseV41(
            reference_states_physical=reference_states,
            candidate_states_physical=candidate_states,
            single_delta_states_physical=single_state,
            interaction_delta_states_physical=interaction_state,
            delta_states_physical=delta_state,
            reference_flows_physical=reference.reference_flows_physical,
            candidate_flows_physical=candidate_flows,
            single_delta_flows_physical=single_flow,
            interaction_delta_flows_physical=interaction_flow,
            delta_flows_physical=delta_flow,
            direct_single_delta_tfv_m3=direct_single,
            direct_interaction_delta_tfv_m3=direct_interaction,
            direct_delta_tfv_m3=direct_tfv,
            trajectory_delta_tfv_m3=trajectory_tfv,
        )


__all__ = [
    "CounterfactualGroupResponseV41",
    "DifferentiableCounterfactualResponseModelV41",
    "PreparedStaticV41",
    "ReferenceEncodingV41",
]
