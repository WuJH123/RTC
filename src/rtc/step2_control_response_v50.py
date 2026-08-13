"""V5.0 state-conditioned candidate-manifold direct response surrogate.

The V5 operator is deliberately separate from the historical additive/residual
surrogates.  A frozen V4.2-compatible reference encoder supplies the causal
hydraulic context; a shared candidate/reference actuator-set operator then emits
two independent heads:

* ``ControlValueSurrogateV50`` predicts the exact cumulative ``delta TFV`` target.
* ``HydraulicResponseSurrogateV50`` predicts causal node/flow response diagnostics.

The heads have no shared trainable output trunk.  Candidate and reference actions
are evaluated by the same operator and subtracted structurally, so equal actions are
an exact zero response rather than a learned approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .step2_control_response_v41 import (
    DifferentiableCounterfactualResponseModelV41,
    PreparedStaticV41,
    _causal_accumulate,
    _directed_scatter,
    _mlp,
    _trapezoid_delta_tfv,
)


@dataclass(frozen=True)
class CounterfactualGroupResponseV50:
    reference_states_physical: torch.Tensor
    candidate_states_physical: torch.Tensor
    delta_states_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    candidate_flows_physical: torch.Tensor
    delta_flows_physical: torch.Tensor
    direct_delta_tfv_m3: torch.Tensor
    direct_delta_tfv_prefix_m3: torch.Tensor
    trajectory_delta_tfv_m3: torch.Tensor
    joint_context_before_node_scatter: torch.Tensor


class StateConditionedActuatorSetEncoderV50(nn.Module):
    """Finite-capacity permutation-equivariant set encoder with causal time conv."""

    def __init__(
        self,
        *,
        token_input_dim: int,
        hidden_dim: int,
        actuator_count: int,
        control_blocks: int = 36,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_input_dim = int(token_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.actuator_count = int(actuator_count)
        self.control_blocks = int(control_blocks)
        self.token_encoder = _mlp(self.token_input_dim, self.hidden_dim, self.hidden_dim)
        self.set_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, batch_first=True
        )
        self.joint_projection = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.temporal_conv = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, padding=0
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("tokens must have shape [B,C,H,A,D]")
        batch, candidates, horizon, actuators, _ = tokens.shape
        if actuators != self.actuator_count:
            raise ValueError("token actuator count does not match encoder")
        if horizon % self.control_blocks:
            raise ValueError("horizon must divide into control blocks")
        block_steps = horizon // self.control_blocks
        token_hidden = self.token_encoder(tokens)
        block_hidden = token_hidden.reshape(
            batch, candidates, self.control_blocks, block_steps, actuators, self.hidden_dim
        ).mean(dim=3)
        flat = block_hidden.reshape(batch * candidates * self.control_blocks, actuators, self.hidden_dim)
        attended, _ = self.set_attention(flat, flat, flat, need_weights=False)
        attended = attended.reshape(batch, candidates, self.control_blocks, actuators, self.hidden_dim)
        mean_context = attended.mean(dim=3, keepdim=True).expand_as(attended)
        joint = self.joint_projection(torch.cat((attended, mean_context), dim=-1))

        # Causal convolution is performed independently for each actuator.  Left
        # padding, rather than symmetric padding, prevents future control blocks
        # from entering an earlier response.
        temporal = joint.permute(0, 1, 3, 4, 2).reshape(
            batch * candidates * actuators, self.hidden_dim, self.control_blocks
        )
        temporal = self.temporal_conv(F.pad(temporal, (2, 0)))
        temporal = F.silu(temporal).reshape(
            batch, candidates, actuators, self.hidden_dim, self.control_blocks
        ).permute(0, 1, 4, 2, 3)
        temporal = temporal.repeat_interleave(block_steps, dim=2)
        return joint.repeat_interleave(block_steps, dim=2) + temporal


class ControlValueSurrogateV50(nn.Module):
    """Primary differentiable control-value head; no hydraulic-head parameters."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.rate_head = _mlp(2 * hidden_dim, hidden_dim, 1)

    def forward(self, joint_hidden: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        pooled = joint_hidden.mean(dim=3)
        return self.rate_head(torch.cat((pooled, global_context), dim=-1)).squeeze(-1)


class HydraulicResponseSurrogateV50(nn.Module):
    """Independent node/flow response head used for world-model diagnostics."""

    def __init__(self, *, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.node_head = _mlp(3 * hidden_dim, hidden_dim, state_dim)
        self.flow_head = _mlp(2 * hidden_dim, hidden_dim, 1)

    def forward_nodes(
        self, node_seed: torch.Tensor, node_context: torch.Tensor, node_static: torch.Tensor
    ) -> torch.Tensor:
        return self.node_head(torch.cat((node_seed, node_context, node_static), dim=-1))

    def forward_flow(self, joint_hidden: torch.Tensor, actuator_static: torch.Tensor) -> torch.Tensor:
        static = actuator_static[None, None, None].expand(
            joint_hidden.shape[0], joint_hidden.shape[1], joint_hidden.shape[2], -1, -1
        )
        return self.flow_head(torch.cat((joint_hidden, static), dim=-1)).squeeze(-1)


class StateConditionedCandidateManifoldModelV50(nn.Module):
    """Unified wrapper exposing decoupled value and hydraulic surrogate heads."""

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        node_static_dim: int,
        actuator_physics_dim: int,
        hidden_dim: int = 64,
        actuator_count: int = 109,
        actuator_embedding_dim: int = 16,
        temporal_embedding_dim: int = 12,
        max_horizon_steps: int = 72,
        control_blocks: int = 36,
        effect_rank: int = 12,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        flow_std: torch.Tensor | None = None,
        d2_state_scale: torch.Tensor | None = None,
        d3_state_scale: torch.Tensor | None = None,
        d2_flow_scale: torch.Tensor | None = None,
        d3_flow_scale: torch.Tensor | None = None,
        d2_tfv_scale: float = 1.0,
        d3_tfv_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.rainfall_dim = int(rainfall_dim)
        self.hidden_dim = int(hidden_dim)
        self.actuator_count = int(actuator_count)
        self.max_horizon_steps = int(max_horizon_steps)
        self.control_blocks = int(control_blocks)
        self.temporal_embedding_dim = int(temporal_embedding_dim)
        zeros_state = torch.zeros(self.state_dim) if state_mean is None else torch.as_tensor(state_mean)
        ones_state = torch.ones(self.state_dim) if state_std is None else torch.as_tensor(state_std)
        ones_flow = torch.ones(self.actuator_count) if flow_std is None else torch.as_tensor(flow_std)
        state_scale_d2 = torch.ones(self.state_dim) if d2_state_scale is None else torch.as_tensor(d2_state_scale)
        state_scale_d3 = torch.ones(self.state_dim) if d3_state_scale is None else torch.as_tensor(d3_state_scale)
        flow_scale_d2 = torch.ones(self.actuator_count) if d2_flow_scale is None else torch.as_tensor(d2_flow_scale)
        flow_scale_d3 = torch.ones(self.actuator_count) if d3_flow_scale is None else torch.as_tensor(d3_flow_scale)
        self.reference_backbone = DifferentiableCounterfactualResponseModelV41(
            state_dim=self.state_dim,
            rainfall_dim=self.rainfall_dim,
            node_static_dim=int(node_static_dim),
            actuator_physics_dim=int(actuator_physics_dim),
            hidden_dim=self.hidden_dim,
            actuator_count=self.actuator_count,
            actuator_embedding_dim=int(actuator_embedding_dim),
            temporal_embedding_dim=self.temporal_embedding_dim,
            state_mean=zeros_state,
            state_std=ones_state,
            flow_std=ones_flow,
            d2_state_scale=state_scale_d2,
            d3_state_scale=state_scale_d3,
            d2_flow_scale=flow_scale_d2,
            d3_flow_scale=flow_scale_d3,
            d2_tfv_scale=float(d2_tfv_scale),
            d3_tfv_scale=float(d3_tfv_scale),
            max_horizon_steps=self.max_horizon_steps,
            effect_rank=int(effect_rank),
            interaction_magnitude_features_enabled=True,
        )
        self.temporal_identity = nn.Embedding(self.max_horizon_steps, self.temporal_embedding_dim)
        # actuator-static hidden + two endpoint dynamic contexts + four scalar
        # action/flow channels + causal temporal identity
        token_dim = 3 * self.hidden_dim + 4 + self.temporal_embedding_dim
        self.action_set_encoder = StateConditionedActuatorSetEncoderV50(
            token_input_dim=token_dim,
            hidden_dim=self.hidden_dim,
            actuator_count=self.actuator_count,
            control_blocks=self.control_blocks,
        )
        self.control_value = ControlValueSurrogateV50(self.hidden_dim)
        self.hydraulic_response = HydraulicResponseSurrogateV50(
            hidden_dim=self.hidden_dim, state_dim=self.state_dim
        )
        self.register_buffer("d2_state_scale", state_scale_d2.reshape(-1).float().clamp_min(1e-6))
        self.register_buffer("d3_state_scale", state_scale_d3.reshape(-1).float().clamp_min(1e-6))
        self.register_buffer("d2_flow_scale", flow_scale_d2.reshape(-1).float().clamp_min(1e-6))
        self.register_buffer("d3_flow_scale", flow_scale_d3.reshape(-1).float().clamp_min(1e-6))
        self.register_buffer("d2_tfv_scale", torch.tensor(float(d2_tfv_scale)).clamp_min(1.0))
        self.register_buffer("d3_tfv_scale", torch.tensor(float(d3_tfv_scale)).clamp_min(1.0))

    def prepare_static(self, **kwargs: Any) -> PreparedStaticV41:
        return self.reference_backbone.prepare_static(**kwargs)

    def freeze_reference(self) -> None:
        for parameter in self.reference_backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.reference_backbone.eval()

    def set_value_trainable(self, enabled: bool = True) -> tuple[str, ...]:
        self.freeze_reference()
        for parameter in self.hydraulic_response.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.action_set_encoder.parameters():
            parameter.requires_grad_(enabled)
        for parameter in self.control_value.parameters():
            parameter.requires_grad_(enabled)
        return tuple(name for name, p in self.named_parameters() if p.requires_grad)

    def set_hydraulic_trainable(self, enabled: bool = True) -> tuple[str, ...]:
        self.freeze_reference()
        for parameter in self.action_set_encoder.parameters():
            # The set encoder is shared only as a frozen causal representation in
            # the hydraulic phase.  Updating it would silently change the value
            # surrogate even though its output head is frozen.
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.control_value.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.hydraulic_response.parameters():
            parameter.requires_grad_(enabled)
        return tuple(name for name, p in self.named_parameters() if p.requires_grad)

    def _source_scales(self, source_kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_kind.upper() == "D2":
            return self.d2_state_scale, self.d2_flow_scale, self.d2_tfv_scale
        if source_kind.upper() == "D3":
            return self.d3_state_scale, self.d3_flow_scale, self.d3_tfv_scale
        raise ValueError("source_kind must be D2 or D3")

    def _tokens(
        self,
        *,
        settings: torch.Tensor,
        reference_settings: torch.Tensor,
        reference: Any,
    ) -> torch.Tensor:
        batch, candidates, horizon, actuators = settings.shape
        device, dtype = settings.device, settings.dtype
        upstream = reference.actuator_upstream.to(device=device)
        downstream = reference.actuator_downstream.to(device=device)
        dynamic_up = reference.node_context[:, :, upstream]
        dynamic_down = reference.node_context[:, :, downstream]
        dynamic = torch.cat((dynamic_up, dynamic_down), dim=-1)
        dynamic = dynamic[:, None].expand(batch, candidates, horizon, actuators, 2 * self.hidden_dim)
        static = reference.actuator_static.to(device=device, dtype=dtype)[None, None, None].expand(
            batch, candidates, horizon, actuators, self.hidden_dim
        )
        ids = torch.arange(horizon, device=device)
        temporal = self.temporal_identity(ids).to(dtype=dtype)[None, None, :, None].expand(
            batch, candidates, horizon, actuators, self.temporal_embedding_dim
        )
        ref = reference_settings[:, None].expand(-1, candidates, -1, -1)
        delta = settings - ref
        flow_std = self.reference_backbone.flow_std.to(device=device, dtype=dtype).clamp_min(1e-6)
        flow = reference.reference_flows_physical[:, None].expand(batch, candidates, -1, -1) / flow_std
        flow = flow[..., None]
        return torch.cat((static, dynamic, temporal, ref[..., None], settings[..., None], delta[..., None], flow), dim=-1)

    def _absolute_response(
        self,
        *,
        settings: torch.Tensor,
        reference_settings: torch.Tensor,
        reference: Any,
        prepared: PreparedStaticV41,
        elapsed_seconds: torch.Tensor,
        source_kind: str,
    ) -> dict[str, torch.Tensor]:
        batch, candidates, horizon, actuators = settings.shape
        nodes = reference.node_context.shape[2]
        dtype, device = settings.dtype, settings.device
        tokens = self._tokens(settings=settings, reference_settings=reference_settings, reference=reference)
        joint = self.action_set_encoder(tokens)
        node_out, node_in = _directed_scatter(
            joint,
            reference.actuator_upstream.to(device=device),
            reference.actuator_downstream.to(device=device),
            nodes,
        )
        node_seed = node_out + node_in
        node_context = reference.node_context[:, None].expand(batch, candidates, horizon, nodes, self.hidden_dim).to(dtype=dtype)
        node_static = reference.node_static_context.to(device=device, dtype=dtype)[None, None, None].expand(
            batch, candidates, horizon, nodes, self.hidden_dim
        )
        state_scale, flow_scale, tfv_scale = self._source_scales(source_kind)
        state_scale = state_scale.to(device=device, dtype=dtype)
        flow_scale = flow_scale.to(device=device, dtype=dtype)
        tfv_scale = tfv_scale.to(device=device, dtype=dtype)
        raw_state = torch.tanh(self.hydraulic_response.forward_nodes(node_seed, node_context, node_static)) * state_scale
        reference_depth = reference.reference_states_physical[:, None, ..., 0].expand(batch, candidates, horizon, nodes)
        reference_flood_latent = reference.reference_flood_latent[:, None].expand(batch, candidates, horizon, nodes)
        depth_latent = reference_depth.clamp_min(1e-8).log() + raw_state[..., 0]
        depth = F.softplus(depth_latent)
        flood = F.softplus(reference_flood_latent + raw_state[..., 2])
        absolute_states = torch.cat(
            (
                depth.unsqueeze(-1),
                (depth + reference.invert_elevation_m.reshape(1, 1, 1, nodes)).unsqueeze(-1),
                flood.unsqueeze(-1),
                reference.reference_states_physical[:, None, ..., 3:] + raw_state[..., 3:],
            ),
            dim=-1,
        )
        raw_flow = self.hydraulic_response.forward_flow(joint, reference.actuator_static.to(device=device, dtype=dtype))
        absolute_flows = _causal_accumulate(raw_flow.unsqueeze(-1)).squeeze(-1) * flow_scale.reshape(1, 1, 1, actuators)
        global_context = reference.global_context[:, None].expand(batch, candidates, horizon, self.hidden_dim).to(dtype=dtype)
        value_rate = self.control_value(joint, global_context)
        elapsed_start = elapsed_seconds[..., :1]
        elapsed_end = elapsed_seconds[..., -1:]
        total = (elapsed_end - elapsed_start).clamp_min(1.0)
        normalized_elapsed = (elapsed_seconds - elapsed_start) / total
        absolute_tfv = _trapezoid_delta_tfv(value_rate.unsqueeze(-1), normalized_elapsed) * tfv_scale
        return {
            "states": absolute_states,
            "flows": absolute_flows,
            "tfv": absolute_tfv,
            "tfv_rate": value_rate,
            "joint": joint,
        }

    @staticmethod
    def _prefix_integral(rate: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        if rate.ndim != 3 or elapsed.shape[-1] != rate.shape[-1] + 1:
            raise ValueError("prefix integral expects rate [B,C,H] and elapsed [B,H+1]")
        elapsed0 = elapsed[..., :1]
        total = (elapsed[..., -1:] - elapsed0).clamp_min(1.0)
        norm_elapsed = (elapsed - elapsed0) / total
        dt = norm_elapsed[..., 1:] - norm_elapsed[..., :-1]
        previous = torch.zeros_like(rate[..., :1])
        values = torch.cat((previous, rate), dim=-1)
        increments = 0.5 * (values[..., 1:] + values[..., :-1]) * dt[:, None, :]
        return increments.cumsum(dim=-1)

    def forward_value_group(
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
    ) -> CounterfactualGroupResponseV50:
        """Value-only forward path; skips the node hydraulic decoder for efficiency."""

        if candidate_settings.ndim != 4:
            raise ValueError("candidate_settings must have shape [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count or horizon > self.max_horizon_steps:
            raise ValueError("candidate dimensions do not match V5 contract")
        with torch.no_grad():
            reference = self.reference_backbone.encode_reference(
                initial_state, rainfall, reference_settings, previous_actuator_flow, prepared
            )
        reference_expanded = reference_settings[:, None].expand(batch, candidates, horizon, actuators)
        reference_tokens = self._tokens(
            settings=reference_expanded,
            reference_settings=reference_settings,
            reference=reference,
        )
        candidate_tokens = self._tokens(
            settings=candidate_settings,
            reference_settings=reference_settings,
            reference=reference,
        )
        reference_joint = self.action_set_encoder(reference_tokens)
        candidate_joint = self.action_set_encoder(candidate_tokens)
        global_context = reference.global_context[:, None].expand(batch, candidates, horizon, self.hidden_dim).to(dtype=candidate_settings.dtype)
        reference_rate = self.control_value(reference_joint, global_context)
        candidate_rate = self.control_value(candidate_joint, global_context)
        elapsed_start = elapsed_seconds[..., :1]
        total = (elapsed_seconds[..., -1:] - elapsed_start).clamp_min(1.0)
        normalized_elapsed = (elapsed_seconds - elapsed_start) / total
        reference_tfv = _trapezoid_delta_tfv(reference_rate.unsqueeze(-1), normalized_elapsed) * self._source_scales(source_kind)[2]
        candidate_tfv = _trapezoid_delta_tfv(candidate_rate.unsqueeze(-1), normalized_elapsed) * self._source_scales(source_kind)[2]
        delta_u = candidate_settings - reference_expanded
        causal_mask = delta_u.detach().abs().sum(dim=-1).cumsum(dim=2).ne(0.0)
        delta_rate = candidate_rate - reference_rate
        direct_prefix = self._prefix_integral(delta_rate, elapsed_seconds) * self._source_scales(source_kind)[2].to(candidate_settings)
        direct_prefix = torch.where(causal_mask, direct_prefix, torch.zeros_like(direct_prefix))
        direct = candidate_tfv - reference_tfv
        direct = torch.where(causal_mask.any(dim=2), direct, torch.zeros_like(direct))
        zeros_state = initial_state.new_zeros(batch, candidates, horizon, initial_state.shape[1], self.state_dim)
        zeros_flow = initial_state.new_zeros(batch, candidates, horizon, actuators)
        return CounterfactualGroupResponseV50(
            reference_states_physical=reference.reference_states_physical,
            candidate_states_physical=reference.reference_states_physical[:, None].expand(-1, candidates, -1, -1, -1),
            delta_states_physical=zeros_state,
            reference_flows_physical=reference.reference_flows_physical,
            candidate_flows_physical=reference.reference_flows_physical[:, None].expand(-1, candidates, -1, -1),
            delta_flows_physical=zeros_flow,
            direct_delta_tfv_m3=direct,
            direct_delta_tfv_prefix_m3=direct_prefix,
            trajectory_delta_tfv_m3=torch.zeros_like(direct),
            joint_context_before_node_scatter=candidate_joint,
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
    ) -> CounterfactualGroupResponseV50:
        if candidate_settings.ndim != 4:
            raise ValueError("candidate_settings must have shape [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count or horizon > self.max_horizon_steps:
            raise ValueError("candidate dimensions do not match V5 contract")
        with torch.set_grad_enabled(any(p.requires_grad for p in self.reference_backbone.parameters())):
            reference = self.reference_backbone.encode_reference(
                initial_state, rainfall, reference_settings, previous_actuator_flow, prepared
            )
        reference_expanded = reference_settings[:, None].expand(batch, candidates, horizon, actuators)
        reference_eval = self._absolute_response(
            settings=reference_expanded,
            reference_settings=reference_settings,
            reference=reference,
            prepared=prepared,
            elapsed_seconds=elapsed_seconds,
            source_kind=source_kind,
        )
        candidate_eval = self._absolute_response(
            settings=candidate_settings,
            reference_settings=reference_settings,
            reference=reference,
            prepared=prepared,
            elapsed_seconds=elapsed_seconds,
            source_kind=source_kind,
        )
        delta_u = candidate_settings - reference_expanded
        causal_mask = delta_u.detach().abs().sum(dim=-1).cumsum(dim=2).ne(0.0)
        delta_states = candidate_eval["states"] - reference_eval["states"]
        delta_flows = candidate_eval["flows"] - reference_eval["flows"]
        direct_tfv = candidate_eval["tfv"] - reference_eval["tfv"]
        delta_states = torch.where(causal_mask[..., None, None], delta_states, torch.zeros_like(delta_states))
        delta_flows = torch.where(causal_mask.unsqueeze(-1), delta_flows, torch.zeros_like(delta_flows))
        direct_rate = candidate_eval["tfv_rate"] - reference_eval["tfv_rate"]
        direct_prefix = self._prefix_integral(direct_rate, elapsed_seconds)
        direct_prefix = torch.where(causal_mask, direct_prefix, torch.zeros_like(direct_prefix))
        direct_tfv = torch.where(causal_mask.any(dim=2), direct_tfv, torch.zeros_like(direct_tfv))
        trajectory = _trapezoid_delta_tfv(
            delta_states[..., 2],
            (elapsed_seconds - elapsed_seconds[..., :1]).clamp_min(0.0) / (elapsed_seconds[..., -1:] - elapsed_seconds[..., :1]).clamp_min(1.0),
        )
        reference_states = reference.reference_states_physical
        candidate_states = reference_states[:, None] + delta_states
        candidate_flows = reference.reference_flows_physical[:, None] + delta_flows
        zero_noop = ~causal_mask.any(dim=2)
        candidate_states = torch.where(
            (~zero_noop)[..., None, None, None], candidate_states,
            reference_states[:, None].expand_as(candidate_states),
        )
        candidate_flows = torch.where(
            (~zero_noop)[..., None, None], candidate_flows,
            reference.reference_flows_physical[:, None].expand_as(candidate_flows),
        )
        joint = candidate_eval["joint"]
        return CounterfactualGroupResponseV50(
            reference_states_physical=reference_states,
            candidate_states_physical=candidate_states,
            delta_states_physical=delta_states,
            reference_flows_physical=reference.reference_flows_physical,
            candidate_flows_physical=candidate_flows,
            delta_flows_physical=delta_flows,
            direct_delta_tfv_m3=direct_tfv,
            direct_delta_tfv_prefix_m3=direct_prefix,
            trajectory_delta_tfv_m3=trajectory,
            joint_context_before_node_scatter=joint,
        )


def load_v42_reference_into_v50(model: StateConditionedCandidateManifoldModelV50, checkpoint: str) -> dict[str, Any]:
    """Load only compatible frozen V4.2 reference/static parameters."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incoming = payload.get("model_state_dict", payload)
    current = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for name, value in incoming.items():
        target = f"reference_backbone.{name}"
        if target in current and tuple(current[target].shape) == tuple(value.shape):
            mapped[target] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    model.freeze_reference()
    return {"loaded": len(mapped), "missing": list(missing), "unexpected": list(unexpected)}


__all__ = [
    "ControlValueSurrogateV50",
    "CounterfactualGroupResponseV50",
    "HydraulicResponseSurrogateV50",
    "StateConditionedActuatorSetEncoderV50",
    "StateConditionedCandidateManifoldModelV50",
    "load_v42_reference_into_v50",
]
