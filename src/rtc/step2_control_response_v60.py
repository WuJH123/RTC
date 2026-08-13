"""Project7 Step2 V6.0: state-conditioned dual surrogate with stable latent control dynamics.

This module deliberately does not inherit any V4/V5 response model. It uses the
frozen GraphSchema only for static topology/actuator metadata. The value and
hydraulic surrogates own separate trainable encoders and latent operators so their
gradients cannot interfere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .step2_v60_contract import MultiResolutionHorizonV60, require_feature

_SAFE_ACTUATOR_FEATURES = (
    "is_pump", "is_orifice", "is_weir", "is_outlet", "min_setting", "max_setting",
    "pump_curve_max_flow_m3s", "pump_curve_max_x_m", "pump_curve_point_count",
    "offset_or_crest_m", "discharge_coefficient", "has_flap_gate",
    "xsection_is_circular", "xsection_is_rect_closed", "xsection_is_rect_open",
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


def _robust_signed_log(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Static transform using mean non-zero magnitude; no outcome statistics."""
    abs_values = values.detach().abs()
    nonzero = abs_values > 1e-8
    count = nonzero.sum(dim=dim).clamp_min(1)
    scale = abs_values.sum(dim=dim) / count
    scale = torch.where(nonzero.sum(dim=dim) > 0, scale.clamp_min(1e-6), torch.ones_like(scale))
    while scale.ndim < values.ndim:
        scale = scale.unsqueeze(dim)
    return values.sign() * torch.log1p(abs_values / scale)


def _causal_ema(values: torch.Tensor, rho: float = 0.85) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError("causal EMA expects [B,H,N,D]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0,1)")
    memory = torch.zeros_like(values[:, 0])
    result: list[torch.Tensor] = []
    for step in range(values.shape[1]):
        memory = rho * memory + (1.0 - rho) * values[:, step]
        result.append(memory)
    return torch.stack(result, dim=1)


def _scatter_actuators_to_nodes(values: torch.Tensor, upstream: torch.Tensor, downstream: torch.Tensor, node_count: int) -> torch.Tensor:
    if values.shape[-2] != upstream.numel() or upstream.shape != downstream.shape:
        raise ValueError("actuator values/endpoints are misaligned")
    leading, hidden = values.shape[:-2], values.shape[-1]
    flat = values.reshape(-1, values.shape[-2], hidden)
    out = values.new_zeros(flat.shape[0], node_count, hidden)
    out.index_add_(1, upstream, 0.5 * flat)
    out.index_add_(1, downstream, 0.5 * flat)
    return out.reshape(*leading, node_count, hidden)


def _integrate_rate_m3(rate_m3s: torch.Tensor, elapsed_seconds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if rate_m3s.ndim != 3 or elapsed_seconds.ndim != 2 or elapsed_seconds.shape[1] != rate_m3s.shape[2] + 1:
        raise ValueError("TFV integration expects rate [B,C,H] and elapsed [B,H+1]")
    dt = (elapsed_seconds[:, 1:] - elapsed_seconds[:, :-1]).clamp_min(0.0)
    increments = rate_m3s * dt[:, None, :]
    return increments.sum(dim=2), increments.cumsum(dim=2)


@dataclass(frozen=True)
class PreparedStaticV60:
    node_static: torch.Tensor
    actuator_physics: torch.Tensor
    actuator_upstream: torch.Tensor
    actuator_downstream: torch.Tensor
    invert_elevation_m: torch.Tensor
    max_depth_m: torch.Tensor
    surcharge_depth_m: torch.Tensor
    storage_capacity_m3: torch.Tensor
    storage_mask: torch.Tensor
    actuator_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ValueOutputV60:
    delta_tfv_m3: torch.Tensor
    delta_tfv_prefix_m3: torch.Tensor
    delta_rate_m3s: torch.Tensor
    candidate_latent: torch.Tensor
    reference_latent: torch.Tensor
    joint_context_before_scatter: torch.Tensor


@dataclass(frozen=True)
class HydraulicOutputV60:
    horizon_indices: torch.Tensor
    reference_states_physical: torch.Tensor
    candidate_states_physical: torch.Tensor
    delta_states_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    candidate_flows_physical: torch.Tensor
    delta_flows_physical: torch.Tensor
    reference_flood_onset_logits: torch.Tensor
    candidate_flood_onset_logits: torch.Tensor
    joint_context_before_scatter: torch.Tensor
    # Optional V9 signed-effect fields.  V6/V7/V8 callers continue to use the
    # projected ``delta_*`` fields; V9 trains only on these raw signed tensors.
    raw_delta_states_physical: torch.Tensor | None = None
    raw_delta_flows_physical: torch.Tensor | None = None


def prepare_static_v60(graph: Any, device: torch.device | str = "cpu") -> PreparedStaticV60:
    target = torch.device(device)
    node_names = tuple(graph.static_node_feature_names)
    physics_names = tuple(graph.actuator_physics_feature_names)
    safe_indices = [require_feature(physics_names, name) for name in _SAFE_ACTUATOR_FEATURES]
    node_raw = torch.as_tensor(np.asarray(graph.static_node_features), dtype=torch.float32, device=target)
    physics_raw = torch.as_tensor(np.asarray(graph.actuator_physics)[:, safe_indices], dtype=torch.float32, device=target)
    return PreparedStaticV60(
        node_static=_robust_signed_log(node_raw, dim=0),
        actuator_physics=_robust_signed_log(physics_raw, dim=0),
        actuator_upstream=torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=target),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=target),
        invert_elevation_m=node_raw[:, require_feature(node_names, "invert_elevation_m")],
        max_depth_m=node_raw[:, require_feature(node_names, "max_depth_m")].clamp_min(0.0),
        surcharge_depth_m=node_raw[:, require_feature(node_names, "surcharge_depth_m")].clamp_min(0.0),
        storage_capacity_m3=node_raw[:, require_feature(node_names, "storage_capacity_m3")].clamp_min(0.0),
        storage_mask=node_raw[:, require_feature(node_names, "is_storage")] > 0.5,
        actuator_feature_names=_SAFE_ACTUATOR_FEATURES,
    )


class CausalBoundaryEncoderV60(nn.Module):
    """Hydraulic state, rainfall and reference control policy are joint boundaries."""
    def __init__(self, *, state_dim: int, rainfall_dim: int, node_static_dim: int, physics_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_initial = _mlp(state_dim + node_static_dim, hidden_dim, hidden_dim)
        self.rainfall = _mlp(rainfall_dim, hidden_dim, hidden_dim)
        self.reference_action = _mlp(physics_dim + 1, hidden_dim, hidden_dim)
        self.context = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, initial_state: torch.Tensor, rainfall: torch.Tensor, reference_settings: torch.Tensor, prepared: PreparedStaticV60) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state.ndim != 3 or rainfall.ndim != 4 or reference_settings.ndim != 3:
            raise ValueError("boundary inputs must be [B,N,S], [B,H,N,R], [B,H,A]")
        batch, horizon, nodes, _ = rainfall.shape
        if initial_state.shape[:2] != (batch, nodes):
            raise ValueError("initial/rainfall node dimensions differ")
        static = prepared.node_static[None].expand(batch, -1, -1)
        initial_hidden = self.node_initial(torch.cat((initial_state, static), dim=-1))
        rain_hidden = _causal_ema(self.rainfall(rainfall))
        physics = prepared.actuator_physics[None, None].expand(batch, horizon, -1, -1)
        ref_action = self.reference_action(torch.cat((physics, reference_settings[..., None]), dim=-1))
        ref_nodes = _scatter_actuators_to_nodes(ref_action, prepared.actuator_upstream, prepared.actuator_downstream, nodes)
        ref_memory = _causal_ema(ref_nodes)
        context = self.context(torch.cat((initial_hidden[:, None].expand(-1, horizon, -1, -1), rain_hidden, ref_memory), dim=-1))
        return context, torch.cat((context.mean(dim=2), context.amax(dim=2)), dim=-1)


class StateConditionedActuatorInteractionV60(nn.Module):
    """h[a,t]=phi(u, local upstream/downstream hydraulics, rainfall context, physics) before actuator interaction."""
    def __init__(self, *, physics_dim: int, hidden_dim: int, temporal_dim: int, actuator_count: int, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must divide attention heads")
        self.actuator_count, self.hidden_dim = int(actuator_count), int(hidden_dim)
        self.temporal = nn.Embedding(72, temporal_dim)
        self.token_encoder = _mlp(physics_dim + 2 * hidden_dim + 3 + temporal_dim, hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.output = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=0)

    def forward(self, settings: torch.Tensor, reference_settings: torch.Tensor, node_context: torch.Tensor, prepared: PreparedStaticV60) -> torch.Tensor:
        if settings.ndim != 4:
            raise ValueError("settings must be [B,C,H,A]")
        batch, candidates, horizon, actuators = settings.shape
        if actuators != self.actuator_count or reference_settings.shape != (batch, horizon, actuators):
            raise ValueError("action tensor violates actuator/horizon contract")
        up = node_context[:, :, prepared.actuator_upstream]
        down = node_context[:, :, prepared.actuator_downstream]
        dynamic = torch.cat((up, down), dim=-1)[:, None].expand(batch, candidates, horizon, actuators, 2 * self.hidden_dim)
        physics = prepared.actuator_physics[None, None, None].expand(batch, candidates, horizon, -1, -1)
        ref = reference_settings[:, None].expand(-1, candidates, -1, -1)
        delta = settings - ref
        temporal = self.temporal(torch.arange(horizon, device=settings.device))[None, None, :, None].expand(batch, candidates, horizon, actuators, -1)
        token = self.token_encoder(torch.cat((physics, dynamic, ref[..., None], settings[..., None], delta[..., None], temporal), dim=-1))
        flat = token.reshape(batch * candidates * horizon, actuators, self.hidden_dim)
        attended, _ = self.attention(flat, flat, flat, need_weights=False)
        attended = attended.reshape(batch, candidates, horizon, actuators, self.hidden_dim)
        mean = attended.mean(dim=3, keepdim=True).expand_as(attended)
        maximum = attended.amax(dim=3, keepdim=True).expand_as(attended)
        joint = self.output(torch.cat((token + attended, mean, maximum), dim=-1))
        temp = joint.permute(0, 1, 3, 4, 2).reshape(batch * candidates * actuators, self.hidden_dim, horizon)
        temp = self.temporal_conv(F.pad(temp, (2, 0)))
        temp = torch.tanh(temp).reshape(batch, candidates, actuators, self.hidden_dim, horizon).permute(0, 1, 4, 2, 3)
        return joint + temp


class StableKoopmanControlOperatorV60(nn.Module):
    """Stable linear Koopman core plus a bounded nonlinear residual in latent space."""
    def __init__(self, *, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.initial = _mlp(2 * hidden_dim, latent_dim, latent_dim)
        self.control = nn.Linear(2 * hidden_dim, latent_dim, bias=False)
        self.boundary = nn.Linear(2 * hidden_dim, latent_dim, bias=False)
        self.raw_diag = nn.Parameter(torch.zeros(latent_dim))
        self.residual = _mlp(3 * latent_dim, latent_dim, latent_dim)

    def forward(self, global_boundary: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        if global_boundary.ndim != 3 or joint.ndim != 5:
            raise ValueError("Koopman inputs must be [B,H,2D] and [B,C,H,A,D]")
        batch, candidates, horizon = joint.shape[:3]
        pooled = torch.cat((joint.mean(dim=3), joint.amax(dim=3)), dim=-1)
        boundary = global_boundary[:, None].expand(batch, candidates, horizon, -1)
        z = self.initial(global_boundary[:, 0])[:, None].expand(batch, candidates, self.latent_dim)
        diag = 0.50 + 0.49 * torch.sigmoid(self.raw_diag)
        result: list[torch.Tensor] = []
        for step in range(horizon):
            u, b = self.control(pooled[:, :, step]), self.boundary(boundary[:, :, step])
            linear = diag * z + u + b
            z = linear + 0.10 * torch.tanh(self.residual(torch.cat((z, u, b), dim=-1)))
            result.append(z)
        return torch.stack(result, dim=2)


class _BaseSurrogateV60(nn.Module):
    def __init__(self, *, state_dim: int, rainfall_dim: int, node_static_dim: int, physics_dim: int, actuator_count: int, hidden_dim: int, latent_dim: int, temporal_dim: int) -> None:
        super().__init__()
        self.state_dim, self.actuator_count, self.hidden_dim = int(state_dim), int(actuator_count), int(hidden_dim)
        self.boundary = CausalBoundaryEncoderV60(state_dim=state_dim, rainfall_dim=rainfall_dim, node_static_dim=node_static_dim, physics_dim=physics_dim, hidden_dim=hidden_dim)
        self.interaction = StateConditionedActuatorInteractionV60(physics_dim=physics_dim, hidden_dim=hidden_dim, temporal_dim=temporal_dim, actuator_count=actuator_count)
        self.koopman = StableKoopmanControlOperatorV60(hidden_dim=hidden_dim, latent_dim=latent_dim)

    def _encode_pair(self, initial_state: torch.Tensor, rainfall: torch.Tensor, reference_settings: torch.Tensor, candidate_settings: torch.Tensor, prepared: PreparedStaticV60):
        batch, candidates, horizon, actuators = candidate_settings.shape
        node_context, global_context = self.boundary(initial_state, rainfall, reference_settings, prepared)
        reference_expanded = reference_settings[:, None].expand(batch, candidates, horizon, actuators)
        reference_joint = self.interaction(reference_expanded, reference_settings, node_context, prepared)
        candidate_joint = self.interaction(candidate_settings, reference_settings, node_context, prepared)
        reference_latent = self.koopman(global_context, reference_joint)
        candidate_latent = self.koopman(global_context, candidate_joint)
        return node_context, global_context, reference_joint, candidate_joint, reference_latent, candidate_latent


class ControlValueSurrogateV60(_BaseSurrogateV60):
    """MPC-facing surrogate trained only by exact TFV/ranking/regret objectives."""
    def __init__(self, *, tfv_rate_scale_m3s: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rate_head = _mlp(self.koopman.latent_dim + 2 * self.hidden_dim, self.hidden_dim, 1)
        self.register_buffer("tfv_rate_scale_m3s", torch.tensor(float(tfv_rate_scale_m3s)).clamp_min(1e-6))

    def forward(self, initial_state: torch.Tensor, rainfall: torch.Tensor, reference_settings: torch.Tensor, candidate_settings: torch.Tensor, prepared: PreparedStaticV60, elapsed_seconds: torch.Tensor) -> ValueOutputV60:
        _, global_context, _, candidate_joint, reference_latent, candidate_latent = self._encode_pair(initial_state, rainfall, reference_settings, candidate_settings, prepared)
        batch, candidates, horizon = candidate_latent.shape[:3]
        global_expanded = global_context[:, None].expand(batch, candidates, horizon, -1)
        ref_rate = F.softplus(self.rate_head(torch.cat((reference_latent, global_expanded), dim=-1)).squeeze(-1))
        cand_rate = F.softplus(self.rate_head(torch.cat((candidate_latent, global_expanded), dim=-1)).squeeze(-1))
        delta_rate = (cand_rate - ref_rate) * self.tfv_rate_scale_m3s.to(candidate_latent)
        delta_tfv, prefix = _integrate_rate_m3(delta_rate, elapsed_seconds)
        return ValueOutputV60(delta_tfv, prefix, delta_rate, candidate_latent, reference_latent, candidate_joint)


class HydraulicResponseSurrogateV60(_BaseSurrogateV60):
    """Independent multi-resolution hydraulic-response surrogate."""
    def __init__(self, *, state_scale: Sequence[float] | torch.Tensor, flow_scale: Sequence[float] | torch.Tensor, horizon_contract: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(), **kwargs: Any) -> None:
        node_static_dim = int(kwargs["node_static_dim"])
        if int(kwargs.get("state_dim", -1)) != 6:
            raise ValueError("V6 physical hydraulic decoder requires the frozen six-state channel contract")
        super().__init__(**kwargs)
        horizon_contract.validate()
        self.horizon_contract = horizon_contract
        latent_dim = self.koopman.latent_dim
        self.node_static_projection = nn.Linear(node_static_dim, self.hidden_dim)
        self.reference_state_latent = _mlp(2 * self.hidden_dim, self.hidden_dim, self.state_dim)
        self.effect_state_latent = _mlp(3 * self.hidden_dim + latent_dim, self.hidden_dim, self.state_dim)
        self.effect_flow_latent = _mlp(self.hidden_dim + latent_dim, self.hidden_dim, 1)
        self.flood_onset_head = _mlp(3 * self.hidden_dim + latent_dim, self.hidden_dim, 1)
        state_scale_tensor = torch.as_tensor(state_scale, dtype=torch.float32).reshape(-1)
        flow_scale_tensor = torch.as_tensor(flow_scale, dtype=torch.float32).reshape(-1)
        if state_scale_tensor.numel() != self.state_dim or flow_scale_tensor.numel() != self.actuator_count:
            raise ValueError("hydraulic scales do not match frozen state/actuator dimensions")
        self.register_buffer("state_scale", state_scale_tensor.clamp_min(1e-6))
        self.register_buffer("flow_scale", flow_scale_tensor.clamp_min(1e-6))

    def _physical_states(self, latent: torch.Tensor, prepared: PreparedStaticV60) -> torch.Tensor:
        scale = self.state_scale.to(latent).reshape(*([1] * (latent.ndim - 1)), -1)
        depth = F.softplus(latent[..., 0]) * scale[..., 0]
        flood = F.softplus(latent[..., 2]) * scale[..., 2]
        volume = F.softplus(latent[..., 3]) * scale[..., 3]
        inflow = F.softplus(latent[..., 4]) * scale[..., 4]
        outflow = F.softplus(latent[..., 5]) * scale[..., 5]
        invert = prepared.invert_elevation_m.to(latent)
        while invert.ndim < depth.ndim:
            invert = invert.unsqueeze(0)
        return torch.stack((depth, depth + invert, flood, volume, inflow, outflow), dim=-1)

    def forward(self, initial_state: torch.Tensor, rainfall: torch.Tensor, reference_settings: torch.Tensor, candidate_settings: torch.Tensor, prepared: PreparedStaticV60) -> HydraulicOutputV60:
        node_context, _, reference_joint, candidate_joint, reference_latent, candidate_latent = self._encode_pair(initial_state, rainfall, reference_settings, candidate_settings, prepared)
        indices = torch.as_tensor(self.horizon_contract.indices(), dtype=torch.long, device=initial_state.device)
        context = node_context.index_select(1, indices)
        batch, candidates, nodes = candidate_settings.shape[0], candidate_settings.shape[1], context.shape[2]
        ref_joint, cand_joint = reference_joint.index_select(2, indices), candidate_joint.index_select(2, indices)
        ref_z, cand_z = reference_latent.index_select(2, indices), candidate_latent.index_select(2, indices)
        ref_seed = _scatter_actuators_to_nodes(ref_joint, prepared.actuator_upstream, prepared.actuator_downstream, nodes)
        cand_seed = _scatter_actuators_to_nodes(cand_joint, prepared.actuator_upstream, prepared.actuator_downstream, nodes)
        context_expanded = context[:, None].expand(batch, candidates, -1, -1, -1)
        static_projected = self.node_static_projection(prepared.node_static).to(context)
        ref_base = self.reference_state_latent(torch.cat((context, static_projected[None, None].expand(batch, context.shape[1], -1, -1)), dim=-1))[:, None].expand(batch, candidates, -1, -1, -1)
        ref_z_nodes = ref_z[:, :, :, None].expand(-1, -1, -1, nodes, -1)
        cand_z_nodes = cand_z[:, :, :, None].expand(-1, -1, -1, nodes, -1)
        static_nodes = static_projected[None, None, None].expand(batch, candidates, context.shape[1], -1, -1)
        ref_input = torch.cat((context_expanded, ref_seed, static_nodes, ref_z_nodes), dim=-1)
        cand_input = torch.cat((context_expanded, cand_seed, static_nodes, cand_z_nodes), dim=-1)
        ref_effect, cand_effect = self.effect_state_latent(ref_input), self.effect_state_latent(cand_input)
        reference_states = self._physical_states(ref_base + ref_effect, prepared)
        candidate_states = self._physical_states(ref_base + cand_effect, prepared)
        ref_z_act = ref_z[:, :, :, None].expand(-1, -1, -1, self.actuator_count, -1)
        cand_z_act = cand_z[:, :, :, None].expand(-1, -1, -1, self.actuator_count, -1)
        ref_flow_raw = self.effect_flow_latent(torch.cat((ref_joint, ref_z_act), dim=-1)).squeeze(-1)
        cand_flow_raw = self.effect_flow_latent(torch.cat((cand_joint, cand_z_act), dim=-1)).squeeze(-1)
        flow_scale = self.flow_scale.to(ref_flow_raw).reshape(1, 1, 1, -1)
        reference_flows, candidate_flows = F.softplus(ref_flow_raw) * flow_scale, F.softplus(cand_flow_raw) * flow_scale
        # Preserve the V6 structural candidate/reference contract on CUDA as well
        # as CPU.  Identical action tensors can otherwise take separately laid-out
        # GEMM paths and differ by a few ulps before the physical transform.  The
        # equality branch is not a learned shortcut: it only applies to an exact
        # candidate==reference action sequence and keeps all non-zero counterfactual
        # gradients untouched.
        same_action = torch.all(
            candidate_settings == reference_settings[:, None], dim=(2, 3)
        )
        state_mask = same_action[..., None, None, None]
        flow_mask = same_action[..., None, None]
        reference_states_expanded = reference_states.expand_as(candidate_states)
        reference_flows_expanded = reference_flows.expand_as(candidate_flows)
        candidate_states = torch.where(state_mask, reference_states_expanded, candidate_states)
        candidate_flows = torch.where(flow_mask, reference_flows_expanded, candidate_flows)
        reference_logits = self.flood_onset_head(ref_input).squeeze(-1)
        candidate_logits = self.flood_onset_head(cand_input).squeeze(-1)
        candidate_logits = torch.where(
            same_action[..., None, None], reference_logits, candidate_logits
        )
        return HydraulicOutputV60(
            horizon_indices=indices,
            reference_states_physical=reference_states,
            candidate_states_physical=candidate_states,
            delta_states_physical=candidate_states - reference_states,
            reference_flows_physical=reference_flows,
            candidate_flows_physical=candidate_flows,
            delta_flows_physical=candidate_flows - reference_flows,
            reference_flood_onset_logits=reference_logits,
            candidate_flood_onset_logits=candidate_logits,
            joint_context_before_scatter=candidate_joint,
        )


class DualStep2SurrogateV60(nn.Module):
    def __init__(self, value_model: ControlValueSurrogateV60, hydraulic_model: HydraulicResponseSurrogateV60) -> None:
        super().__init__()
        self.value, self.hydraulic = value_model, hydraulic_model

    def assert_disjoint_parameters(self) -> None:
        if {id(p) for p in self.value.parameters()} & {id(p) for p in self.hydraulic.parameters()}:
            raise RuntimeError("V6 value/hydraulic models must not share trainable parameters")


__all__ = [
    "ControlValueSurrogateV60", "DualStep2SurrogateV60", "HydraulicOutputV60",
    "HydraulicResponseSurrogateV60", "PreparedStaticV60", "StableKoopmanControlOperatorV60",
    "StateConditionedActuatorInteractionV60", "ValueOutputV60", "prepare_static_v60",
]
