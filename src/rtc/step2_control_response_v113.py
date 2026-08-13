"""V11.3 support-conditioned direct signed hydraulic-effect mechanism model.

The model is intentionally small and bounded for the nested development gate.
It learns a facility-flow response first, then transports that learned latent to
node domains using a frozen *soft* influence prior.  The prior is never a hard
mask and no signed D2 magnitudes are added.  Every output is anchored by a
causal action-prefix gate, so an unchanged candidate is exactly zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


STATE_EFFECT_NAMES_V113 = (
    "depth_m", "flooding_rate_m3s", "storage_volume_m3", "inflow_m3s", "outflow_m3s"
)
STATE_EFFECT_CHANNELS_V113 = (0, 2, 3, 4, 5)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden_dim)), nn.SiLU(),
        nn.Linear(int(hidden_dim), int(output_dim)),
    )


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.log1p(x.abs())


@dataclass(frozen=True)
class HydraulicEffectOutputV113:
    horizon_indices: torch.Tensor
    response_minutes: torch.Tensor
    raw_delta_states_physical: torch.Tensor  # [B,C,T,N,6]
    raw_delta_flows_physical: torch.Tensor  # [B,C,T,A]
    facility_flow_latent: torch.Tensor  # [B,C,T,A,H]
    state_support_context: torch.Tensor  # [B,C,T,N,5]
    prior_mode: str


class SupportConditionedHydraulicSurrogateV113(nn.Module):
    """Direct, signed, facility-first effect model for V113 development gates."""

    def __init__(
        self,
        *,
        node_static_dim: int,
        physics_dim: int,
        rainfall_dim: int,
        actuator_count: int,
        node_count: int,
        state_scale: np.ndarray | torch.Tensor,
        flow_scale: np.ndarray | torch.Tensor,
        overall_state_prior: np.ndarray | torch.Tensor,
        overall_flow_prior: np.ndarray | torch.Tensor,
        phase_state_priors: np.ndarray | torch.Tensor,
        phase_flow_priors: np.ndarray | torch.Tensor,
        phase_boundaries: tuple[float, float],
        retained_indices: Sequence[int],
        prior_mode: str = "phase",
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        if prior_mode not in {"none", "overall", "phase", "oracle"}:
            raise ValueError("V113 prior_mode must be none/overall/phase/oracle")
        self.node_count = int(node_count)
        self.actuator_count = int(actuator_count)
        self.rainfall_dim = int(rainfall_dim)
        self.hidden_dim = int(hidden_dim)
        self.prior_mode = str(prior_mode)
        retained = torch.as_tensor(tuple(int(x) for x in retained_indices), dtype=torch.long)
        if retained.ndim != 1 or retained.numel() == 0:
            raise ValueError("V113 retained indices must be non-empty")
        self.register_buffer("retained_indices", retained)
        self.register_buffer("upstream", torch.empty(0, dtype=torch.long))
        self.register_buffer("downstream", torch.empty(0, dtype=torch.long))
        state_scale_t = torch.as_tensor(state_scale, dtype=torch.float32)
        flow_scale_t = torch.as_tensor(flow_scale, dtype=torch.float32).reshape(-1)
        if state_scale_t.shape != (node_count, 5) or flow_scale_t.shape != (actuator_count,):
            raise ValueError("V113 scale dimensions do not match graph")
        self.register_buffer("state_scale", state_scale_t.clamp_min(1e-6))
        self.register_buffer("flow_scale", flow_scale_t.clamp_min(1e-6))

        def prior_tensor(value: np.ndarray | torch.Tensor, shape: tuple[int, ...], name: str) -> None:
            t = torch.as_tensor(value, dtype=torch.float32)
            if tuple(t.shape) != shape:
                raise ValueError(f"V113 {name} shape {tuple(t.shape)} != {shape}")
            self.register_buffer(name, t.clamp(0.0, 1.0))

        t_count = int(retained.numel())
        prior_tensor(overall_state_prior, (actuator_count, t_count, node_count, 5), "overall_state_prior")
        prior_tensor(overall_flow_prior, (actuator_count, t_count, actuator_count), "overall_flow_prior")
        phase_state_t = torch.as_tensor(phase_state_priors, dtype=torch.float32)
        phase_flow_t = torch.as_tensor(phase_flow_priors, dtype=torch.float32)
        if tuple(phase_state_t.shape) != (3, actuator_count, t_count, node_count, 5):
            raise ValueError("V113 phase state prior shape mismatch")
        if tuple(phase_flow_t.shape) != (3, actuator_count, t_count, actuator_count):
            raise ValueError("V113 phase flow prior shape mismatch")
        self.register_buffer("phase_state_priors", phase_state_t.clamp(0.0, 1.0))
        self.register_buffer("phase_flow_priors", phase_flow_t.clamp(0.0, 1.0))
        q1, q2 = (float(x) for x in phase_boundaries)
        if not (np.isfinite(q1) and np.isfinite(q2) and q1 < q2):
            raise ValueError("V113 phase boundaries must be finite and ordered")
        self.register_buffer("phase_boundaries", torch.tensor([q1, q2], dtype=torch.float32))

        # Explicit identity is required because many actuator physics rows are
        # duplicated.  The source head is facility-first, not a dense node decoder.
        self.actuator_identity = nn.Embedding(actuator_count, 12)
        self.node_identity = nn.Embedding(node_count, 8)
        source_features = (
            3  # current/prefix mean/absolute prefix action
            + 4  # reference/current/previous flow/prior source-flow support
            + 12  # local upstream/downstream six-state vectors
            + physics_dim + 12 + rainfall_dim + 4  # physics, identity, rain, time
        )
        self.facility_encoder = _mlp(source_features, hidden_dim, hidden_dim)
        self.facility_head = nn.Linear(hidden_dim, 1)
        self.facility_gate = nn.Linear(hidden_dim, 1)

        # Facility latent -> endpoint transport -> node/state heads.  A global
        # pooled path is retained deliberately; there is no hard graph-hop mask.
        node_features = 6 + node_static_dim + 8 + rainfall_dim + hidden_dim + 5 + hidden_dim + 4
        self.node_transport = _mlp(node_features, hidden_dim, hidden_dim)
        self.node_state_head = _mlp(hidden_dim, hidden_dim, 5)

    def bind_endpoints(self, upstream: torch.Tensor, downstream: torch.Tensor) -> None:
        up = torch.as_tensor(upstream, dtype=torch.long, device=self.retained_indices.device).reshape(-1)
        down = torch.as_tensor(downstream, dtype=torch.long, device=self.retained_indices.device).reshape(-1)
        if up.shape != (self.actuator_count,) or down.shape != up.shape:
            raise ValueError("V113 actuator endpoint dimensions do not match")
        if int(up.min()) < 0 or int(down.min()) < 0 or int(up.max()) >= self.node_count or int(down.max()) >= self.node_count:
            raise ValueError("V113 actuator endpoints outside node domain")
        self.upstream = up
        self.downstream = down

    def _phase_weights(self, initial_state_physical: torch.Tensor | None, batch: int) -> torch.Tensor:
        if initial_state_physical is None:
            return torch.zeros(batch, 3, device=self.retained_indices.device, dtype=self.state_scale.dtype).fill_(1.0 / 3.0)
        if initial_state_physical.ndim != 3 or initial_state_physical.shape[1] != self.node_count:
            raise ValueError("V113 physical initial state must be [B,N,6]")
        depth = initial_state_physical[..., 0].clamp_min(0.0)
        # max_depth is supplied through the caller's physical-state contract;
        # using a robust p90 depth proxy here keeps this a continuous prior mix.
        fill = torch.quantile(depth, 0.90, dim=1)
        q1, q2 = self.phase_boundaries.to(fill)
        centers = torch.stack((q1 * 0.5, (q1 + q2) * 0.5, q2 + 0.5 * (q2 - q1)))
        temperature = (q2 - q1).clamp_min(1e-2) * 0.25
        return torch.softmax(-torch.abs(fill[:, None] - centers[None]) / temperature, dim=-1)

    def _select_prior(
        self,
        initial_state_physical: torch.Tensor | None,
        batch: int,
        support_override: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support_override is not None:
            p = torch.as_tensor(support_override, dtype=self.state_scale.dtype, device=self.state_scale.device)
            if p.ndim != 6 or p.shape[0] != batch or p.shape[2:] != (self.actuator_count, self.retained_indices.numel(), self.node_count, 5):
                raise ValueError("V113 support override must be [B,C,A,T,N,5]")
            flow = self.overall_flow_prior[None, None].expand(batch, p.shape[1], -1, -1, -1)
            return p.clamp(0.0, 1.0), flow
        overall_state = self.overall_state_prior.to(self.state_scale)
        overall_flow = self.overall_flow_prior.to(self.state_scale)
        if self.prior_mode in {"none"}:
            return torch.zeros_like(overall_state)[None, None].expand(batch, 1, -1, -1, -1, -1), torch.zeros_like(overall_flow)[None, None].expand(batch, 1, -1, -1, -1)
        if self.prior_mode == "overall":
            return overall_state[None, None].expand(batch, 1, -1, -1, -1, -1), overall_flow[None, None].expand(batch, 1, -1, -1, -1)
        weights = self._phase_weights(initial_state_physical, batch).to(overall_state)
        state = torch.einsum("bp,patnk->batnk", weights, self.phase_state_priors.to(overall_state))
        flow = torch.einsum("bp,patq->batq", weights, self.phase_flow_priors.to(overall_flow))
        return state[:, None], flow[:, None]

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: object,
        *,
        initial_state_physical: torch.Tensor | None = None,
        support_override: torch.Tensor | None = None,
    ) -> HydraulicEffectOutputV113:
        if self.upstream.numel() == 0:
            self.bind_endpoints(prepared.actuator_upstream, prepared.actuator_downstream)
        if initial_state.ndim != 3 or initial_state.shape[-1] != 6:
            raise ValueError("V113 initial state must be [B,N,6]")
        if rainfall.ndim != 4 or candidate_settings.ndim != 4 or reference_settings.ndim != 3:
            raise ValueError("V113 input rank contract violated")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if (batch, initial_state.shape[1], actuators) != (initial_state.shape[0], self.node_count, self.actuator_count):
            raise ValueError("V113 input dimensions do not match graph")
        if reference_settings.shape != (batch, horizon, actuators) or previous_actuator_flow.shape != (batch, actuators):
            raise ValueError("V113 reference/previous-flow shape mismatch")
        indices = self.retained_indices.to(candidate_settings.device)
        state_prior, flow_prior = self._select_prior(initial_state_physical, batch, support_override)
        # The selected prior has [B,1,A,T,...] (or B,C override); broadcast to
        # candidates only after the joint action prefix is known.
        if state_prior.shape[1] == 1:
            state_prior = state_prior.expand(batch, candidates, -1, -1, -1, -1)
            flow_prior = flow_prior.expand(batch, candidates, -1, -1, -1)

        node_static = torch.as_tensor(prepared.node_static, dtype=initial_state.dtype, device=initial_state.device)
        physics = torch.as_tensor(prepared.actuator_physics, dtype=initial_state.dtype, device=initial_state.device)
        up = self.upstream.to(initial_state.device); down = self.downstream.to(initial_state.device)
        ids = torch.arange(actuators, device=initial_state.device)
        identity = self.actuator_identity(ids).to(initial_state.dtype)

        raw_state_records: list[torch.Tensor] = []
        raw_flow_records: list[torch.Tensor] = []
        latent_records: list[torch.Tensor] = []
        support_records: list[torch.Tensor] = []
        for pos, raw_index in enumerate(indices.detach().cpu().tolist()):
            idx = int(raw_index)
            prefix = candidate_settings[:, :, : idx + 1] - reference_settings[:, None, : idx + 1]
            current = prefix[:, :, -1]
            mean = prefix.mean(dim=2)
            abs_mean = prefix.abs().mean(dim=2)
            # Presence anchors exact-zero without attenuating every nonzero
            # action by its 72-step mean.  The previous smooth dose gate was
            # O(1e-3) for a single 0.15 setting move spread over H72, forcing
            # the facility head and all transported effects toward zero before
            # optimization could learn a realized-flow response.  A nonzero
            # action can still map to a near-zero flow through the learned head.
            actuator_presence = (abs_mean > 1e-8).to(initial_state.dtype)
            action_presence = (actuator_presence.sum(dim=-1) > 0).to(initial_state.dtype)
            rain_context = rainfall[:, : idx + 1].mean(dim=(1, 2))
            time_x = (torch.as_tensor(float(idx + 1), device=initial_state.device, dtype=initial_state.dtype) / float(horizon))
            time_features = torch.stack((time_x, time_x.square(), torch.sin(torch.pi * time_x), torch.cos(torch.pi * time_x)))
            ref_now = reference_settings[:, idx]
            up_state = _signed_log(initial_state[:, up])
            down_state = _signed_log(initial_state[:, down])
            prev = previous_actuator_flow
            own_prior = flow_prior[:, :, :, pos, :].diagonal(dim1=2, dim2=3)
            # flow_prior is [B,C,A,T,A]; diagonal over source and target gives
            # [B,C,T] and is used as a source-effect context (not a target).
            own_prior = flow_prior[:, :, :, pos, :].diagonal(dim1=2, dim2=3)
            if own_prior.ndim != 3:
                raise ValueError("V113 source-flow prior diagonal has unexpected shape")
            source_input = torch.cat(
                (
                    current[..., None],
                    mean[..., None],
                    abs_mean[..., None],
                    actuator_presence[..., None],
                    ref_now[:, None, :, None].expand(batch, candidates, actuators, 1),
                    prev[:, None, :, None].expand(batch, candidates, actuators, 1),
                    own_prior[..., None],
                    up_state[:, None].expand(batch, candidates, actuators, 6),
                    down_state[:, None].expand(batch, candidates, actuators, 6),
                    physics[None, None].expand(batch, candidates, actuators, -1),
                    identity[None, None].expand(batch, candidates, actuators, -1),
                    rain_context[:, None, None].expand(batch, candidates, actuators, self.rainfall_dim),
                    time_features[None, None, None].expand(batch, candidates, actuators, -1),
                ), dim=-1,
            )
            facility_hidden = self.facility_encoder(source_input)
            gate = actuator_presence[..., None]
            facility_latent = facility_hidden * gate
            flow_norm = torch.tanh(self.facility_head(facility_hidden).squeeze(-1)) * gate.squeeze(-1)
            flow_delta = flow_norm * self.flow_scale[None, None]

            prior_t = state_prior[:, :, :, pos]  # [B,C,A,N,5]
            support_context = torch.einsum("bca, bcanf->bcnf", flow_delta, prior_t)
            # Endpoint latent transport and a global escape path; no hop mask.
            endpoint = initial_state.new_zeros(batch, candidates, self.node_count, self.hidden_dim)
            flat = facility_latent.reshape(batch * candidates, actuators, self.hidden_dim)
            endpoint_flat = endpoint.reshape(batch * candidates, self.node_count, self.hidden_dim)
            endpoint_flat.index_add_(1, up, 0.5 * flat)
            endpoint_flat.index_add_(1, down, 0.5 * flat)
            endpoint = endpoint_flat.reshape(batch, candidates, self.node_count, self.hidden_dim)
            endpoint = endpoint[:, :, None].expand(batch, candidates, 1, self.node_count, self.hidden_dim)
            global_latent = facility_latent.mean(dim=2)[:, :, None, None, :].expand(batch, candidates, 1, self.node_count, self.hidden_dim)
            static_node = node_static[None, None].expand(batch, candidates, self.node_count, -1)
            node_base = initial_state[:, None].expand(batch, candidates, self.node_count, -1)
            rain_node = rainfall[:, : idx + 1].mean(dim=(1, 2))[:, None, None].expand(batch, candidates, self.node_count, self.rainfall_dim)
            support_node = support_context
            node_in = torch.cat((node_base, static_node, self.node_identity(torch.arange(self.node_count, device=initial_state.device))[None, None].expand(batch, candidates, self.node_count, -1), rain_node, endpoint.squeeze(2), support_node, global_latent.squeeze(2), time_features[None, None, None].expand(batch, candidates, self.node_count, -1)), dim=-1)
            node_hidden = self.node_transport(node_in)
            state_norm = torch.tanh(self.node_state_head(node_hidden))
            state_delta = state_norm * self.state_scale[None, None]
            # Storage is a physical domain, not a negative example at junctions.
            storage_mask = torch.as_tensor(prepared.storage_mask, dtype=torch.bool, device=state_delta.device)
            state_delta[..., 2] = torch.where(storage_mask[None, None], state_delta[..., 2], torch.zeros_like(state_delta[..., 2]))
            raw_state = torch.stack((state_delta[..., 0], state_delta[..., 0], state_delta[..., 1], state_delta[..., 2], state_delta[..., 3], state_delta[..., 4]), dim=-1)
            # Keep the candidate axis aligned: raw_state is [B,C,N,6], so the
            # action activity gate must be [B,C,1,1].  A five-dimensional
            # broadcast would silently introduce a second candidate axis.
            raw_state = raw_state * action_presence[:, :, None, None]
            flow_delta = flow_delta * action_presence[:, :, None]
            raw_state_records.append(raw_state)
            raw_flow_records.append(flow_delta)
            latent_records.append(facility_latent)
            support_records.append(support_context)

        raw_states = torch.stack(raw_state_records, dim=2)
        raw_flows = torch.stack(raw_flow_records, dim=2)
        latent = torch.stack(latent_records, dim=2)
        support = torch.stack(support_records, dim=2)
        response_minutes = (indices.to(initial_state.dtype) + 1.0) * 300.0 / 60.0
        return HydraulicEffectOutputV113(indices, response_minutes, raw_states, raw_flows, latent, support, self.prior_mode)


__all__ = ["HydraulicEffectOutputV113", "SupportConditionedHydraulicSurrogateV113", "STATE_EFFECT_NAMES_V113", "STATE_EFFECT_CHANNELS_V113"]
