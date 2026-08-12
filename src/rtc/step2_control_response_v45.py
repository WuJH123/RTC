"""V4.5 direct joint-action counterfactual response prototype.

The model keeps the validated V4.1 reference encoder and normalization buffers,
but does not use the D2-single-plus-D3-residual forward equation.  Candidate
and reference action sequences pass through one shared effect encoder and are
subtracted structurally.  A permutation-equivariant joint actuator context is
formed before endpoint scattering, allowing nonlinear multi-actuator effects
without a K=1/K>=2 hard switch or an unrestricted recurrent rollout.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .step2_control_response_v41 import (
    CounterfactualGroupResponseV41,
    DifferentiableCounterfactualResponseModelV41,
    _causal_accumulate,
    _directed_scatter,
    _mlp,
    _trapezoid_delta_tfv,
)


class DifferentiableCounterfactualResponseModelV45(
    DifferentiableCounterfactualResponseModelV41
):
    """Finite-capacity direct response operator for single and joint actions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden = self.hidden_dim
        token_dim = hidden + self.temporal_embedding_dim + 4
        self.effect_encoder_v45 = _mlp(token_dim, hidden, hidden)
        # h, sum(h), mean(h), max(h), and a low-rank second-order summary.
        self.joint_context_encoder_v45 = _mlp(5 * hidden, hidden, hidden)
        self.node_response_encoder_v45 = _mlp(3 * hidden, hidden, hidden)
        self.node_state_head_v45 = nn.Linear(hidden, self.state_dim)
        # TFV is a system-level control value.  Use a shared joint-action
        # summary for its primary head; the node field remains responsible for
        # spatial state/flood trajectories.  This avoids imposing a 1/N
        # gradient penalty on a global target while retaining the same
        # candidate/reference subtraction contract.
        self.global_tfv_head_v45 = _mlp(3 * hidden, hidden, 1)
        self.flow_head_v45 = nn.Linear(2 * hidden, 1)

    def _source_scales(self, source_kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = source_kind.upper()
        if source == "D2":
            return self.d2_state_scale, self.d2_flow_scale, self.d2_tfv_scale
        if source == "D3":
            return self.d3_state_scale, self.d3_flow_scale, self.d3_tfv_scale
        raise ValueError("source_kind must be D2 or D3")

    def _effect_joint_hidden(
        self,
        *,
        settings: torch.Tensor,
        reference_settings: torch.Tensor,
        reference_flow: torch.Tensor,
        actuator_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one action sequence and its joint context causally."""

        batch, candidates, horizon, actuators = settings.shape
        dtype = settings.dtype
        device = settings.device
        delta = settings - reference_settings[:, None]
        static = actuator_static.to(device=device, dtype=dtype)[None, None, None].expand(
            batch, candidates, horizon, actuators, -1
        )
        temporal_ids = torch.arange(horizon, device=device)
        temporal = self.temporal_identity(temporal_ids)[None, None, :, None].expand(
            batch, candidates, -1, actuators, -1
        )
        reference = reference_settings[:, None, :, :, None].expand(
            -1, candidates, -1, -1, -1
        )
        flow_scale = self.flow_std.to(device=device, dtype=dtype).clamp_min(1e-6)
        flow = reference_flow[:, None].expand(batch, candidates, -1, -1) / flow_scale
        token_input = torch.cat(
            (
                static,
                temporal,
                reference,
                settings[..., None],
                delta[..., None],
                flow[..., None],
            ),
            dim=-1,
        )
        token_hidden = self.effect_encoder_v45(token_input)
        sum_hidden = token_hidden.sum(dim=3, keepdim=True).expand_as(token_hidden)
        mean_hidden = token_hidden.mean(dim=3, keepdim=True).expand_as(token_hidden)
        max_hidden = token_hidden.amax(dim=3, keepdim=True).expand_as(token_hidden)
        second_order = token_hidden.square().mean(dim=3, keepdim=True).expand_as(token_hidden)
        joint_input = torch.cat(
            (token_hidden, sum_hidden, mean_hidden, max_hidden, second_order), dim=-1
        )
        joint_hidden = self.joint_context_encoder_v45(joint_input)
        # Fixed 0.65 causal memory, inherited from the bounded response
        # contract; no learned/unrestricted recurrent state is introduced.
        joint_hidden = _causal_accumulate(joint_hidden)
        return joint_hidden, delta

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp_min(torch.finfo(value.dtype).tiny)
        return value + torch.log(-torch.expm1(-value))

    def _effect_response(
        self,
        *,
        settings: torch.Tensor,
        reference_settings: torch.Tensor,
        reference: object,
        prepared,
        elapsed_seconds: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        source_kind: str,
    ) -> dict[str, torch.Tensor]:
        """Return one absolute response evaluation before candidate-reference subtraction."""

        del previous_actuator_flow
        batch, candidates, horizon, actuators = settings.shape
        nodes = reference.node_context.shape[2]
        dtype = settings.dtype
        device = settings.device
        joint_hidden, delta = self._effect_joint_hidden(
            settings=settings,
            reference_settings=reference_settings,
            reference_flow=reference.reference_flows_physical,
            actuator_static=reference.actuator_static,
        )
        upstream = reference.actuator_upstream.to(device=device)
        downstream = reference.actuator_downstream.to(device=device)
        node_seed_out, node_seed_in = _directed_scatter(
            joint_hidden, upstream, downstream, nodes
        )
        node_seed = node_seed_out + node_seed_in
        node_context = reference.node_context[:, None].expand(
            batch, candidates, horizon, nodes, self.hidden_dim
        ).to(dtype=dtype)
        static_context = reference.node_static_context.to(device=device, dtype=dtype)[None, None, None]
        static_context = static_context.expand(batch, candidates, horizon, nodes, self.hidden_dim)
        node_hidden = self.node_response_encoder_v45(
            torch.cat((node_seed, node_context, static_context), dim=-1)
        )
        state_raw = self.node_state_head_v45(node_hidden)
        # Keep sparse actions visible: mean pooling alone divides a one-device
        # perturbation by all 109 actuators.  The normalized sum retains the
        # joint signal while remaining bounded as the actuator set grows.
        actuator_norm = float(max(self.actuator_count, 1)) ** 0.5
        global_joint = torch.cat(
            (
                joint_hidden.sum(dim=3) / actuator_norm,
                joint_hidden.mean(dim=3),
                joint_hidden.amax(dim=3),
            ),
            dim=-1,
        )
        tfv_rate_raw = self.global_tfv_head_v45(global_joint).squeeze(-1)

        flow_context = torch.cat((joint_hidden, joint_hidden.mean(dim=3, keepdim=True).expand_as(joint_hidden)), dim=-1)
        flow_raw = self.flow_head_v45(flow_context).squeeze(-1)
        state_scale, flow_scale, tfv_scale = self._source_scales(source_kind)
        state_scale = state_scale.to(device=device, dtype=dtype)
        flow_scale = flow_scale.to(device=device, dtype=dtype)
        tfv_scale = tfv_scale.to(device=device, dtype=dtype)
        # Keep the direct physical heads in the same bounded normalized
        # contract as V4.1.  Without this, an unconstrained linear state head
        # can turn a small action token into a multi-horizon depth/flood
        # explosion before the optimizer has any useful signal.
        state_latent = torch.tanh(state_raw) * state_scale.reshape(1, 1, 1, 1, -1)
        reference_depth = reference.reference_states_physical[:, None, ..., 0].expand(
            batch, candidates, horizon, nodes
        )
        reference_depth_latent = self._inverse_softplus(reference_depth)
        reference_flood_latent = reference.reference_flood_latent[:, None].expand(
            batch, candidates, horizon, nodes
        )
        depth = F.softplus(reference_depth_latent + state_latent[..., 0])
        flood = F.softplus(reference_flood_latent + state_latent[..., 2])
        physical_other = reference.reference_states_physical[:, None, ..., 3:] + state_latent[..., 3:]
        absolute_states = torch.cat(
            (
                depth.unsqueeze(-1),
                (depth + reference.invert_elevation_m.reshape(1, 1, 1, nodes)).unsqueeze(-1),
                flood.unsqueeze(-1),
                physical_other,
            ),
            dim=-1,
        )
        absolute_flows = (
            _causal_accumulate(flow_raw.unsqueeze(-1)).squeeze(-1)
            * flow_scale.reshape(1, 1, 1, actuators)
        )
        # ``tfv_rate_raw`` is a dimensionless system response score, not an
        # m3/s quantity.  Integrate over normalized horizon time before
        # applying the one physical TFV scale.  This avoids accidentally
        # multiplying the scale once for every H×N cell while preserving a
        # causal multi-horizon operator.
        total_seconds = (
            elapsed_seconds[..., -1:] - elapsed_seconds[..., :1]
        ).clamp_min(1.0)
        normalized_elapsed = (
            elapsed_seconds - elapsed_seconds[..., :1]
        ) / total_seconds
        # The frozen Train-only TFV scale has observed |target/scale| < 8;
        # retain that fixed physical envelope while keeping unit slope around
        # zero (plain tanh would saturate at ±1 and collapse candidate spread).
        normalized_rate = (8.0 * torch.tanh(tfv_rate_raw / 8.0)).unsqueeze(-1)
        absolute_tfv = _trapezoid_delta_tfv(normalized_rate, normalized_elapsed)
        absolute_tfv = absolute_tfv * tfv_scale
        return {
            "states": absolute_states,
            "flows": absolute_flows,
            "tfv": absolute_tfv,
            "tfv_rate": tfv_rate_raw,
            "delta": delta,
        }

    def forward_group(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared,
        elapsed_seconds: torch.Tensor,
        *,
        source_kind: str,
    ) -> CounterfactualGroupResponseV41:
        if candidate_settings.dim() != 4:
            raise ValueError("candidate_settings must be [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count:
            raise ValueError("candidate actuator count does not match model")
        reference = self.encode_reference(
            initial_state, rainfall, reference_settings, previous_actuator_flow, prepared
        )
        reference_settings_expanded = reference_settings[:, None].expand(
            batch, candidates, horizon, actuators
        )
        reference_eval = self._effect_response(
            settings=reference_settings_expanded,
            reference_settings=reference_settings,
            reference=reference,
            prepared=prepared,
            elapsed_seconds=elapsed_seconds,
            previous_actuator_flow=previous_actuator_flow,
            source_kind=source_kind,
        )
        candidate_eval = self._effect_response(
            settings=candidate_settings,
            reference_settings=reference_settings,
            reference=reference,
            prepared=prepared,
            elapsed_seconds=elapsed_seconds,
            previous_actuator_flow=previous_actuator_flow,
            source_kind=source_kind,
        )
        delta_states = candidate_eval["states"] - reference_eval["states"]
        delta_flows = candidate_eval["flows"] - reference_eval["flows"]
        direct_tfv = candidate_eval["tfv"] - reference_eval["tfv"]
        # The causal operator already excludes future actions from each prefix.
        # Ensure exact no-op rows use the frozen reference tensor bit-for-bit.
        delta_u = candidate_settings - reference_settings[:, None]
        causal_action_mask = delta_u.detach().abs().sum(dim=-1).cumsum(dim=2).ne(0.0)
        delta_states = torch.where(
            causal_action_mask[..., None, None], delta_states, torch.zeros_like(delta_states)
        )
        delta_flows = torch.where(
            causal_action_mask.unsqueeze(-1), delta_flows, torch.zeros_like(delta_flows)
        )
        candidate_states = torch.where(
            causal_action_mask[..., None, None],
            reference.reference_states_physical[:, None] + delta_states,
            reference.reference_states_physical[:, None].expand(-1, candidates, -1, -1, -1),
        )
        candidate_flows = reference.reference_flows_physical[:, None] + delta_flows
        trajectory_tfv = _trapezoid_delta_tfv(delta_states[..., 2], elapsed_seconds)
        zero_state = torch.zeros_like(delta_states)
        zero_flow = torch.zeros_like(delta_flows)
        zero_tfv = torch.zeros_like(direct_tfv)
        return CounterfactualGroupResponseV41(
            reference_states_physical=reference.reference_states_physical,
            candidate_states_physical=candidate_states,
            single_delta_states_physical=delta_states,
            interaction_delta_states_physical=zero_state,
            delta_states_physical=delta_states,
            reference_flows_physical=reference.reference_flows_physical,
            candidate_flows_physical=candidate_flows,
            single_delta_flows_physical=delta_flows,
            interaction_delta_flows_physical=zero_flow,
            delta_flows_physical=delta_flows,
            direct_single_delta_tfv_m3=direct_tfv,
            direct_interaction_delta_tfv_m3=zero_tfv,
            nodewise_local_residual_delta_tfv_m3=zero_tfv,
            direct_delta_tfv_m3=direct_tfv,
            trajectory_delta_tfv_m3=trajectory_tfv,
        )


def direct_effect_parameter_names_v45(model: nn.Module) -> tuple[str, ...]:
    prefixes = (
        "effect_encoder_v45.",
        "joint_context_encoder_v45.",
        "node_response_encoder_v45.",
        "node_state_head_v45.",
        "global_tfv_head_v45.",
        "flow_head_v45.",
    )
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def set_trainable_direct_effect_v45(model: nn.Module, *, enabled: bool = True) -> tuple[str, ...]:
    names = set(direct_effect_parameter_names_v45(model))
    if enabled and not names:
        raise RuntimeError("V4.5 direct effect parameter set is empty")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(enabled and name in names))
        if not (enabled and name in names):
            parameter.grad = None
    return tuple(sorted(names if enabled else ()))


__all__ = [
    "DifferentiableCounterfactualResponseModelV45",
    "direct_effect_parameter_names_v45",
    "set_trainable_direct_effect_v45",
]
