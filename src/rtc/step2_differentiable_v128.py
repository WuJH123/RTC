"""Project7 V128 typed/physics-aware differentiable hydraulic surrogate.

V127 already routes predicted actuator flow into the hydraulic transition, but its direct
setting context reduces every actuator touching a node to two scalar sums (outgoing and
incoming setting).  That representation can alias physically different pump/orifice/weir
combinations and is especially damaging to the action gradients used by continuous MPC.

V128 keeps the frozen 300-s model / 600-s control / H360-H120 contract and the same
physical actuator-flow injection, but replaces that lossy direct setting context with a
learned actuator-to-node message.  Every message sees:

* upstream and downstream hydraulic state;
* requested setting;
* previous and predicted managed flow;
* learned flow responsiveness;
* frozen actuator physics/type features;
* learned actuator identity.

Messages are direction-aware and aggregated separately at upstream and downstream nodes.
No future SWMM state or realised rainfall is introduced; this is a representation change
only.  Authoritative truth remains SWMM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import HydraulicTransition, Rollout
from .step2_differentiable_v127 import (
    ControlOrientedDifferentiableSurrogateV127,
    V127SurrogateDesign,
)

V128_STEP2_CONTRACT = (
    "PROJECT7_V128_TYPED_PHYSICS_AWARE_ACTUATOR_MESSAGE_SURROGATE_V1"
)
V128_ACTION_CONTEXT_CONTRACT = (
    "ACTUATOR_TO_NODE_MESSAGE_STATE_SETTING_FLOW_PHYSICS_ID_DIRECTION_V1"
)


@dataclass(frozen=True)
class V128SurrogateDesign:
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    free_control_horizon_steps: int = 24
    hidden_dim: int = 160
    actuator_embedding_dim: int = 16
    action_message_dim: int = 24

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V128 requires the frozen 300-s model / 600-s control clock")
        if (self.prediction_horizon_steps, self.free_control_horizon_steps) != (72, 24):
            raise ValueError("V128 requires H360 prediction / H120 free control")
        if self.control_update_seconds % self.model_step_seconds:
            raise ValueError("V128 control clock must align with the model clock")
        if min(self.hidden_dim, self.actuator_embedding_dim, self.action_message_dim) <= 0:
            raise ValueError("V128 model dimensions must be positive")


class TypedActuatorMessageSurrogateV128(ControlOrientedDifferentiableSurrogateV127):
    """V127 hydraulic world model with non-aliasing actuator-to-node action messages."""

    contract = V128_STEP2_CONTRACT

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        node_static_dim: int,
        actuator_physics_dim: int,
        actuator_count: int,
        hidden_dim: int = 160,
        actuator_embedding_dim: int = 16,
        action_message_dim: int = 24,
        delta_state_scale: torch.Tensor | np.ndarray | None = None,
        delta_flow_scale: torch.Tensor | np.ndarray | None = None,
        smooth_flood_scale_m3s: float = 0.01,
        **runtime_metadata: Any,
    ) -> None:
        if action_message_dim <= 0:
            raise ValueError("V128 action_message_dim must be positive")
        super().__init__(
            state_dim=state_dim,
            rainfall_dim=rainfall_dim,
            node_static_dim=node_static_dim,
            actuator_physics_dim=actuator_physics_dim,
            actuator_count=actuator_count,
            hidden_dim=hidden_dim,
            actuator_embedding_dim=actuator_embedding_dim,
            delta_state_scale=delta_state_scale,
            delta_flow_scale=delta_flow_scale,
            smooth_flood_scale_m3s=smooth_flood_scale_m3s,
            **runtime_metadata,
        )
        self.action_message_dim = int(action_message_dim)
        identity_dim = int(self.actuator.actuator_embedding_dim)
        message_input_dim = (
            2 * int(state_dim)
            + int(actuator_physics_dim)
            + identity_dim
            + 4  # setting, previous flow, predicted flow, responsiveness
        )
        self.action_message_encoder = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_message_dim),
            nn.SiLU(),
        )
        self.outgoing_role = nn.Linear(self.action_message_dim, self.action_message_dim)
        self.incoming_role = nn.Linear(self.action_message_dim, self.action_message_dim)

        # Replace only the transition input layer/graph with the V128 action-context width.
        # Normalisation is configured after construction by the canonical training runner.
        self.transition = HydraulicTransition(
            state_dim,
            rainfall_dim,
            node_static_dim,
            hidden_dim=hidden_dim,
            action_context_dim=2 * self.action_message_dim,
            bounded_state_residual=True,
            delta_state_scale=(
                None
                if delta_state_scale is None
                else torch.as_tensor(delta_state_scale, dtype=torch.float32)
            ),
        )
        self.v128_contract = V128_STEP2_CONTRACT
        # The canonical streaming runner historically reads ``model.v127_contract`` when
        # emitting the report.  Keep the attribute as an interface alias, but point it to
        # the V128 scientific contract so no artifact can masquerade as V127.
        self.v127_contract = V128_STEP2_CONTRACT
        self.runtime_metadata.update(
            {
                "v128_step2_contract": V128_STEP2_CONTRACT,
                "action_context_contract": V128_ACTION_CONTEXT_CONTRACT,
                "action_message_dim": self.action_message_dim,
            }
        )

    def _typed_action_context(
        self,
        *,
        state: torch.Tensor,
        setting: torch.Tensor,
        previous_flow: torch.Tensor,
        predicted_flow: torch.Tensor,
        responsiveness: torch.Tensor,
        upstream: torch.Tensor,
        downstream: torch.Tensor,
        physics_norm: torch.Tensor,
        identity_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        up_state = (state[:, upstream] - self.actuator.state_mean) / self.actuator.state_std
        down_state = (state[:, downstream] - self.actuator.state_mean) / self.actuator.state_std
        parts = [
            up_state,
            down_state,
            setting[..., None],
            previous_flow[..., None] / self.actuator.flow_std,
            predicted_flow[..., None] / self.actuator.flow_std,
            responsiveness[..., None],
            physics_norm,
        ]
        if identity_embedding is not None:
            parts.append(identity_embedding)
        raw_message = self.action_message_encoder(torch.cat(parts, dim=-1))
        outgoing_message = self.outgoing_role(raw_message)
        incoming_message = self.incoming_role(raw_message)

        batch, node_count = int(state.shape[0]), int(state.shape[1])
        outgoing = torch.zeros(
            batch,
            node_count,
            self.action_message_dim,
            device=state.device,
            dtype=state.dtype,
        ).index_add(1, upstream, outgoing_message.to(dtype=state.dtype))
        incoming = torch.zeros_like(outgoing).index_add(
            1, downstream, incoming_message.to(dtype=state.dtype)
        )

        # Degree normalisation prevents a high-degree node from acquiring a large context
        # merely because many controllable links meet there, while preserving actuator
        # identity/type inside the learned messages.
        out_degree = torch.zeros(node_count, device=state.device, dtype=state.dtype)
        out_degree.index_add_(0, upstream, torch.ones_like(upstream, dtype=state.dtype))
        in_degree = torch.zeros(node_count, device=state.device, dtype=state.dtype)
        in_degree.index_add_(0, downstream, torch.ones_like(downstream, dtype=state.dtype))
        outgoing = outgoing / torch.sqrt(out_degree.clamp_min(1.0))[None, :, None]
        incoming = incoming / torch.sqrt(in_degree.clamp_min(1.0))[None, :, None]
        return torch.cat((outgoing, incoming), dim=-1)

    def rollout(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Rollout:
        if rainfall.shape[:2] != settings.shape[:2]:
            raise ValueError("V128 rainfall/settings batch and horizon dimensions must match")
        state, flow = initial_state, previous_actuator_flow
        states: list[torch.Tensor] = []
        flows: list[torch.Tensor] = []
        responses: list[torch.Tensor] = []
        up, down = actuator_upstream.long(), actuator_downstream.long()
        batch = int(state.shape[0])
        physics_norm, identity = self.actuator.prepare_static(
            actuator_physics, batch_size=batch
        )
        static_norm, edges, inv_degree = self.transition.prepare_static(
            static_node_features,
            edge_index,
            batch_size=batch,
            dtype=state.dtype,
        )
        for k in range(settings.shape[1]):
            previous_flow = flow
            q, response = self.actuator.forward_prepared(
                state[:, up],
                state[:, down],
                settings[:, k],
                previous_flow,
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
            action_context = self._typed_action_context(
                state=state,
                setting=settings[:, k],
                previous_flow=previous_flow,
                predicted_flow=q,
                responsiveness=response,
                upstream=up,
                downstream=down,
                physics_norm=physics_norm,
                identity_embedding=identity,
            )
            state = self.transition.forward_prepared(
                state,
                rainfall[:, k],
                static_norm,
                injection,
                edges,
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


def build_v128_model_from_graph(
    graph: Any,
    *,
    state_dim: int,
    rainfall_dim: int,
    delta_state_scale: torch.Tensor | np.ndarray,
    delta_flow_scale: torch.Tensor | np.ndarray,
    design: V128SurrogateDesign | V127SurrogateDesign = V128SurrogateDesign(),
) -> TypedActuatorMessageSurrogateV128:
    # The V128 wrapper may be called from the canonical V127 streaming implementation,
    # which supplies a V127SurrogateDesign.  Timing/dimensions must still match exactly.
    model_step_seconds = int(getattr(design, "model_step_seconds", -1))
    control_update_seconds = int(getattr(design, "control_update_seconds", -1))
    prediction_horizon_steps = int(getattr(design, "prediction_horizon_steps", -1))
    free_control_horizon_steps = int(getattr(design, "free_control_horizon_steps", -1))
    if (
        model_step_seconds,
        control_update_seconds,
        prediction_horizon_steps,
        free_control_horizon_steps,
    ) != (300, 600, 72, 24):
        raise ValueError("V128 builder received an incompatible time/horizon design")
    hidden_dim = int(getattr(design, "hidden_dim", 160))
    actuator_embedding_dim = int(getattr(design, "actuator_embedding_dim", 16))
    action_message_dim = int(getattr(design, "action_message_dim", 24))
    return TypedActuatorMessageSurrogateV128(
        state_dim=int(state_dim),
        rainfall_dim=int(rainfall_dim),
        node_static_dim=int(np.asarray(graph.static_node_features).shape[1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=hidden_dim,
        actuator_embedding_dim=actuator_embedding_dim,
        action_message_dim=action_message_dim,
        delta_state_scale=delta_state_scale,
        delta_flow_scale=delta_flow_scale,
        model_step_seconds=300,
        horizon_steps=72,
        control_update_seconds=600,
        free_control_horizon_steps=24,
        time_contract="PROJECT7_V128_300S_MODEL_600S_RECEDING_CONTROL_V1",
        v128_step2_contract=V128_STEP2_CONTRACT,
        action_context_contract=V128_ACTION_CONTEXT_CONTRACT,
    )


__all__ = [
    "TypedActuatorMessageSurrogateV128",
    "V128_ACTION_CONTEXT_CONTRACT",
    "V128_STEP2_CONTRACT",
    "V128SurrogateDesign",
    "build_v128_model_from_graph",
]
