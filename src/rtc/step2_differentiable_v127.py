"""Project7 V127 control-oriented differentiable hydraulic surrogate.

This module restores the original Project7 scientific target: learn a differentiable
mapping from a causal current hydraulic state and causal rainfall forecast plus a
continuous future actuator target sequence to future hydraulics and flood volumes.  It
does not use Sparse-RBC as a Value reference and it does not restrict the learned action
space to an anchor neighbourhood.

The implementation deliberately reuses the graph-based differentiable world-model
primitives that remain in :mod:`rtc.models`, but freezes a new scientific contract and
an MPC-facing physical objective interface.  Sparse-RBC belongs to Step3 as a warm start
and fail-safe, not inside this model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .flood_volume import trapezoid_node_flood_volume
from .models import DifferentiableHydraulicWorldModel, Rollout

V127_STEP2_CONTRACT = "PROJECT7_V127_CONTROL_ORIENTED_DIFFERENTIABLE_HYDRAULIC_SURROGATE_V1"


@dataclass(frozen=True)
class V127SurrogateDesign:
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    free_control_horizon_steps: int = 24
    hidden_dim: int = 160
    actuator_embedding_dim: int = 16
    direct_action_context: bool = True
    bounded_state_residual: bool = True
    bounded_flow_residual: bool = True

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    @property
    def free_control_blocks(self) -> int:
        return self.free_control_horizon_steps // self.control_block_steps

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V127 requires the frozen 300-s observation / 600-s control clock")
        if self.prediction_horizon_steps != 72:
            raise ValueError("V127 retains the H72 / 360-min prediction horizon")
        if self.free_control_horizon_steps != 24:
            raise ValueError("V127 retains a 120-min free control horizon")
        if self.free_control_horizon_steps > self.prediction_horizon_steps:
            raise ValueError("V127 free control horizon exceeds prediction horizon")
        if self.control_update_seconds % self.model_step_seconds:
            raise ValueError("V127 control update must align to model step")
        if self.free_control_horizon_steps % self.control_block_steps:
            raise ValueError("V127 free control horizon must contain complete control blocks")
        if self.hidden_dim <= 0 or self.actuator_embedding_dim <= 0:
            raise ValueError("V127 model dimensions must be positive")
        if not self.direct_action_context:
            raise ValueError("V127 requires direct action context in the hydraulic transition")
        if not self.bounded_state_residual or not self.bounded_flow_residual:
            raise ValueError("V127 requires bounded hydraulic and actuator residual dynamics")


@dataclass(frozen=True)
class V127ObjectiveOutput:
    rollout: Rollout
    node_flood_volume_m3: torch.Tensor
    tfv_m3: torch.Tensor
    pfv_m3: torch.Tensor


class ControlOrientedDifferentiableSurrogateV127(DifferentiableHydraulicWorldModel):
    """Graph hydraulic world model with an explicit differentiable TFV/PFV interface."""

    contract = V127_STEP2_CONTRACT

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
        delta_state_scale: torch.Tensor | np.ndarray | None = None,
        delta_flow_scale: torch.Tensor | np.ndarray | None = None,
        **runtime_metadata: Any,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            rainfall_dim=rainfall_dim,
            node_static_dim=node_static_dim,
            actuator_physics_dim=actuator_physics_dim,
            hidden_dim=hidden_dim,
            actuator_count=actuator_count,
            actuator_embedding_dim=actuator_embedding_dim,
            direct_action_context=True,
            bounded_state_residual=True,
            bounded_flow_residual=True,
            delta_state_scale=(
                None
                if delta_state_scale is None
                else torch.as_tensor(delta_state_scale, dtype=torch.float32)
            ),
            delta_flow_scale=(
                None
                if delta_flow_scale is None
                else torch.as_tensor(delta_flow_scale, dtype=torch.float32)
            ),
            **runtime_metadata,
        )
        self.v127_contract = V127_STEP2_CONTRACT

    def objective_rollout(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        flood_rate_index: int,
        priority_indices: torch.Tensor | None,
        dt_seconds: float = 300.0,
    ) -> V127ObjectiveOutput:
        if settings.ndim != 3 or rainfall.ndim != 4:
            raise ValueError("V127 objective expects settings [B,H,A] and rainfall [B,H,N,R]")
        if settings.shape[:2] != rainfall.shape[:2]:
            raise ValueError("V127 action/rainfall batch-horizon mismatch")
        if initial_state.ndim != 3 or initial_state.shape[0] != settings.shape[0]:
            raise ValueError("V127 initial state batch mismatch")
        if not bool(torch.isfinite(settings).all()) or bool(torch.any((settings < 0.0) | (settings > 1.0))):
            raise ValueError("V127 settings must be finite physical targets in [0,1]")
        rollout = self.rollout(
            initial_state,
            rainfall,
            settings,
            previous_actuator_flow,
            actuator_upstream,
            actuator_downstream,
            actuator_physics,
            static_node_features,
            edge_index,
        )
        node_volume = trapezoid_node_flood_volume(
            initial_state,
            rollout.states,
            flood_rate_index=int(flood_rate_index),
            dt_seconds=float(dt_seconds),
        )
        if not bool(torch.isfinite(node_volume).all()):
            raise RuntimeError("V127 surrogate produced non-finite flood volume")
        tfv = node_volume.sum(dim=-1)
        if priority_indices is None or int(priority_indices.numel()) == 0:
            pfv = torch.zeros_like(tfv)
        else:
            p = priority_indices.to(device=node_volume.device, dtype=torch.long)
            pfv = node_volume.index_select(-1, p).sum(dim=-1)
        return V127ObjectiveOutput(
            rollout=rollout,
            node_flood_volume_m3=node_volume,
            tfv_m3=tfv,
            pfv_m3=pfv,
        )


def build_v127_model_from_graph(
    graph: Any,
    *,
    state_dim: int,
    rainfall_dim: int,
    delta_state_scale: torch.Tensor | np.ndarray,
    delta_flow_scale: torch.Tensor | np.ndarray,
    design: V127SurrogateDesign = V127SurrogateDesign(),
) -> ControlOrientedDifferentiableSurrogateV127:
    design.validate()
    model = ControlOrientedDifferentiableSurrogateV127(
        state_dim=int(state_dim),
        rainfall_dim=int(rainfall_dim),
        node_static_dim=int(np.asarray(graph.static_node_features).shape[1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=design.hidden_dim,
        actuator_embedding_dim=design.actuator_embedding_dim,
        delta_state_scale=delta_state_scale,
        delta_flow_scale=delta_flow_scale,
        model_step_seconds=design.model_step_seconds,
        horizon_steps=design.prediction_horizon_steps,
        control_update_seconds=design.control_update_seconds,
        free_control_horizon_steps=design.free_control_horizon_steps,
        time_contract="PROJECT7_V127_300S_MODEL_600S_RECEDING_CONTROL_V1",
        v127_step2_contract=V127_STEP2_CONTRACT,
    )
    return model


__all__ = [
    "ControlOrientedDifferentiableSurrogateV127",
    "V127ObjectiveOutput",
    "V127SurrogateDesign",
    "V127_STEP2_CONTRACT",
    "build_v127_model_from_graph",
]
