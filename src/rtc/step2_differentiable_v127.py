"""Project7 V127 control-oriented differentiable hydraulic surrogate.

The model maps causal current hydraulics, causal rainfall and a continuous future target
sequence to future hydraulic states.  It exposes both a hard physical flood-volume
operator for prediction/reporting and a smooth positive proxy for action gradients.  The
smooth proxy is never called authoritative SWMM truth; it exists solely to avoid the
zero-gradient dead zone of a hard clamp at predicted flood rate zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .flood_volume import smooth_trapezoid_node_flood_volume, trapezoid_node_flood_volume
from .models import DifferentiableHydraulicWorldModel, Rollout

V127_STEP2_CONTRACT = "PROJECT7_V127_CONTROL_ORIENTED_DIFFERENTIABLE_HYDRAULIC_SURROGATE_V2_SMOOTH_MPC_OBJECTIVE"
V127_SMOOTH_FLOOD_SCALE_M3S = 0.01


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
    smooth_flood_scale_m3s: float = V127_SMOOTH_FLOOD_SCALE_M3S

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
        if not np.isfinite(self.smooth_flood_scale_m3s) or self.smooth_flood_scale_m3s <= 0:
            raise ValueError("V127 smooth flood scale must be finite and positive")


@dataclass(frozen=True)
class V127ObjectiveOutput:
    rollout: Rollout
    node_flood_volume_m3: torch.Tensor
    tfv_m3: torch.Tensor
    pfv_m3: torch.Tensor
    optimization_node_flood_volume_m3: torch.Tensor
    optimization_tfv_m3: torch.Tensor
    optimization_pfv_m3: torch.Tensor


class ControlOrientedDifferentiableSurrogateV127(DifferentiableHydraulicWorldModel):
    """Graph hydraulic world model with hard-reporting and smooth-optimization TFV."""

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
        smooth_flood_scale_m3s: float = V127_SMOOTH_FLOOD_SCALE_M3S,
        **runtime_metadata: Any,
    ) -> None:
        if not np.isfinite(float(smooth_flood_scale_m3s)) or float(smooth_flood_scale_m3s) <= 0:
            raise ValueError("V127 smooth flood scale must be finite and positive")
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
                None if delta_state_scale is None
                else torch.as_tensor(delta_state_scale, dtype=torch.float32)
            ),
            delta_flow_scale=(
                None if delta_flow_scale is None
                else torch.as_tensor(delta_flow_scale, dtype=torch.float32)
            ),
            **runtime_metadata,
        )
        self.v127_contract = V127_STEP2_CONTRACT
        self.register_buffer(
            "v127_smooth_flood_scale_m3s",
            torch.as_tensor(float(smooth_flood_scale_m3s), dtype=torch.float32),
        )

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
        hard_volume = trapezoid_node_flood_volume(
            initial_state,
            rollout.states,
            flood_rate_index=int(flood_rate_index),
            dt_seconds=float(dt_seconds),
        )
        smooth_volume = smooth_trapezoid_node_flood_volume(
            initial_state,
            rollout.states,
            flood_rate_index=int(flood_rate_index),
            dt_seconds=float(dt_seconds),
            softplus_scale_m3s=self.v127_smooth_flood_scale_m3s,
        )
        if not bool(torch.isfinite(hard_volume).all()) or not bool(torch.isfinite(smooth_volume).all()):
            raise RuntimeError("V127 surrogate produced non-finite flood volume")
        hard_tfv = hard_volume.sum(dim=-1)
        smooth_tfv = smooth_volume.sum(dim=-1)
        if priority_indices is None or int(priority_indices.numel()) == 0:
            hard_pfv = torch.zeros_like(hard_tfv)
            smooth_pfv = torch.zeros_like(smooth_tfv)
        else:
            p = priority_indices.to(device=hard_volume.device, dtype=torch.long)
            hard_pfv = hard_volume.index_select(-1, p).sum(dim=-1)
            smooth_pfv = smooth_volume.index_select(-1, p).sum(dim=-1)
        return V127ObjectiveOutput(
            rollout=rollout,
            node_flood_volume_m3=hard_volume,
            tfv_m3=hard_tfv,
            pfv_m3=hard_pfv,
            optimization_node_flood_volume_m3=smooth_volume,
            optimization_tfv_m3=smooth_tfv,
            optimization_pfv_m3=smooth_pfv,
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
    return ControlOrientedDifferentiableSurrogateV127(
        state_dim=int(state_dim),
        rainfall_dim=int(rainfall_dim),
        node_static_dim=int(np.asarray(graph.static_node_features).shape[1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=design.hidden_dim,
        actuator_embedding_dim=design.actuator_embedding_dim,
        delta_state_scale=delta_state_scale,
        delta_flow_scale=delta_flow_scale,
        smooth_flood_scale_m3s=design.smooth_flood_scale_m3s,
        model_step_seconds=design.model_step_seconds,
        horizon_steps=design.prediction_horizon_steps,
        control_update_seconds=design.control_update_seconds,
        free_control_horizon_steps=design.free_control_horizon_steps,
        time_contract="PROJECT7_V127_300S_MODEL_600S_RECEDING_CONTROL_V1",
        v127_step2_contract=V127_STEP2_CONTRACT,
    )


__all__ = [
    "ControlOrientedDifferentiableSurrogateV127",
    "V127ObjectiveOutput",
    "V127SurrogateDesign",
    "V127_SMOOTH_FLOOD_SCALE_M3S",
    "V127_STEP2_CONTRACT",
    "build_v127_model_from_graph",
]
