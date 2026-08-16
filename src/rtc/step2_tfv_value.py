"""Direct Project7 Step2 model: causal state + 109-facility actions -> delta TFV.

This module deliberately removes the historical requirement that Step2 must first predict every
future hydraulic state accurately before it can learn control value.  The research target is
system-wide cumulative TFV.  Existing authoritative SWMM counterfactual data already provide
same-prefix action -> exact delta-TFV labels, so the primary Step2 path learns those labels
explicitly.

The model has two interpretable parts:

* per-facility main effects, trained primarily from exact single-actuator counterfactuals; and
* one joint interaction residual, trained from multi-actuator counterfactuals.

Zero action has exactly zero predicted delta TFV.  The interaction residual is exactly zero when
fewer than two facilities change.  Full hydraulic trajectory prediction can remain an auxiliary
diagnostic, but it is no longer a prerequisite for the TFV-control learning problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


DIRECT_TFV_VALUE_CONTRACT = "PROJECT7_DIRECT_109ACT_ACTION_TO_DELTA_TFV_VALUE_V1"


@dataclass(frozen=True)
class DirectTFVValueDesign:
    actuator_count: int = 109
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    free_control_blocks: int = 12
    hidden_dim: int = 96
    actuator_embedding_dim: int = 16
    rainfall_bins: int = 12

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    @property
    def action_blocks(self) -> int:
        return self.prediction_horizon_steps // self.control_block_steps

    def validate(self) -> None:
        if self.actuator_count != 109:
            raise ValueError("direct TFV value model requires the frozen 109 actuators")
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("direct TFV value model requires 300-s state / 600-s control")
        if self.prediction_horizon_steps != 72 or self.free_control_blocks != 12:
            raise ValueError("direct TFV value model requires H360 prediction / H120 free control")
        if self.prediction_horizon_steps % self.control_block_steps:
            raise ValueError("prediction horizon must contain complete control blocks")
        if self.prediction_horizon_steps % self.rainfall_bins:
            raise ValueError("rainfall bins must divide the prediction horizon")
        if min(self.hidden_dim, self.actuator_embedding_dim, self.rainfall_bins) <= 0:
            raise ValueError("direct TFV value model dimensions must be positive")


@dataclass(frozen=True)
class DirectTFVValueOutput:
    total_delta_tfv_m3: torch.Tensor
    facility_main_effect_m3: torch.Tensor
    interaction_residual_m3: torch.Tensor
    action_activity: torch.Tensor


class DirectFacilityTFVValueModel(nn.Module):
    """Learn state-dependent TFV value of a complete 109-facility action sequence.

    Inputs use the same online-equivalent surfaces already available in Project7:
    reconstructed current state, causal rainfall forecast, current/reference settings and
    current managed flows.  ``candidate_settings`` may contain any differentiable H360 action
    sequence that respects the Step3 decoder.
    """

    contract = DIRECT_TFV_VALUE_CONTRACT

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        actuator_physics_dim: int,
        target_scale_m3: float,
        design: DirectTFVValueDesign = DirectTFVValueDesign(),
    ) -> None:
        super().__init__()
        design.validate()
        if min(state_dim, rainfall_dim, actuator_physics_dim) <= 0:
            raise ValueError("direct TFV value input dimensions must be positive")
        if not math.isfinite(float(target_scale_m3)) or float(target_scale_m3) <= 0.0:
            raise ValueError("target_scale_m3 must be finite and positive")
        self.design = design
        self.state_dim = int(state_dim)
        self.rainfall_dim = int(rainfall_dim)
        self.actuator_physics_dim = int(actuator_physics_dim)
        self.register_buffer(
            "target_scale_m3",
            torch.as_tensor(float(target_scale_m3), dtype=torch.float32),
        )

        h = int(design.hidden_dim)
        self.actuator_embedding = nn.Embedding(design.actuator_count, design.actuator_embedding_dim)
        self.global_state_encoder = nn.Sequential(
            nn.Linear(2 * self.state_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.rainfall_encoder = nn.Sequential(
            nn.Linear(design.rainfall_bins * self.rainfall_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        facility_input = (
            2 * self.state_dim
            + self.actuator_physics_dim
            + design.actuator_embedding_dim
            + 2  # normalized previous flow + current/reference setting
            + design.action_blocks
            + 2 * h  # global hydraulic and rainfall contexts
        )
        self.facility_encoder = nn.Sequential(
            nn.Linear(facility_input, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.facility_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.SiLU(),
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )

    def _validate_inputs(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
    ) -> None:
        b = int(current_state.shape[0])
        h = self.design.prediction_horizon_steps
        a = self.design.actuator_count
        if current_state.ndim != 3 or current_state.shape[-1] != self.state_dim:
            raise ValueError("current_state must be [B,node,state_feature]")
        if rainfall.ndim != 4 or rainfall.shape[0] != b or rainfall.shape[1] != h:
            raise ValueError("rainfall must be [B,H72,node,rain_feature]")
        if rainfall.shape[2] != current_state.shape[1] or rainfall.shape[-1] != self.rainfall_dim:
            raise ValueError("rainfall/current-state node or feature dimensions do not align")
        expected = (b, h, a)
        if tuple(reference_settings.shape) != expected or tuple(candidate_settings.shape) != expected:
            raise ValueError("reference/candidate settings must be [B,H72,109]")
        if tuple(previous_actuator_flow.shape) != (b, a):
            raise ValueError("previous_actuator_flow must be [B,109]")
        if tuple(actuator_upstream.shape) != (a,) or tuple(actuator_downstream.shape) != (a,):
            raise ValueError("actuator endpoint indices must contain 109 entries")
        if tuple(actuator_physics.shape) != (a, self.actuator_physics_dim):
            raise ValueError("actuator_physics shape does not match model")
        if bool(torch.any((reference_settings < 0.0) | (reference_settings > 1.0))):
            raise ValueError("reference settings leave physical [0,1] bounds")
        if bool(torch.any((candidate_settings < 0.0) | (candidate_settings > 1.0))):
            raise ValueError("candidate settings leave physical [0,1] bounds")
        for value in (
            current_state,
            rainfall,
            reference_settings,
            candidate_settings,
            previous_actuator_flow,
            actuator_physics,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("direct TFV value inputs must be finite")

    def _control_blocks(self, settings: torch.Tensor) -> torch.Tensor:
        b, _, a = settings.shape
        step = self.design.control_block_steps
        return settings.reshape(b, self.design.action_blocks, step, a).mean(dim=2)

    def _rainfall_summary(self, rainfall: torch.Tensor) -> torch.Tensor:
        b, h, n, r = rainfall.shape
        width = h // self.design.rainfall_bins
        # Spatial mean preserves the causal future forcing pattern while keeping the primary
        # model small. Local hydraulic state supplies facility-specific spatial conditioning.
        spatial = rainfall.mean(dim=2)
        return spatial.reshape(b, self.design.rainfall_bins, width, r).mean(dim=2).reshape(b, -1)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
    ) -> DirectTFVValueOutput:
        self._validate_inputs(
            current_state=current_state,
            rainfall=rainfall,
            reference_settings=reference_settings,
            candidate_settings=candidate_settings,
            previous_actuator_flow=previous_actuator_flow,
            actuator_upstream=actuator_upstream,
            actuator_downstream=actuator_downstream,
            actuator_physics=actuator_physics,
        )
        b = int(current_state.shape[0])
        a = self.design.actuator_count
        up = actuator_upstream.long()
        down = actuator_downstream.long()

        global_state = torch.cat((current_state.mean(dim=1), current_state.amax(dim=1)), dim=-1)
        state_context = self.global_state_encoder(global_state)
        rainfall_context = self.rainfall_encoder(self._rainfall_summary(rainfall))

        local_up = current_state[:, up]
        local_down = current_state[:, down]
        physics = actuator_physics.to(dtype=current_state.dtype)
        physics_mean = physics.mean(dim=0, keepdim=True)
        physics_std = physics.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        physics_norm = ((physics - physics_mean) / physics_std)[None].expand(b, -1, -1)
        ids = torch.arange(a, device=current_state.device)
        identity = self.actuator_embedding(ids)[None].expand(b, -1, -1).to(current_state.dtype)

        reference_blocks = self._control_blocks(reference_settings)
        candidate_blocks = self._control_blocks(candidate_settings)
        action = (candidate_blocks - reference_blocks).transpose(1, 2)
        current_setting = reference_blocks[:, 0]
        common = torch.cat(
            (
                local_up,
                local_down,
                physics_norm,
                identity,
                previous_actuator_flow[..., None],
                current_setting[..., None],
                state_context[:, None].expand(-1, a, -1),
                rainfall_context[:, None].expand(-1, a, -1),
            ),
            dim=-1,
        )
        zero_action = torch.zeros_like(action)
        latent = self.facility_encoder(torch.cat((common, action), dim=-1))
        latent_zero = self.facility_encoder(torch.cat((common, zero_action), dim=-1))

        # Subtracting the exact same network evaluated at zero action gives an exact-zero
        # contract without thresholds or a non-differentiable active-facility mask.
        main_normalized = (
            self.facility_head(latent).squeeze(-1)
            - self.facility_head(latent_zero).squeeze(-1)
        )
        facility_effect = main_normalized * self.target_scale_m3

        activity = torch.mean(torch.abs(action), dim=-1)
        total_activity = activity.sum(dim=-1)
        pair_mass = 0.5 * (
            torch.square(total_activity) - torch.sum(torch.square(activity), dim=-1)
        )
        pair_mass = pair_mass.clamp_min(0.0)
        pair_gate = pair_mass / (pair_mass + 0.05)
        pooled_delta = (latent - latent_zero).sum(dim=1) / math.sqrt(float(a))
        interaction_features = torch.cat((pooled_delta, state_context, rainfall_context), dim=-1)
        interaction = (
            self.interaction_head(interaction_features).squeeze(-1)
            * self.target_scale_m3
            * pair_gate
        )
        total = facility_effect.sum(dim=-1) + interaction
        return DirectTFVValueOutput(
            total_delta_tfv_m3=total,
            facility_main_effect_m3=facility_effect,
            interaction_residual_m3=interaction,
            action_activity=activity,
        )


__all__ = [
    "DIRECT_TFV_VALUE_CONTRACT",
    "DirectFacilityTFVValueModel",
    "DirectTFVValueDesign",
    "DirectTFVValueOutput",
]
