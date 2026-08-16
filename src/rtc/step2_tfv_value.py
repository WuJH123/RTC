"""Direct Project7 Step2 model: causal state + 109-facility actions -> delta TFV.

The primary target is authoritative SWMM delta TFV, not a full future hydraulic world model.
The model represents delta TFV as a *pairwise value difference* between a complete candidate
sequence and its complete reference sequence under the same causal state and rainfall context:

    delta TFV = V(state, rain, candidate) - V(state, rain, reference)

This is intentionally stronger than encoding only ``candidate-reference`` plus the first reference
setting. Project7 Development data contain different valid reference families (D2 base action,
D3 HOLD and D4 causal Sparse-RBC anchor), while future Step3 scores candidates relative to HOLD.
Encoding both complete H360 sequences with the same value network makes the comparison structurally
reference-family aware without adding another model or another hydraulic prediction stage.

The value difference has two interpretable parts:

* per-facility main-value differences, trained primarily from exact single-actuator branches; and
* one system interaction-value difference, trained from multi-actuator branches.

Candidate == reference is exactly zero. Swapping candidate/reference flips the sign exactly. The
interaction residual is gated to exactly zero when fewer than two facilities change. Full future
hydraulic trajectory prediction remains an auxiliary ablation only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


DIRECT_TFV_VALUE_CONTRACT = "PROJECT7_DIRECT_109ACT_PAIRWISE_VALUE_TO_DELTA_TFV_V2"


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
    """Learn state-dependent TFV value differences for complete 109-facility sequences."""

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
            + 1  # normalized previous managed flow
            + design.action_blocks  # complete absolute H360 sequence for this facility
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
        b, h, _, r = rainfall.shape
        width = h // self.design.rainfall_bins
        spatial = rainfall.mean(dim=2)
        return spatial.reshape(b, self.design.rainfall_bins, width, r).mean(dim=2).reshape(b, -1)

    def _facility_context(
        self,
        *,
        current_state: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        state_context: torch.Tensor,
        rainfall_context: torch.Tensor,
    ) -> torch.Tensor:
        b = int(current_state.shape[0])
        a = self.design.actuator_count
        up = actuator_upstream.long()
        down = actuator_downstream.long()
        local_up = current_state[:, up]
        local_down = current_state[:, down]
        physics = actuator_physics.to(dtype=current_state.dtype)
        physics_mean = physics.mean(dim=0, keepdim=True)
        physics_std = physics.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        physics_norm = ((physics - physics_mean) / physics_std)[None].expand(b, -1, -1)
        ids = torch.arange(a, device=current_state.device)
        identity = self.actuator_embedding(ids)[None].expand(b, -1, -1).to(current_state.dtype)
        return torch.cat(
            (
                local_up,
                local_down,
                physics_norm,
                identity,
                previous_actuator_flow[..., None],
                state_context[:, None].expand(-1, a, -1),
                rainfall_context[:, None].expand(-1, a, -1),
            ),
            dim=-1,
        )

    def _sequence_latent(self, common: torch.Tensor, settings: torch.Tensor) -> torch.Tensor:
        blocks = self._control_blocks(settings).transpose(1, 2)
        return self.facility_encoder(torch.cat((common, blocks), dim=-1))

    def _interaction_value(
        self,
        latent: torch.Tensor,
        *,
        state_context: torch.Tensor,
        rainfall_context: torch.Tensor,
    ) -> torch.Tensor:
        pooled = latent.sum(dim=1) / math.sqrt(float(self.design.actuator_count))
        features = torch.cat((pooled, state_context, rainfall_context), dim=-1)
        return self.interaction_head(features).squeeze(-1)

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
        global_state = torch.cat((current_state.mean(dim=1), current_state.amax(dim=1)), dim=-1)
        state_context = self.global_state_encoder(global_state)
        rainfall_context = self.rainfall_encoder(self._rainfall_summary(rainfall))
        common = self._facility_context(
            current_state=current_state,
            previous_actuator_flow=previous_actuator_flow,
            actuator_upstream=actuator_upstream,
            actuator_downstream=actuator_downstream,
            actuator_physics=actuator_physics,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )

        reference_latent = self._sequence_latent(common, reference_settings)
        candidate_latent = self._sequence_latent(common, candidate_settings)
        reference_main = self.facility_head(reference_latent).squeeze(-1)
        candidate_main = self.facility_head(candidate_latent).squeeze(-1)
        facility_effect = (candidate_main - reference_main) * self.target_scale_m3

        reference_blocks = self._control_blocks(reference_settings)
        candidate_blocks = self._control_blocks(candidate_settings)
        action_delta = (candidate_blocks - reference_blocks).transpose(1, 2)
        activity = torch.mean(torch.abs(action_delta), dim=-1)
        total_activity = activity.sum(dim=-1)
        pair_mass = 0.5 * (
            torch.square(total_activity) - torch.sum(torch.square(activity), dim=-1)
        )
        pair_mass = pair_mass.clamp_min(0.0)
        pair_gate = pair_mass / (pair_mass + 0.05)
        reference_interaction = self._interaction_value(
            reference_latent,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        candidate_interaction = self._interaction_value(
            candidate_latent,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        interaction = (
            (candidate_interaction - reference_interaction)
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
