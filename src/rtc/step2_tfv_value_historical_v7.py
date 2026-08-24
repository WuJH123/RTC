"""Development-only V7 interaction representation for Project7 Direct-TFV Step2.

V6 showed that preserving the MAIN single-facility backbone modestly improves D2 but does not
transfer to D3 joint-action generalization.  V7 therefore keeps the current Direct-TFV facility
value pathway unchanged and replaces only the joint residual with an explicitly action-difference-
conditioned interaction operator inspired by the successful mechanisms isolated in historical V4.1.

The new residual uses active-aware latent pooling, an elementwise second-order latent pair moment,
an actuator-identity-weighted signed action moment, and simple signed/absolute/quadratic action
moments.  The directed interaction score is antisymmetrized explicitly, while a continuous pair gate
keeps the interaction exactly zero for HOLD and single-facility perturbations.

This module is Development-only.  It does not alter the frozen V23/V5 publication lineage.
"""
from __future__ import annotations

import torch
from torch import nn

from .step2_tfv_value import (
    DirectFacilityTFVValueModel,
    DirectTFVValueDesign,
    DirectTFVValueOutput,
)


HISTORICAL_INTERACTION_VALUE_CONTRACT = (
    "PROJECT7_DIRECT_TFV_HISTORICAL_ACTION_IDENTITY_INTERACTION_V7"
)


class HistoricalInteractionTFVValueModelV7(DirectFacilityTFVValueModel):
    """V5-compatible main value with a bounded historical joint-action interaction residual."""

    contract = HISTORICAL_INTERACTION_VALUE_CONTRACT

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        actuator_physics_dim: int,
        target_scale_m3: float,
        design: DirectTFVValueDesign = DirectTFVValueDesign(),
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            rainfall_dim=rainfall_dim,
            actuator_physics_dim=actuator_physics_dim,
            target_scale_m3=target_scale_m3,
            design=design,
        )
        h = int(design.hidden_dim)
        interaction_input_dim = 4 * h + int(design.actuator_embedding_dim) + 3
        self.historical_interaction_head = nn.Sequential(
            nn.Linear(interaction_input_dim, h),
            nn.SiLU(),
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )
        # Start from the preserved additive MAIN model rather than injecting a random joint residual.
        final = self.historical_interaction_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _directed_interaction_features(
        self,
        *,
        base_latent: torch.Tensor,
        alt_latent: torch.Tensor,
        delta_blocks_by_facility: torch.Tensor,
        state_context: torch.Tensor,
        rainfall_context: torch.Tensor,
    ) -> torch.Tensor:
        """Construct one directed base->alternative interaction feature vector."""
        latent_delta = alt_latent - base_latent
        activity = torch.mean(torch.abs(delta_blocks_by_facility), dim=-1)
        activity_total = activity.sum(dim=-1, keepdim=True)
        weights = activity / activity_total.clamp_min(1.0e-7)
        active_pooled = torch.sum(weights[..., None] * latent_delta, dim=1)

        active_mask = activity.detach().gt(1.0e-7).to(latent_delta.dtype)
        active_count = active_mask.sum(dim=-1, keepdim=True)
        masked_delta = latent_delta * active_mask[..., None]
        hidden_sum = masked_delta.sum(dim=1)
        pair_denominator = (active_count * (active_count - 1.0)).clamp_min(1.0)
        pair_moment = (
            hidden_sum.square() - masked_delta.square().sum(dim=1)
        ) / pair_denominator

        free = int(self.design.free_control_blocks)
        signed_action = delta_blocks_by_facility[..., :free].mean(dim=-1)
        identity = self.actuator_embedding(
            torch.arange(self.design.actuator_count, device=latent_delta.device)
        ).to(dtype=latent_delta.dtype)
        signed_abs = signed_action.abs().sum(dim=-1, keepdim=True).clamp_min(1.0e-7)
        identity_moment = torch.sum(signed_action[..., None] * identity[None], dim=1) / signed_abs

        action_abs = activity.mean(dim=-1, keepdim=True)
        action_signed = signed_action.mean(dim=-1, keepdim=True)
        action_square = torch.mean(torch.square(delta_blocks_by_facility), dim=(1, 2)).unsqueeze(-1)
        return torch.cat(
            (
                active_pooled,
                pair_moment,
                identity_moment,
                state_context,
                rainfall_context,
                action_abs,
                action_signed,
                action_square,
            ),
            dim=-1,
        )

    def _historical_interaction_residual(
        self,
        *,
        reference_latent: torch.Tensor,
        candidate_latent: torch.Tensor,
        action_delta: torch.Tensor,
        state_context: torch.Tensor,
        rainfall_context: torch.Tensor,
    ) -> torch.Tensor:
        # action_delta is [B, A, blocks].  Antisymmetrization guarantees exact sign reversal when
        # candidate/reference are swapped, even though the feature map contains second-order terms.
        forward_features = self._directed_interaction_features(
            base_latent=reference_latent,
            alt_latent=candidate_latent,
            delta_blocks_by_facility=action_delta,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        reverse_features = self._directed_interaction_features(
            base_latent=candidate_latent,
            alt_latent=reference_latent,
            delta_blocks_by_facility=-action_delta,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        directed = 0.5 * (
            self.historical_interaction_head(forward_features).squeeze(-1)
            - self.historical_interaction_head(reverse_features).squeeze(-1)
        )

        activity = torch.mean(torch.abs(action_delta), dim=-1)
        total_activity = activity.sum(dim=-1)
        pair_mass = 0.5 * (
            torch.square(total_activity) - torch.sum(torch.square(activity), dim=-1)
        )
        pair_mass = pair_mass.clamp_min(0.0)
        pair_gate = pair_mass / (pair_mass + 0.05)
        return directed * self.target_scale_m3 * pair_gate

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
        interaction = self._historical_interaction_residual(
            reference_latent=reference_latent,
            candidate_latent=candidate_latent,
            action_delta=action_delta,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        total = facility_effect.sum(dim=-1) + interaction
        return DirectTFVValueOutput(
            total_delta_tfv_m3=total,
            facility_main_effect_m3=facility_effect,
            interaction_residual_m3=interaction,
            action_activity=activity,
        )


__all__ = [
    "HISTORICAL_INTERACTION_VALUE_CONTRACT",
    "HistoricalInteractionTFVValueModelV7",
]
