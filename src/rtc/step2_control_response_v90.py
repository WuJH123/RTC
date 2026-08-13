"""V9 signed hydraulic-effect output and physical reconstruction helpers.

The signed counterfactual effect is a primary prediction.  Physical candidate
reconstruction is a separate diagnostic and must never overwrite that effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .step2_control_response_v80 import (
    DirectHydraulicEffectSurrogateV80,
    PreparedStaticV80,
)


@dataclass(frozen=True)
class HydraulicEffectOutputV90:
    """Raw signed effects plus independently projected candidate diagnostics."""

    raw_delta_states_physical: torch.Tensor
    raw_delta_flows_physical: torch.Tensor
    reference_states_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    candidate_states_projected_physical: torch.Tensor
    candidate_flows_projected_physical: torch.Tensor
    horizon_indices: torch.Tensor | None = None
    reference_flood_onset_logits: torch.Tensor | None = None
    candidate_flood_onset_logits: torch.Tensor | None = None
    joint_context_before_scatter: torch.Tensor | None = None

    @property
    def delta_states_physical(self) -> torch.Tensor:
        """Compatibility view used by the frozen V8 sparse-effect objective."""
        return self.raw_delta_states_physical

    @property
    def delta_flows_physical(self) -> torch.Tensor:
        """Compatibility view used by the frozen V8 sparse-effect objective."""
        return self.raw_delta_flows_physical

    @property
    def candidate_states_physical(self) -> torch.Tensor:
        return self.candidate_states_projected_physical

    @property
    def candidate_flows_physical(self) -> torch.Tensor:
        return self.candidate_flows_projected_physical


def project_candidate_states_v90(
    reference_states_physical: torch.Tensor,
    raw_delta_states_physical: torch.Tensor,
    *,
    invert_elevation_m: torch.Tensor,
) -> torch.Tensor:
    """Project a candidate state for absolute diagnostics only.

    Channel contract is ``depth, head, flooding, storage, inflow, outflow``.
    Signed raw effects are not clipped or reconstructed here; the projection is
    intentionally a one-way diagnostic transform.
    """
    if reference_states_physical.shape != raw_delta_states_physical.shape:
        raise ValueError("V9 state reference/effect shapes must match")
    if reference_states_physical.shape[-1] != 6:
        raise ValueError("V9 state channel contract requires six channels")
    invert = torch.as_tensor(
        invert_elevation_m, dtype=reference_states_physical.dtype,
        device=reference_states_physical.device,
    )
    if invert.ndim != 1 or invert.numel() != reference_states_physical.shape[-2]:
        raise ValueError("V9 invert elevation must match node dimension")
    depth = (reference_states_physical[..., 0] + raw_delta_states_physical[..., 0]).clamp_min(0.0)
    return torch.stack(
        (
            depth,
            depth + invert.reshape(*((1,) * (depth.ndim - 1)), -1),
            (reference_states_physical[..., 2] + raw_delta_states_physical[..., 2]).clamp_min(0.0),
            (reference_states_physical[..., 3] + raw_delta_states_physical[..., 3]).clamp_min(0.0),
            (reference_states_physical[..., 4] + raw_delta_states_physical[..., 4]).clamp_min(0.0),
            (reference_states_physical[..., 5] + raw_delta_states_physical[..., 5]).clamp_min(0.0),
        ),
        dim=-1,
    )


def project_candidate_flows_v90(
    reference_flows_physical: torch.Tensor,
    raw_delta_flows_physical: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct signed actuator flows without imposing a false non-negative prior."""
    if reference_flows_physical.shape != raw_delta_flows_physical.shape:
        raise ValueError("V9 flow reference/effect shapes must match")
    return reference_flows_physical + raw_delta_flows_physical


class TrajectoryConditionedHydraulicEffectSurrogateV90(nn.Module):
    """V8 effect branch with optional detached predicted/oracle trajectory context."""

    def __init__(self, base_model: DirectHydraulicEffectSurrogateV80) -> None:
        super().__init__()
        self.base_model = base_model

    @property
    def reference_model(self):
        return self.base_model.reference_model

    @property
    def graph_blocks(self):
        return self.base_model.graph_blocks

    def trainable_parameters(self):
        return self.base_model.trainable_parameters()

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV80,
        *,
        reference_trajectory_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> HydraulicEffectOutputV90:
        output = self.base_model(
            initial_state,
            rainfall,
            reference_settings,
            candidate_settings,
            previous_actuator_flow,
            prepared,
            reference_trajectory_context=reference_trajectory_context,
        )
        raw_state = output.raw_delta_states_physical
        raw_flow = output.raw_delta_flows_physical
        if raw_state is None or raw_flow is None:
            raise RuntimeError("V9 requires raw signed effects from the base model")
        projected_states = project_candidate_states_v90(
            output.reference_states_physical,
            raw_state,
            invert_elevation_m=prepared.base.invert_elevation_m,
        )
        projected_flows = project_candidate_flows_v90(
            output.reference_flows_physical,
            raw_flow,
        )
        return HydraulicEffectOutputV90(
            raw_delta_states_physical=raw_state,
            raw_delta_flows_physical=raw_flow,
            reference_states_physical=output.reference_states_physical,
            reference_flows_physical=output.reference_flows_physical,
            candidate_states_projected_physical=projected_states,
            candidate_flows_projected_physical=projected_flows,
            horizon_indices=output.horizon_indices,
            reference_flood_onset_logits=output.reference_flood_onset_logits,
            candidate_flood_onset_logits=output.candidate_flood_onset_logits,
            joint_context_before_scatter=output.joint_context_before_scatter,
        )


__all__ = [
    "HydraulicEffectOutputV90",
    "TrajectoryConditionedHydraulicEffectSurrogateV90",
    "project_candidate_flows_v90",
    "project_candidate_states_v90",
]
