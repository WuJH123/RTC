"""V4.3.3 dimensionally consistent nodewise TFV residual.

The historical V4.3 global interaction TFV head remains the baseline.  This
module adds a zero-initialized, bounded local residual driven by the already
physical interaction flooding-rate response.  The residual is therefore an
optional correction in m3, never a replacement scalar and never a second TFV
scale multiplication.
"""

from __future__ import annotations

import torch
from torch import nn

from .step2_control_response_v41 import ReferenceEncodingV41
from .step2_control_response_v43 import (
    DifferentiableCounterfactualResponseModelV43,
    _mlp,
)


def trapezoid_delta_flood_volume_per_node_v433(
    delta_flood_rate_m3s: torch.Tensor,
    elapsed_seconds: torch.Tensor,
) -> torch.Tensor:
    """Integrate [B,C,H,N] m3/s rates over time and retain the node axis.

    The returned [B,C,N] tensor is m3.  Only the time axis is reduced; node
    contributions remain available for local diagnostics and summation.
    """

    if delta_flood_rate_m3s.dim() != 4:
        raise ValueError("delta_flood_rate_m3s must be [B,C,H,N]")
    if elapsed_seconds.shape[-1] != delta_flood_rate_m3s.shape[-2] + 1:
        raise ValueError("elapsed_seconds must contain prefix plus every forecast time")
    if not torch.isfinite(delta_flood_rate_m3s).all():
        raise ValueError("delta_flood_rate_m3s must be finite")
    if not torch.isfinite(elapsed_seconds).all():
        raise ValueError("elapsed_seconds must be finite")
    # The first forecast rate is the value at the prefix boundary.  Repeating
    # it avoids inventing a zero-rate half-step and makes a constant process
    # invariant to the number of subdivisions.
    initial = delta_flood_rate_m3s[..., :1, :]
    rate = torch.cat((initial, delta_flood_rate_m3s), dim=-2)
    dt = elapsed_seconds[..., 1:] - elapsed_seconds[..., :-1]
    while dt.dim() < rate.dim() - 1:
        dt = dt.unsqueeze(1)
    return (
        0.5 * (rate[..., 1:, :] + rate[..., :-1, :]) * dt.unsqueeze(-1)
    ).sum(dim=-2)


def nodewise_residual_parameter_names_v433(model: nn.Module) -> tuple[str, ...]:
    """Return only the new local-residual parameters."""

    return tuple(
        name for name, _ in model.named_parameters()
        if name.startswith("nodewise_residual_correction.")
    )


def set_trainable_nodewise_residual_v433(
    model: nn.Module,
    *,
    enabled: bool = True,
) -> tuple[str, ...]:
    """Freeze the complete backbone and optionally train only the residual."""

    names = set(nodewise_residual_parameter_names_v433(model))
    if enabled and not names:
        raise RuntimeError("V4.3.3 residual parameter set is empty")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(enabled and name in names))
    return tuple(sorted(names if enabled else ()))


class DifferentiableCounterfactualResponseModelV433(
    DifferentiableCounterfactualResponseModelV43
):
    """V4.3 global interaction TFV plus a corrected local residual."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, nodewise_tfv_enabled=False, **kwargs)
        self.nodewise_residual_enabled = True
        self.nodewise_residual_active = True
        residual_input_dim = self.effect_rank + self.hidden_dim + 2
        self.nodewise_residual_correction = _mlp(
            residual_input_dim, self.hidden_dim, 1
        )
        # Exact baseline equivalence at initialization, while retaining a
        # nonzero derivative for the final layer when physical volume exists.
        nn.init.zeros_(self.nodewise_residual_correction[-1].weight)
        nn.init.zeros_(self.nodewise_residual_correction[-1].bias)

    def _nodewise_local_residual_delta_tfv(
        self,
        topology_node_latent: torch.Tensor,
        interaction_flood_rate_m3s: torch.Tensor,
        reference: ReferenceEncodingV41,
        elapsed_seconds: torch.Tensor,
        interaction_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Return a signed per-candidate local TFV correction in m3."""

        if not bool(getattr(self, "nodewise_residual_active", True)):
            return interaction_gate.new_zeros(
                topology_node_latent.shape[0], topology_node_latent.shape[1]
            )
        if topology_node_latent.dim() != 5:
            raise ValueError("topology_node_latent must be [B,C,H,N,R]")
        if interaction_flood_rate_m3s.shape != topology_node_latent.shape[:-1]:
            raise ValueError("node latent and flooding-rate shapes differ")
        node_volume_m3 = trapezoid_delta_flood_volume_per_node_v433(
            interaction_flood_rate_m3s,
            elapsed_seconds,
        )
        batch, candidates, horizon, nodes, _ = topology_node_latent.shape
        reference_context = reference.node_context[:, None].expand(
            batch, candidates, horizon, nodes, -1
        )
        rate_scale_m3s = self.d3_state_scale[2].to(
            device=interaction_flood_rate_m3s.device,
            dtype=interaction_flood_rate_m3s.dtype,
        ).clamp_min(1e-6)
        volume_scale_m3 = self.d3_tfv_scale.to(
            device=interaction_flood_rate_m3s.device,
            dtype=interaction_flood_rate_m3s.dtype,
        ).clamp_min(1.0)
        rate_feature = interaction_flood_rate_m3s.div(rate_scale_m3s).unsqueeze(-1)
        volume_feature = (
            node_volume_m3.div(volume_scale_m3)
            .unsqueeze(2)
            .expand(-1, -1, horizon, -1)
            .unsqueeze(-1)
        )
        correction_input = torch.cat(
            (
                topology_node_latent,
                reference_context,
                rate_feature,
                volume_feature,
            ),
            dim=-1,
        )
        correction_weight = torch.tanh(
            self.nodewise_residual_correction(correction_input).squeeze(-1)
        )
        corrected_rate_m3s = interaction_flood_rate_m3s * correction_weight
        residual_volume_m3 = trapezoid_delta_flood_volume_per_node_v433(
            corrected_rate_m3s,
            elapsed_seconds,
        ).sum(dim=-1)
        residual_volume_m3 = residual_volume_m3 * interaction_gate.to(
            dtype=residual_volume_m3.dtype
        )
        return residual_volume_m3


__all__ = [
    "DifferentiableCounterfactualResponseModelV433",
    "nodewise_residual_parameter_names_v433",
    "set_trainable_nodewise_residual_v433",
    "trapezoid_delta_flood_volume_per_node_v433",
]
