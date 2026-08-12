"""V4.3.2 primary-TFV consistency primitives.

This module deliberately keeps the V4.3 topology propagator unchanged.  It
adds only the nodewise additive TFV mode and the small, independently tested
gradient-surgery primitive used by the bounded D3 ablation.
"""

from __future__ import annotations

import torch

from .step2_control_response_v43 import (
    DifferentiableCounterfactualResponseModelV43,
    interaction_parameter_names,
    parameter_sha256,
    phase_parameter_names,
    reference_parameter_names,
    set_trainable_phase,
    single_parameter_names,
)


class DifferentiableCounterfactualResponseModelV432(
    DifferentiableCounterfactualResponseModelV43
):
    """V4.3 model with nodewise additive interaction TFV enabled."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["nodewise_tfv_enabled"] = True
        super().__init__(*args, **kwargs)


def nodewise_tfv_from_contributions_v432(
    contributions: torch.Tensor,
) -> torch.Tensor:
    """Sum signed per-node (or per-time/per-node) contributions.

    The final dimensions are intentionally reduced by summation, never mean
    pooling.  A [B,C,N] input returns [B,C]; a [B,C,H,N] input does the same.
    """

    if contributions.dim() < 3:
        raise ValueError("contributions must be [B,C,N] or [B,C,H,N]")
    if not torch.isfinite(contributions).all():
        raise ValueError("contributions must be finite")
    return contributions.sum(dim=tuple(range(2, contributions.dim())))


def project_auxiliary_gradient_v432(
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project an auxiliary gradient away from a conflicting primary gradient.

    The primary objective is never modified.  A zero/non-finite primary is a
    fail-closed condition because a direction-preserving projection is then
    undefined.
    """

    if primary.shape != auxiliary.shape:
        raise ValueError("primary and auxiliary gradients must have equal shape")
    if not torch.isfinite(primary).all() or not torch.isfinite(auxiliary).all():
        raise ValueError("gradients must be finite")
    primary_norm_sq = torch.dot(primary.reshape(-1), primary.reshape(-1))
    if not torch.isfinite(primary_norm_sq) or primary_norm_sq <= eps:
        raise RuntimeError("zero primary gradient; cannot project auxiliary")
    dot = torch.dot(auxiliary.reshape(-1), primary.reshape(-1))
    if dot >= 0:
        return auxiliary.clone()
    coefficient = dot / (primary_norm_sq + eps)
    projected = auxiliary - coefficient * primary
    if not torch.isfinite(projected).all():
        raise RuntimeError("projected auxiliary gradient is non-finite")
    return projected


__all__ = [
    "DifferentiableCounterfactualResponseModelV432",
    "interaction_parameter_names",
    "nodewise_tfv_from_contributions_v432",
    "parameter_sha256",
    "phase_parameter_names",
    "project_auxiliary_gradient_v432",
    "reference_parameter_names",
    "set_trainable_phase",
    "single_parameter_names",
]
