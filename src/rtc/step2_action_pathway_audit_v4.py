"""Small, reusable diagnostics for the isolated Step2 response experiment."""

from __future__ import annotations

import numpy as np
import torch

from .step2_counterfactual import same_prefix_diagnostic


def direct_pair_delta_tfv(
    candidate_flood_rate: torch.Tensor,
    reference_flood_rate: torch.Tensor,
    *,
    dt_seconds: float | torch.Tensor,
    smooth: bool = False,
    softplus_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Integrate candidate-minus-reference flooding rate before node/time reduction.

    This is a diagnostic/training operator, not authoritative SWMM truth.  The direct
    subtraction avoids subtracting two large cumulative node totals.  Accumulation is
    forced to at least FP32 and the operator is causal over the supplied trajectory.
    """

    if candidate_flood_rate.shape != reference_flood_rate.shape:
        raise ValueError("candidate/reference flooding-rate shapes differ")
    if candidate_flood_rate.dim() < 3:
        raise ValueError("flooding-rate trajectory must include batch, time and node axes")
    candidate = candidate_flood_rate.float()
    reference = reference_flood_rate.float()
    if smooth:
        scale = torch.as_tensor(softplus_scale, device=candidate.device, dtype=candidate.dtype).clamp_min(1e-6)
        candidate = torch.nn.functional.softplus(candidate / scale) * scale
        reference = torch.nn.functional.softplus(reference / scale) * scale
    else:
        candidate = candidate.clamp_min(0.0)
        reference = reference.clamp_min(0.0)
    delta_rate = candidate - reference
    dt = torch.as_tensor(dt_seconds, device=delta_rate.device, dtype=delta_rate.dtype)
    if dt.numel() == 1:
        return delta_rate.sum(dim=tuple(range(1, delta_rate.dim()))) * dt.reshape(())
    if dt.dim() == 1 and dt.shape[0] == delta_rate.shape[1]:
        shape = [1, dt.shape[0]] + [1] * (delta_rate.dim() - 2)
    elif dt.dim() == 2 and dt.shape == delta_rate.shape[:2]:
        shape = [dt.shape[0], dt.shape[1]] + [1] * (delta_rate.dim() - 2)
    else:
        raise ValueError("dt_seconds must be scalar, [H] or [B,H]")
    return (delta_rate * dt.reshape(shape)).sum(dim=tuple(range(1, delta_rate.dim())))


def verify_pair_prefix(
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Return a structured same-prefix result without weakening the guard."""

    try:
        same_prefix_diagnostic(initial_state, rainfall, previous_flow, atol=atol)
    except ValueError as exc:
        return {"passed": False, "error": str(exc)}
    return {"passed": True, "error": None}


def response_norms(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Summarize candidate-reference response magnitudes for an audit table."""

    delta = (candidate.detach().float() - reference.detach().float()).cpu().numpy()
    rows = delta.reshape(delta.shape[0], -1)
    norms = np.linalg.norm(rows, axis=1)
    return {
        "mean_l2": float(np.mean(norms)),
        "median_l2": float(np.median(norms)),
        "max_l2": float(np.max(norms)),
    }
