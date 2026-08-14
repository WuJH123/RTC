"""V125 anchor-relative TFV/PFV scoring.

Unlike V123 passive-relative scoring, the direct Sparse-RBC reference has known exact
zero counterfactual difference.  The calibrated PFV model-error margin therefore applies
only to candidates whose executable first move differs from the anchor; applying it to
the anchor itself would destroy the reference identity and could penalise the default.
"""
from __future__ import annotations

import torch

from .step3_objective_v123 import TFVPFVObjectiveV123


def tfv_pfv_score_v125(
    delta_tfv_scenarios_m3: torch.Tensor,
    delta_pfv_scenarios_m3: torch.Tensor,
    *,
    movement_from_anchor: torch.Tensor,
    contract: TFVPFVObjectiveV123,
    anchor_atol: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    contract.validate()
    if delta_tfv_scenarios_m3.shape != delta_pfv_scenarios_m3.shape:
        raise ValueError("V125 TFV/PFV scenario tensors must have identical shape")
    if delta_tfv_scenarios_m3.ndim != 2:
        raise ValueError("V125 TFV/PFV scenario tensors must be [scenario,candidate]")
    movement = torch.as_tensor(
        movement_from_anchor,
        dtype=delta_tfv_scenarios_m3.dtype,
        device=delta_tfv_scenarios_m3.device,
    ).reshape(-1)
    if movement.numel() != delta_tfv_scenarios_m3.shape[1]:
        raise ValueError("V125 movement must have one value per candidate")
    if not bool(torch.isfinite(movement).all()) or bool((movement < 0.0).any()):
        raise ValueError("V125 movement must be finite and non-negative")

    tfv_mean = delta_tfv_scenarios_m3.mean(dim=0)
    pfv_mean = delta_pfv_scenarios_m3.mean(dim=0)
    tfv_std = delta_tfv_scenarios_m3.std(dim=0, unbiased=False)
    pfv_std = delta_pfv_scenarios_m3.std(dim=0, unbiased=False)
    tfv_risk = tfv_mean + float(contract.risk_weight) * tfv_std
    pfv_risk_without_error = pfv_mean + float(contract.risk_weight) * pfv_std
    changed = (movement > float(anchor_atol)).to(dtype=pfv_mean.dtype)
    pfv_error = changed * float(contract.pfv_model_error_margin_m3)
    pfv_risk = pfv_risk_without_error + pfv_error
    pfv_soft_excess = torch.clamp(
        pfv_risk - float(contract.pfv_soft_margin_m3), min=0.0
    )
    pfv_penalty = (
        float(contract.pfv_penalty_weight)
        * float(contract.tfv_scale_m3)
        * pfv_soft_excess
        / float(contract.pfv_scale_m3)
    )
    movement_penalty = float(contract.movement_penalty_m3) * movement
    score = tfv_risk + pfv_penalty + movement_penalty
    return {
        "delta_tfv_mean_m3": tfv_mean,
        "delta_pfv_mean_m3": pfv_mean,
        "delta_tfv_std_m3": tfv_std,
        "delta_pfv_std_m3": pfv_std,
        "tfv_risk_m3": tfv_risk,
        "pfv_risk_without_model_error_m3": pfv_risk_without_error,
        "pfv_model_error_applied_m3": pfv_error,
        "pfv_risk_m3": pfv_risk,
        "pfv_soft_excess_m3": pfv_soft_excess,
        "pfv_penalty_m3_equivalent": pfv_penalty,
        "movement_penalty_m3_equivalent": movement_penalty,
        "score_m3_equivalent": score,
    }


__all__ = ["tfv_pfv_score_v125"]
