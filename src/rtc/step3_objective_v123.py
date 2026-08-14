"""Project7 V12.3 Step3 objective.

TFV remains the primary whole-system objective.  PFV is a one-sided *soft* penalty:
only deterioration beyond a frozen margin is penalised, PFV improvement does not buy
worse TFV, and PFV is never converted into a hard feasibility condition.

Because the PFV Value model can under-predict deterioration, the soft penalty uses a
Development-calibrated one-sided PFV error budget.  The error budget is applied only to
non-passive executable first moves so the exact HOLD reference remains exactly zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

V123_OBJECTIVE_CONTRACT = "PROJECT7_V123_TFV_PRIMARY_SOFT_PFV_CALIBRATED_RISK_V2"


@dataclass(frozen=True)
class TFVPFVObjectiveV123:
    pfv_soft_margin_m3: float
    pfv_scale_m3: float
    tfv_scale_m3: float
    pfv_penalty_weight: float
    pfv_model_error_margin_m3: float = 0.0
    tfv_uncertainty_weight: float = 0.0
    pfv_uncertainty_weight: float = 0.0
    movement_penalty_m3: float = 0.0

    def validate(self) -> None:
        values = tuple(float(v) for v in self.__dict__.values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError("V123 objective contains non-finite values")
        if self.pfv_soft_margin_m3 < 0.0 or self.pfv_model_error_margin_m3 < 0.0:
            raise ValueError("V123 PFV margins cannot be negative")
        if self.pfv_scale_m3 <= 0.0 or self.tfv_scale_m3 <= 0.0:
            raise ValueError("V123 volume scales must be positive")
        if min(
            self.pfv_penalty_weight,
            self.tfv_uncertainty_weight,
            self.pfv_uncertainty_weight,
            self.movement_penalty_m3,
        ) < 0.0:
            raise ValueError("V123 objective weights cannot be negative")


def _mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim < 1 or values.shape[0] <= 0:
        raise ValueError("V123 ensemble tensor requires a non-empty leading model axis")
    mean = values.mean(dim=0)
    std = (
        torch.zeros_like(mean)
        if values.shape[0] == 1
        else values.std(dim=0, unbiased=False)
    )
    return mean, std


def tfv_pfv_score_v123(
    delta_tfv_ensemble_m3: torch.Tensor,
    delta_pfv_ensemble_m3: torch.Tensor,
    *,
    movement: torch.Tensor | None,
    contract: TFVPFVObjectiveV123,
) -> dict[str, torch.Tensor]:
    """Return TFV-like m3-equivalent candidate scores and diagnostics."""
    contract.validate()
    if delta_tfv_ensemble_m3.shape != delta_pfv_ensemble_m3.shape:
        raise ValueError("V123 TFV/PFV ensemble predictions must align")
    if not bool(torch.isfinite(delta_tfv_ensemble_m3).all()) or not bool(
        torch.isfinite(delta_pfv_ensemble_m3).all()
    ):
        raise ValueError("V123 objective received non-finite predictions")

    tfv_mean, tfv_std = _mean_std(delta_tfv_ensemble_m3)
    pfv_mean, pfv_std = _mean_std(delta_pfv_ensemble_m3)
    tfv_risk = tfv_mean + float(contract.tfv_uncertainty_weight) * tfv_std

    if movement is None:
        movement_penalty = torch.zeros_like(tfv_risk)
        active = torch.ones_like(tfv_risk, dtype=torch.bool)
    else:
        movement = movement.to(tfv_risk)
        if (
            movement.shape != tfv_risk.shape
            or not bool(torch.isfinite(movement).all())
            or bool(torch.any(movement < 0.0))
        ):
            raise ValueError("V123 movement must be finite, non-negative and score-aligned")
        movement_penalty = float(contract.movement_penalty_m3) * movement
        active = movement > 1.0e-12

    # Conservative PFV risk for active commands only.  The exact PASSIVE/HOLD reference
    # must remain zero even when the calibrated error margin is positive.
    pfv_error = torch.where(
        active,
        torch.full_like(pfv_mean, float(contract.pfv_model_error_margin_m3)),
        torch.zeros_like(pfv_mean),
    )
    pfv_risk = (
        pfv_mean
        + float(contract.pfv_uncertainty_weight) * pfv_std
        + pfv_error
    )
    pfv_excess = torch.relu(pfv_risk - float(contract.pfv_soft_margin_m3))
    pfv_penalty = (
        float(contract.pfv_penalty_weight)
        * float(contract.tfv_scale_m3)
        * pfv_excess
        / float(contract.pfv_scale_m3)
    )

    return {
        "score_m3_equivalent": tfv_risk + pfv_penalty + movement_penalty,
        "delta_tfv_mean_m3": tfv_mean,
        "delta_tfv_std_m3": tfv_std,
        "tfv_risk_m3": tfv_risk,
        "delta_pfv_mean_m3": pfv_mean,
        "delta_pfv_std_m3": pfv_std,
        "pfv_model_error_margin_m3": pfv_error,
        "pfv_risk_m3": pfv_risk,
        "pfv_soft_excess_m3": pfv_excess,
        "pfv_penalty_m3_equivalent": pfv_penalty,
        "movement_penalty_m3_equivalent": movement_penalty,
    }


__all__ = ["TFVPFVObjectiveV123", "V123_OBJECTIVE_CONTRACT", "tfv_pfv_score_v123"]
