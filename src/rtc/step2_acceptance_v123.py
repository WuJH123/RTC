"""Fail-closed scientific acceptance for Project7 V12.3 Value control.

Finite shooting and continuous differentiable MPC are different claims.  A causal Value
model may be used for calibrated Development finite shooting while continuous gradient
search remains blocked.  This module makes that distinction executable instead of
leaving it as a comment/runbook convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

V123_VALUE_ACCEPTANCE_CONTRACT = "PROJECT7_V123_CAUSAL_VALUE_CONTINUOUS_GATE_V1"


@dataclass(frozen=True)
class ContinuousValueGateV123:
    rank_min: float = 0.70
    top1_min: float = 0.50
    gradient_sign_min: float = 0.70
    gradient_cosine_min: float = 0.60

    def validate(self) -> None:
        values = tuple(float(value) for value in asdict(self).values())
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("V123 continuous Value thresholds must lie in [0,1]")


def evaluate_continuous_value_gate_v123(
    *,
    causal_input_verified: bool,
    holdout_rank: float,
    holdout_top1: float,
    gradient_sign_accuracy: float,
    gradient_cosine: float,
    contract: ContinuousValueGateV123 = ContinuousValueGateV123(),
) -> dict[str, object]:
    """Return an auditable gate; never silently enable continuous optimisation."""
    contract.validate()
    metrics = {
        "holdout_rank": float(holdout_rank),
        "holdout_top1": float(holdout_top1),
        "gradient_sign_accuracy": float(gradient_sign_accuracy),
        "gradient_cosine": float(gradient_cosine),
    }
    finite = all(math.isfinite(value) for value in metrics.values())
    checks = {
        "causal_input_verified": bool(causal_input_verified),
        "metrics_finite": bool(finite),
        "rank": bool(finite and metrics["holdout_rank"] >= contract.rank_min),
        "top1": bool(finite and metrics["holdout_top1"] >= contract.top1_min),
        "gradient_sign": bool(
            finite and metrics["gradient_sign_accuracy"] >= contract.gradient_sign_min
        ),
        "gradient_cosine": bool(
            finite and metrics["gradient_cosine"] >= contract.gradient_cosine_min
        ),
    }
    passed = all(checks.values())
    return {
        "contract": V123_VALUE_ACCEPTANCE_CONTRACT,
        "thresholds": asdict(contract),
        "metrics": metrics,
        "checks": checks,
        "continuous_gradient_search": bool(passed),
        "verdict": (
            "V123_CONTINUOUS_GRADIENT_ACCEPTED"
            if passed
            else "V123_GRADIENT_BLOCKED_FINITE_ONLY"
        ),
    }


__all__ = [
    "ContinuousValueGateV123",
    "V123_VALUE_ACCEPTANCE_CONTRACT",
    "evaluate_continuous_value_gate_v123",
]
