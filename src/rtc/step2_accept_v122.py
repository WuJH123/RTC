"""Publication-strength Step2 evidence gate for Project7 V12.2.

Hard numeric thresholds are inherited from the preregistered
``MODEL_ACCEPTANCE_CONTRACT_V4_DIMENSIONLESS_PREREGISTERED`` contract.  V12.2 adds
semantic/action-support gates because continuous MPC interpolates between labelled
candidates.  Other value metrics remain mandatory diagnostics rather than new ad-hoc
pass/fail thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

V122_STEP2_ACCEPTANCE_CONTRACT = "PROJECT7_V122_VALUE_AND_LOCAL_GRADIENT_ACCEPTANCE_V2"


@dataclass(frozen=True)
class Step2AcceptanceThresholdsV122:
    rank_correlation_min: float = 0.70
    top1_rate_min: float = 0.50
    gradient_sign_accuracy_min: float = 0.70
    gradient_cosine_similarity_min: float = 0.60

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"V122 threshold {name} is non-finite")
        if not 0.0 <= self.rank_correlation_min <= 1.0:
            raise ValueError("invalid V122 rank threshold")
        if not 0.0 <= self.top1_rate_min <= 1.0:
            raise ValueError("invalid V122 top1 threshold")
        if not 0.0 <= self.gradient_sign_accuracy_min <= 1.0:
            raise ValueError("invalid V122 gradient-sign threshold")
        if not -1.0 <= self.gradient_cosine_similarity_min <= 1.0:
            raise ValueError("invalid V122 gradient-cosine threshold")


@dataclass(frozen=True)
class Step2AcceptanceV122:
    passed: bool
    reasons: tuple[str, ...]
    value_metrics: Mapping[str, float]
    gradient_metrics: Mapping[str, float]
    passive_reference_verified: bool
    train_runtime_action_support_verified: bool


def evaluate_step2_acceptance_v122(
    *,
    value_metrics: Mapping[str, float],
    gradient_metrics: Mapping[str, float],
    passive_reference_verified: bool,
    train_runtime_action_support_verified: bool,
    thresholds: Step2AcceptanceThresholdsV122 = Step2AcceptanceThresholdsV122(),
) -> Step2AcceptanceV122:
    thresholds.validate()
    reasons: list[str] = []

    def require_min(source: Mapping[str, float], key: str, floor: float) -> None:
        value = float(source.get(key, float("nan")))
        if not np.isfinite(value) or value < floor:
            reasons.append(f"{key}={value} < {floor}")

    require_min(value_metrics, "rank", thresholds.rank_correlation_min)
    require_min(value_metrics, "top1_rate", thresholds.top1_rate_min)
    require_min(
        gradient_metrics,
        "tfv_gradient_sign_accuracy",
        thresholds.gradient_sign_accuracy_min,
    )
    require_min(
        gradient_metrics,
        "tfv_gradient_cosine_similarity",
        thresholds.gradient_cosine_similarity_min,
    )

    # These metrics are explicitly required diagnostics in the existing scientific
    # contract.  They must be finite when emitted, but V122 does not invent new hard
    # thresholds after prior development results exist.
    for key in (
        "pairwise",
        "sign_accuracy",
        "tfv_mae_m3",
        "mean_regret_m3",
        "max_regret_m3",
        "spread_ratio",
        "response_ratio",
    ):
        if key in value_metrics and not np.isfinite(float(value_metrics[key])):
            reasons.append(f"required value diagnostic {key} is non-finite")
    for key in (
        "tfv_gradient_mae",
        "gradient_cases",
        "gradient_rainfall_groups",
    ):
        if key not in gradient_metrics or not np.isfinite(float(gradient_metrics[key])):
            reasons.append(f"required gradient diagnostic {key} is missing/non-finite")

    if not passive_reference_verified:
        reasons.append("passive no-new-command reference semantics are not verified")
    if not train_runtime_action_support_verified:
        reasons.append("runtime action search exceeds demonstrated Train/D3 support")

    return Step2AcceptanceV122(
        passed=not reasons,
        reasons=tuple(reasons),
        value_metrics=dict(value_metrics),
        gradient_metrics=dict(gradient_metrics),
        passive_reference_verified=bool(passive_reference_verified),
        train_runtime_action_support_verified=bool(train_runtime_action_support_verified),
    )


__all__ = [
    "Step2AcceptanceThresholdsV122",
    "Step2AcceptanceV122",
    "V122_STEP2_ACCEPTANCE_CONTRACT",
    "evaluate_step2_acceptance_v122",
]
