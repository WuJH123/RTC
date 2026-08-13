"""Publication-strength Step2 evidence gate for Project7 V12.2.

V120 demonstrated useful finite-bank TFV ranking, but V12.2 continuous MPC interpolates
between labelled candidates.  That stronger use requires stronger evidence: value skill,
local SWMM gradient agreement, reference semantics and action-support coverage must all
pass before the differentiable optimiser can be instantiated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

V122_STEP2_ACCEPTANCE_CONTRACT = "PROJECT7_V122_VALUE_AND_LOCAL_GRADIENT_ACCEPTANCE_V1"


@dataclass(frozen=True)
class Step2AcceptanceThresholdsV122:
    rank_correlation_min: float = 0.70
    pairwise_accuracy_min: float = 0.70
    sign_accuracy_min: float = 0.70
    top1_rate_min: float = 0.50
    gradient_sign_accuracy_min: float = 0.70
    gradient_cosine_similarity_min: float = 0.60
    spread_ratio_min: float = 0.50
    spread_ratio_max: float = 1.75
    response_ratio_min: float = 0.50
    response_ratio_max: float = 1.75
    local_gradient_case_min: int = 100

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "local_gradient_case_min":
                if int(value) <= 0:
                    raise ValueError("V122 local gradient case minimum must be positive")
                continue
            if not np.isfinite(float(value)):
                raise ValueError(f"V122 threshold {name} is non-finite")
        if not 0.0 <= self.rank_correlation_min <= 1.0:
            raise ValueError("invalid rank threshold")
        for value in (
            self.pairwise_accuracy_min,
            self.sign_accuracy_min,
            self.top1_rate_min,
            self.gradient_sign_accuracy_min,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("invalid probability/accuracy threshold")
        if not -1.0 <= self.gradient_cosine_similarity_min <= 1.0:
            raise ValueError("invalid gradient cosine threshold")
        if not 0.0 < self.spread_ratio_min <= self.spread_ratio_max:
            raise ValueError("invalid spread-ratio range")
        if not 0.0 < self.response_ratio_min <= self.response_ratio_max:
            raise ValueError("invalid response-ratio range")


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
    require_min(value_metrics, "pairwise_accuracy", thresholds.pairwise_accuracy_min)
    require_min(value_metrics, "sign_accuracy", thresholds.sign_accuracy_min)
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
    gradient_cases = float(gradient_metrics.get("gradient_cases", float("nan")))
    if not np.isfinite(gradient_cases) or gradient_cases < thresholds.local_gradient_case_min:
        reasons.append(
            f"gradient_cases={gradient_cases} < {thresholds.local_gradient_case_min}"
        )

    for key, lower, upper in (
        (
            "spread_ratio",
            thresholds.spread_ratio_min,
            thresholds.spread_ratio_max,
        ),
        (
            "response_ratio",
            thresholds.response_ratio_min,
            thresholds.response_ratio_max,
        ),
    ):
        value = float(value_metrics.get(key, float("nan")))
        if not np.isfinite(value) or not lower <= value <= upper:
            reasons.append(f"{key}={value} outside [{lower},{upper}]")

    if not passive_reference_verified:
        reasons.append("passive no-new-command reference semantics are not verified")
    if not train_runtime_action_support_verified:
        reasons.append("runtime action search exceeds the demonstrated Train/D3 support")

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
