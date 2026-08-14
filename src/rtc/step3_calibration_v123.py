"""Development-only one-sided error calibration for V12.3 finite RTC.

The controller should not issue a command for a tiny predicted improvement when the
Value model's own SWMM residual is larger.  Rather than hard-coding an after-the-fact
1000 m3 threshold, V12.3 derives a one-sided error budget from a frozen development
calibration split and requires predicted TFV benefit to clear that budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

V123_ADMISSION_CONTRACT = "PROJECT7_V123_ONE_SIDED_VALUE_FALSE_BENEFIT_CALIBRATION_V1"


@dataclass(frozen=True)
class OneSidedValueCalibrationV123:
    quantile: float
    residual_count: int
    rainfall_group_count: int
    tfv_truth_minus_prediction_q_m3: float
    pfv_truth_minus_prediction_q_m3: float

    @property
    def tfv_false_benefit_margin_m3(self) -> float:
        return max(float(self.tfv_truth_minus_prediction_q_m3), 0.0)

    def validate(self) -> None:
        if not 0.5 < float(self.quantile) < 1.0:
            raise ValueError("V123 one-sided quantile must lie in (0.5,1)")
        if self.residual_count <= 0 or self.rainfall_group_count <= 0:
            raise ValueError("V123 calibration requires positive support counts")
        if not all(math.isfinite(float(v)) for v in (
            self.tfv_truth_minus_prediction_q_m3,
            self.pfv_truth_minus_prediction_q_m3,
        )):
            raise ValueError("V123 calibration contains non-finite residual quantiles")

    def as_payload(self) -> dict[str, object]:
        self.validate()
        return {"contract": V123_ADMISSION_CONTRACT, **asdict(self),
                "tfv_false_benefit_margin_m3": self.tfv_false_benefit_margin_m3}


def _event_balanced_residual_sample(
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    rainfall_groups: Sequence[str],
) -> tuple[np.ndarray, int]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    groups = np.asarray([str(x) for x in rainfall_groups])
    if truth.shape != prediction.shape or truth.shape != groups.shape or truth.size == 0:
        raise ValueError("V123 calibration truth/prediction/group vectors must align")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("V123 calibration vectors contain non-finite values")

    # Give each rainfall group equal total mass by taking deterministic empirical
    # quantile grids within group. This avoids a group with more checkpoints dominating
    # the one-sided error budget.
    samples: list[np.ndarray] = []
    unique = sorted(set(groups.tolist()))
    per_group = min(int(np.sum(groups == group)) for group in unique)
    if per_group <= 0:
        raise ValueError("V123 calibration has an empty rainfall group")
    grid = (np.arange(per_group, dtype=np.float64) + 0.5) / per_group
    for group in unique:
        residual = np.sort((truth - prediction)[groups == group])
        indices = np.minimum((grid * residual.size).astype(int), residual.size - 1)
        samples.append(residual[indices])
    return np.concatenate(samples), len(unique)


def fit_one_sided_value_calibration_v123(
    *,
    tfv_truth_m3: Sequence[float] | np.ndarray,
    tfv_prediction_m3: Sequence[float] | np.ndarray,
    pfv_truth_m3: Sequence[float] | np.ndarray,
    pfv_prediction_m3: Sequence[float] | np.ndarray,
    rainfall_groups: Sequence[str],
    quantile: float = 0.90,
) -> OneSidedValueCalibrationV123:
    """Fit a deterministic development calibration; never call this on Final evidence."""
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("V123 calibration quantile must lie in (0.5,1)")
    tfv_residual, group_count = _event_balanced_residual_sample(
        tfv_truth_m3, tfv_prediction_m3, rainfall_groups
    )
    pfv_residual, pfv_group_count = _event_balanced_residual_sample(
        pfv_truth_m3, pfv_prediction_m3, rainfall_groups
    )
    if pfv_group_count != group_count or pfv_residual.size != tfv_residual.size:
        raise ValueError("V123 TFV/PFV calibration supports do not align")
    result = OneSidedValueCalibrationV123(
        quantile=float(quantile),
        residual_count=int(tfv_residual.size),
        rainfall_group_count=int(group_count),
        tfv_truth_minus_prediction_q_m3=float(np.quantile(tfv_residual, quantile)),
        pfv_truth_minus_prediction_q_m3=float(np.quantile(pfv_residual, quantile)),
    )
    result.validate()
    return result


__all__ = [
    "OneSidedValueCalibrationV123",
    "V123_ADMISSION_CONTRACT",
    "fit_one_sided_value_calibration_v123",
]
