"""Calibration and audit helpers for V125 anchor-relative learned overrides."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

from .step2_policy_v125 import V125_OVERRIDE_CALIBRATION_CONTRACT


@dataclass(frozen=True)
class AnchorOverrideCalibrationV125:
    quantile: float
    margin_m3: float
    sample_count: int
    rainfall_groups: tuple[str, ...]
    row_identity_sha256: str
    contract: str = V125_OVERRIDE_CALIBRATION_CONTRACT

    def validate(self) -> None:
        if not 0.5 < float(self.quantile) < 1.0:
            raise ValueError("V125 calibration quantile must lie in (0.5,1)")
        if not np.isfinite(self.margin_m3) or self.margin_m3 < 0.0:
            raise ValueError("V125 calibration margin must be finite and non-negative")
        if self.sample_count <= 0 or not self.rainfall_groups:
            raise ValueError("V125 calibration cannot be empty")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "contract": self.contract,
            "quantile": float(self.quantile),
            "anchor_override_margin_m3": float(self.margin_m3),
            "sample_count": int(self.sample_count),
            "rainfall_groups": list(self.rainfall_groups),
            "row_identity_sha256": self.row_identity_sha256,
        }


def _higher_quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # numpy<1.22 compatibility
        return float(np.quantile(values, q, interpolation="higher"))


def calibrate_anchor_override_margin_v125(
    *,
    truth_candidate_tfv_m3: Sequence[float],
    truth_anchor_tfv_m3: Sequence[float],
    predicted_candidate_delta_tfv_m3: Sequence[float],
    predicted_anchor_delta_tfv_m3: Sequence[float],
    rainfall_groups: Sequence[str],
    row_ids: Sequence[str],
    quantile: float = 0.95,
) -> AnchorOverrideCalibrationV125:
    """Calibrate a one-sided upper error budget for candidate-vs-anchor TFV advantage.

    Advantage is ``candidate - anchor``; negative is beneficial.  With
    ``residual = truth_advantage - predicted_advantage``, admitting only when
    ``predicted_advantage + margin < 0`` is an empirical one-sided false-benefit guard.
    The caller must pass D4-FIT rows only; D4-AUDIT is intentionally external.
    """
    q = float(quantile)
    if not 0.5 < q < 1.0:
        raise ValueError("V125 quantile must lie in (0.5,1)")
    arrays = [
        np.asarray(x, dtype=np.float64).reshape(-1)
        for x in (
            truth_candidate_tfv_m3,
            truth_anchor_tfv_m3,
            predicted_candidate_delta_tfv_m3,
            predicted_anchor_delta_tfv_m3,
        )
    ]
    n = arrays[0].size
    if n == 0 or any(x.size != n for x in arrays):
        raise ValueError("V125 calibration arrays must be non-empty and aligned")
    rain = tuple(str(x) for x in rainfall_groups)
    ids = tuple(str(x) for x in row_ids)
    if len(rain) != n or len(ids) != n or len(set(ids)) != n:
        raise ValueError("V125 calibration rainfall/row identities must be aligned and unique")
    stacked = np.column_stack(arrays)
    if not np.isfinite(stacked).all():
        raise ValueError("V125 calibration contains non-finite values")

    truth_adv = arrays[0] - arrays[1]
    pred_adv = arrays[2] - arrays[3]
    residual = truth_adv - pred_adv
    margin = max(0.0, _higher_quantile(residual, q))
    identity = "\n".join(f"{rid}|{rg}" for rid, rg in sorted(zip(ids, rain, strict=True)))
    result = AnchorOverrideCalibrationV125(
        quantile=q,
        margin_m3=margin,
        sample_count=n,
        rainfall_groups=tuple(sorted(set(rain))),
        row_identity_sha256=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
    )
    result.validate()
    return result


def anchor_override_audit_v125(
    *,
    truth_advantage_m3: Iterable[float],
    predicted_advantage_m3: Iterable[float],
    margin_m3: float,
) -> dict[str, float | int]:
    truth = np.asarray(list(truth_advantage_m3), dtype=np.float64)
    pred = np.asarray(list(predicted_advantage_m3), dtype=np.float64)
    if truth.size == 0 or truth.shape != pred.shape or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("V125 audit advantage arrays must be finite, non-empty and aligned")
    margin = float(margin_m3)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("V125 audit margin must be finite and non-negative")
    admitted = pred + margin < 0.0
    beneficial = truth < 0.0
    tp = int(np.sum(admitted & beneficial))
    fp = int(np.sum(admitted & ~beneficial))
    fn = int(np.sum(~admitted & beneficial))
    selected = int(np.sum(admitted))
    return {
        "count": int(truth.size),
        "admitted_count": selected,
        "beneficial_count": int(np.sum(beneficial)),
        "false_benefit_count": fp,
        "false_benefit_rate_among_admitted": float(fp / selected) if selected else 0.0,
        "beneficial_override_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "beneficial_override_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "mean_truth_advantage_admitted_m3": float(np.mean(truth[admitted])) if selected else 0.0,
        "worst_truth_advantage_admitted_m3": float(np.max(truth[admitted])) if selected else 0.0,
    }


def calibration_json_v125(calibration: AnchorOverrideCalibrationV125) -> str:
    return json.dumps({"calibration": calibration.as_dict()}, indent=2, sort_keys=True) + "\n"


__all__ = [
    "AnchorOverrideCalibrationV125",
    "anchor_override_audit_v125",
    "calibrate_anchor_override_margin_v125",
    "calibration_json_v125",
]
