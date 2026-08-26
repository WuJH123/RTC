"""Development-only exact policy-return calibration contract for Project7 V25R2.

V25R2 deliberately reuses the deterministic group-disjoint ridge fitting implementation from V25,
but changes the supervised estimand from the H120 diagnostic window to the exact system-wide TFV
policy return already stored by the matched counterfactual branches.  No new SWMM truth is required.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from .direct_tfv_v25_value_calibration import (
    FittedV25ValueCalibrator,
    V25ValueCalibratorModule,
    fit_v25_value_calibrator,
    value_metrics,
)

V25R2_VALUE_CALIBRATOR_CONTRACT = (
    "PROJECT7_STEP3_V25R2_MATCHED_EXACT_POLICY_RETURN_CALIBRATION_V1"
)
V25R2_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_V25R2_EXACT_POLICY_RETURN_CALIBRATOR_CHECKPOINT_V1"
)
V25R2_VALUE_FEATURE_CONTRACT = (
    "PROJECT7_STEP3_V25R2_SELECTED_V23_PORTFOLIO_V15_RANK_FEATURE_V1"
)
V25R2_TRUTH_FIELD = "true_policy_return_delta_tfv_m3"
V25R2_ESTIMAND = (
    "SYSTEM_WIDE_TFV_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def fitted_payload_v25r2(
    fitted: FittedV25ValueCalibrator,
    *,
    lineage: Mapping[str, Any],
    oof_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": V25R2_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT,
        "calibrator_contract": V25R2_VALUE_CALIBRATOR_CONTRACT,
        "feature_contract": V25R2_VALUE_FEATURE_CONTRACT,
        "truth_field": V25R2_TRUTH_FIELD,
        "estimand": V25R2_ESTIMAND,
        "development_only": True,
        "formal_evidence": False,
        "feature_mean": fitted.feature_mean.tolist(),
        "feature_scale": fitted.feature_scale.tolist(),
        "weight": fitted.weight.tolist(),
        "intercept": float(fitted.intercept),
        "one_sided_error_margin_m3": float(fitted.one_sided_error_margin_m3),
        "ridge": float(fitted.ridge),
        "error_quantile": float(fitted.error_quantile),
        "oof_metrics": dict(oof_metrics),
        "lineage": dict(lineage),
    }


def load_v25r2_value_calibrator(
    path: str,
    *,
    device: torch.device,
    expected_lineage: Mapping[str, Any],
) -> tuple[V25ValueCalibratorModule, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != V25R2_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT:
        raise ValueError("V25R2 runtime requires the exact policy-return calibrator checkpoint")
    if payload.get("development_only") is not True or payload.get("formal_evidence") is not False:
        raise ValueError("V25R2 calibrator must remain Development-only")
    if payload.get("truth_field") != V25R2_TRUTH_FIELD or payload.get("estimand") != V25R2_ESTIMAND:
        raise ValueError("V25R2 calibrator estimand drifted")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V25R2 calibrator lacks lineage")
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(f"V25R2 calibrator lineage mismatch: {key}")
    mean = np.asarray(payload["feature_mean"], dtype=np.float64).reshape(-1)
    scale = np.maximum(np.asarray(payload["feature_scale"], dtype=np.float64).reshape(-1), 1.0e-6)
    weight = np.asarray(payload["weight"], dtype=np.float64).reshape(-1)
    if mean.size == 0 or mean.size != scale.size or mean.size != weight.size:
        raise ValueError("V25R2 calibrator feature width mismatch")
    margin = float(payload["one_sided_error_margin_m3"])
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError("V25R2 calibrator error margin is invalid")
    fitted = FittedV25ValueCalibrator(
        feature_mean=mean,
        feature_scale=scale,
        weight=weight,
        intercept=float(payload["intercept"]),
        one_sided_error_margin_m3=margin,
        ridge=float(payload["ridge"]),
        error_quantile=float(payload["error_quantile"]),
    )
    model = V25ValueCalibratorModule(fitted).to(device)
    model.eval()
    return model, payload


__all__ = [
    "V25R2_ESTIMAND",
    "V25R2_TRUTH_FIELD",
    "V25R2_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT",
    "V25R2_VALUE_CALIBRATOR_CONTRACT",
    "V25R2_VALUE_FEATURE_CONTRACT",
    "fit_v25_value_calibrator",
    "fitted_payload_v25r2",
    "load_v25r2_value_calibrator",
    "value_metrics",
]
