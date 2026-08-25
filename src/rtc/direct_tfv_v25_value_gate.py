"""Fail-closed H120 TFV value admission for the Development-only V25 policy.

The module is deliberately small and policy-independent.  Stress and blend are retained as
diagnostics, but no hydraulic heuristic is allowed to turn an unavailable or non-beneficial value
prediction into an admitted action.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


V25_VALUE_GATE_CONTRACT = "PROJECT7_STEP3_V25_CONSERVATIVE_TFV_VALUE_GATE_V1"


@dataclass(frozen=True)
class V25ValueGateResult:
    """Auditable result of one selected-candidate ACTION/HOLD value decision."""

    action: bool
    reason: str
    tfv_value_available: bool
    tfv_value_prediction_m3: float | None
    tfv_value_upper_bound_m3: float | None
    tfv_value_admission_passed: bool
    engineering_feasible: bool
    passive_channels_unchanged: bool
    sequence_support_valid: bool
    candidate_source: str
    network_stress_q75: float | None
    strong_storm_blend: float | None
    contract: str = V25_VALUE_GATE_CONTRACT


def validate_v25_lineage(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    """Return whether every required V25 lineage field matches exactly.

    Missing fields and non-string values fail closed.  The caller supplies the expected hashes and
    contracts from the frozen asset/truth manifest; this function does not discover or relax them.
    """

    if not expected:
        return False
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, str):
            if not isinstance(actual_value, str) or actual_value.lower() != expected_value.lower():
                return False
        elif actual_value != expected_value:
            return False
    return True


def evaluate_v25_value_gate(
    *,
    candidate_source: str,
    predicted_delta_tfv_m3: float | None,
    one_sided_error_margin_m3: float | None,
    tfv_value_available: bool,
    calibration_available: bool,
    lineage_valid: bool,
    engineering_feasible: bool,
    passive_channels_unchanged: bool,
    sequence_support_valid: bool,
    network_stress_q75: float | None = None,
    strong_storm_blend: float | None = None,
) -> V25ValueGateResult:
    """Apply the strict V25 H120 value contract.

    The conservative upper bound is ``prediction + non-negative error margin``.  ACTION is
    admitted only when that bound is strictly negative.  In particular, a zero bound is HOLD and
    hydraulic stress cannot override the value result.
    """

    prediction = (
        float(predicted_delta_tfv_m3)
        if predicted_delta_tfv_m3 is not None
        else None
    )
    margin = (
        float(one_sided_error_margin_m3)
        if one_sided_error_margin_m3 is not None
        else None
    )

    if not lineage_valid:
        return _hold(
            reason="LINEAGE_INVALID",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=engineering_feasible,
            passive_channels_unchanged=passive_channels_unchanged,
            sequence_support_valid=sequence_support_valid,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )
    if not tfv_value_available or not calibration_available:
        return _hold(
            reason="TFV_VALUE_UNAVAILABLE",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=engineering_feasible,
            passive_channels_unchanged=passive_channels_unchanged,
            sequence_support_valid=sequence_support_valid,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )
    if not engineering_feasible:
        return _hold(
            reason="ENGINEERING_INFEASIBLE",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=False,
            passive_channels_unchanged=passive_channels_unchanged,
            sequence_support_valid=sequence_support_valid,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )
    if not passive_channels_unchanged:
        return _hold(
            reason="PASSIVE_CHANNEL_VIOLATION",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=engineering_feasible,
            passive_channels_unchanged=False,
            sequence_support_valid=sequence_support_valid,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )
    if not sequence_support_valid:
        return _hold(
            reason="SEQUENCE_SUPPORT_VIOLATION",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=engineering_feasible,
            passive_channels_unchanged=passive_channels_unchanged,
            sequence_support_valid=False,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )
    if (
        prediction is None
        or margin is None
        or not math.isfinite(prediction)
        or not math.isfinite(margin)
        or margin < 0.0
    ):
        return _hold(
            reason="TFV_VALUE_INVALID",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=engineering_feasible,
            passive_channels_unchanged=passive_channels_unchanged,
            sequence_support_valid=sequence_support_valid,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )

    upper_bound = prediction + margin
    if not math.isfinite(upper_bound):
        return _hold(
            reason="TFV_VALUE_INVALID",
            candidate_source=candidate_source,
            prediction=prediction,
            upper_bound=None,
            tfv_value_available=tfv_value_available,
            engineering_feasible=engineering_feasible,
            passive_channels_unchanged=passive_channels_unchanged,
            sequence_support_valid=sequence_support_valid,
            network_stress_q75=network_stress_q75,
            strong_storm_blend=strong_storm_blend,
        )

    admitted = upper_bound < 0.0
    return V25ValueGateResult(
        action=admitted,
        reason="TFV_UCB_NEGATIVE" if admitted else "TFV_UCB_NONNEGATIVE",
        tfv_value_available=True,
        tfv_value_prediction_m3=prediction,
        tfv_value_upper_bound_m3=upper_bound,
        tfv_value_admission_passed=admitted,
        engineering_feasible=True,
        passive_channels_unchanged=True,
        sequence_support_valid=True,
        candidate_source=str(candidate_source),
        network_stress_q75=network_stress_q75,
        strong_storm_blend=strong_storm_blend,
    )


def _hold(
    *,
    reason: str,
    candidate_source: str,
    prediction: float | None,
    upper_bound: float | None,
    tfv_value_available: bool,
    engineering_feasible: bool,
    passive_channels_unchanged: bool,
    sequence_support_valid: bool,
    network_stress_q75: float | None,
    strong_storm_blend: float | None,
) -> V25ValueGateResult:
    return V25ValueGateResult(
        action=False,
        reason=reason,
        tfv_value_available=bool(tfv_value_available),
        tfv_value_prediction_m3=prediction,
        tfv_value_upper_bound_m3=upper_bound,
        tfv_value_admission_passed=False,
        engineering_feasible=bool(engineering_feasible),
        passive_channels_unchanged=bool(passive_channels_unchanged),
        sequence_support_valid=bool(sequence_support_valid),
        candidate_source=str(candidate_source),
        network_stress_q75=network_stress_q75,
        strong_storm_blend=strong_storm_blend,
    )


__all__ = [
    "V25_VALUE_GATE_CONTRACT",
    "V25ValueGateResult",
    "evaluate_v25_value_gate",
    "validate_v25_lineage",
]
