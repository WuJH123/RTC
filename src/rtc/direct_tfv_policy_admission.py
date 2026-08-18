"""Policy-matched one-sided admission for optimizer-consistent Direct-TFV MPC.

V6 changed the optimizer query distribution by contracting every H120 sequence to the
D3-HOLD q95 joint temporal support.  A residual maximum measured from an older, pre-contraction
optimizer is therefore useful historical evidence, but it is not the right object to control the
current policy margin.  This module calibrates the admission margin on authoritative SWMM labels
of the *current raw optimizer queries* while retaining the fresh D3 rainfall-group conformal bound
as a floor.

The independent conformal unit is still the rainfall group.  The policy panel must use the same
fresh Development calibration rainfall groups already reserved for admission, never Step2 TrainFit,
T10/T20 post-calibration probes, Validation, Final, Formal or Policy Lock.  No online SWMM call is
introduced.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .direct_tfv_admission import (
    DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
    DIRECT_TFV_ADMISSION_COVERAGE,
    _minimum_conformal_sample_size,
    _one_sided_conformal_upper,
)


DIRECT_TFV_POLICY_ADMISSION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_MATCHED_ONE_SIDED_ADMISSION_V2"
)
DIRECT_TFV_POLICY_PANEL_CONTRACT = (
    "PROJECT7_DIRECT_TFV_V6_RAW_OPTIMIZER_CALIBRATION_PANEL_V1"
)
DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6"
)
DIRECT_TFV_POLICY_EXECUTION_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V7"
)


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def derive_policy_matched_admission(
    *,
    base_admission: Mapping[str, Any],
    panel_contract: str,
    panel_step3_contract: str,
    panel_records: Sequence[Mapping[str, Any]],
    expected_rainfall_groups: Sequence[str],
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
) -> dict[str, Any]:
    """Derive a V2 margin from current-policy optimizer residuals.

    Each rainfall group contributes its maximum ``true - predicted`` residual.  The final margin is
    the maximum of the fresh generic-D3 conformal bound and the current-policy optimizer conformal
    bound.  The historical pre-V6 optimizer maximum is retained as diagnostic metadata only and is
    deliberately not a floor on the current policy margin.
    """

    if str(base_admission.get("contract", "")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
        raise ValueError("policy admission requires the accepted V1 fresh-D3 admission artifact")
    if base_admission.get("development_only") is not True:
        raise ValueError("policy admission base artifact must be Development-only")
    if str(panel_contract) != DIRECT_TFV_POLICY_PANEL_CONTRACT:
        raise ValueError("policy admission panel has the wrong contract")
    if str(panel_step3_contract) != DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT:
        raise ValueError("policy admission must calibrate the current V6 raw optimizer query contract")
    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("policy admission coverage must lie in (0.5,1)")

    expected = {str(value) for value in expected_rainfall_groups if str(value)}
    if not expected:
        raise ValueError("policy admission has no expected fresh calibration rainfall groups")
    minimum = _minimum_conformal_sample_size(float(coverage))
    if len(expected) < minimum:
        raise ValueError(
            f"{float(coverage):.3f} policy calibration needs at least {minimum} rainfall groups; "
            f"got {len(expected)}"
        )

    density_floor = int(base_admission.get("density_floor_changed_facilities", 2))
    by_group: dict[str, list[float]] = defaultdict(list)
    dense_by_group: dict[str, list[float]] = defaultdict(list)
    plan_hashes: set[str] = set()
    for row in panel_records:
        group = str(row.get("rainfall_group", ""))
        if group not in expected:
            raise ValueError(f"policy panel contains non-calibration rainfall group: {group!r}")
        plan_sha = str(row.get("plan_sha256", ""))
        if not plan_sha:
            raise ValueError("policy panel record lacks plan_sha256")
        if plan_sha in plan_hashes:
            raise ValueError("policy panel repeats a plan SHA")
        plan_hashes.add(plan_sha)
        predicted = _finite(row.get("predicted_delta_tfv_m3"), label="policy predicted delta TFV")
        truth = _finite(row.get("true_delta_tfv_m3"), label="policy true delta TFV")
        residual = truth - predicted
        by_group[group].append(residual)
        if int(row.get("active_facility_count", -1)) >= density_floor:
            dense_by_group[group].append(residual)

    actual = set(by_group)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "policy calibration must cover every fresh rainfall group exactly as a calibration role; "
            f"missing={missing}, extra={extra}"
        )

    group_max = {group: max(values) for group, values in by_group.items() if values}
    policy_q = _one_sided_conformal_upper(list(group_max.values()), float(coverage))
    dense_group_max = {group: max(values) for group, values in dense_by_group.items() if values}
    dense_used = len(dense_group_max) >= minimum
    policy_dense_q = (
        _one_sided_conformal_upper(list(dense_group_max.values()), float(coverage))
        if dense_used
        else policy_q
    )

    d3_q = _finite(
        base_admission.get("d3_rainfall_group_residual_conformal_upper_m3"),
        label="base D3 conformal upper",
    )
    d3_dense_q = _finite(
        base_admission.get("d3_dense_rainfall_group_residual_conformal_upper_m3", d3_q),
        label="base dense D3 conformal upper",
    )
    global_margin = max(0.0, d3_q, policy_q)
    dense_margin = max(global_margin, d3_dense_q, policy_dense_q)

    legacy_optimizer_max = _finite(
        base_admission.get("optimizer_replay_residual_max_m3", 0.0),
        label="legacy optimizer residual maximum",
    )
    legacy_optimizer_dense_max = _finite(
        base_admission.get("optimizer_replay_dense_residual_max_m3", legacy_optimizer_max),
        label="legacy dense optimizer residual maximum",
    )

    return {
        "contract": DIRECT_TFV_POLICY_ADMISSION_CONTRACT,
        "development_only": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "d3_conformal_coverage": float(coverage),
        "d3_conformal_unit": "RAINFALL_GROUP_MAX_TRUE_MINUS_PREDICTED_RESIDUAL",
        "policy_conformal_unit": "RAINFALL_GROUP_MAX_CURRENT_OPTIMIZER_TRUE_MINUS_PREDICTED_RESIDUAL",
        "residual_definition": "true_delta_tfv_m3_minus_predicted_delta_tfv_m3",
        "admission_rule": "predicted_delta_tfv_m3 + policy_matched_one_sided_margin_m3 < 0",
        "density_floor_changed_facilities": density_floor,
        "global_margin_m3": float(global_margin),
        "dense_margin_m3": float(dense_margin),
        "d3_rainfall_group_residual_conformal_upper_m3": float(d3_q),
        "d3_dense_rainfall_group_residual_conformal_upper_m3": float(d3_dense_q),
        "policy_optimizer_residual_conformal_upper_m3": float(policy_q),
        "policy_optimizer_dense_residual_conformal_upper_m3": float(policy_dense_q),
        "policy_dense_conformal_used": bool(dense_used),
        "policy_calibration_rainfall_group_count": len(group_max),
        "policy_dense_calibration_rainfall_group_count": len(dense_group_max),
        "policy_calibration_rainfall_groups": sorted(group_max),
        "policy_calibration_plan_count": len(panel_records),
        "policy_query_step3_contract": DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
        "execution_step3_contract": DIRECT_TFV_POLICY_EXECUTION_STEP3_CONTRACT,
        "raw_optimizer_query_distribution_unchanged_between_v6_and_v7": True,
        "legacy_optimizer_replay_residual_max_m3": float(legacy_optimizer_max),
        "legacy_optimizer_replay_dense_residual_max_m3": float(legacy_optimizer_dense_max),
        "legacy_optimizer_replay_controls_current_margin": False,
        "optimizer_replay_coverage_claimed": False,
        "policy_conformal_coverage_claimed": True,
        "scientific_role": (
            "V7 keeps the V6 q95-supported raw optimizer unchanged but calibrates admission on "
            "authoritative SWMM residuals from that current optimizer query distribution. The fresh "
            "generic-D3 conformal bound remains a floor; obsolete pre-V6 optimizer extrema remain "
            "historical diagnostics only."
        ),
    }


def policy_admission_margin_m3(
    calibration: Mapping[str, Any], active_facility_count: int
) -> tuple[float, str]:
    if str(calibration.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("Direct-TFV V7 received the wrong policy-matched admission contract")
    density_floor = int(calibration.get("density_floor_changed_facilities", 2))
    if int(active_facility_count) >= density_floor:
        return float(calibration["dense_margin_m3"]), "policy_dense"
    return float(calibration["global_margin_m3"]), "policy_global"


__all__ = [
    "DIRECT_TFV_POLICY_ADMISSION_CONTRACT",
    "DIRECT_TFV_POLICY_EXECUTION_STEP3_CONTRACT",
    "DIRECT_TFV_POLICY_PANEL_CONTRACT",
    "DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT",
    "derive_policy_matched_admission",
    "policy_admission_margin_m3",
]
