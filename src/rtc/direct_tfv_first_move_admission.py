"""Optimizer-matched one-sided admission for an exact refined executable first move."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .direct_tfv_admission import DIRECT_TFV_ADMISSION_COVERAGE, _one_sided_conformal_upper
from .direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS


DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT = (
    "PROJECT7_DIRECT_TFV_REFINED_FIRST_MOVE_CALIBRATION_PANEL_V1"
)
DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_REFINED_FIRST_MOVE_NORMALIZED_ONE_SIDED_ADMISSION_V1"
)
DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V9"
)
DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V10_CAUSAL_RAINFALL_SCENARIO_MEAN"
)
DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS = 24
DIRECT_TFV_FIRST_MOVE_SCALE = "SQRT_FIRST_MOVE_CHANGED_FACILITY_COUNT"
_ALLOWED_QUERY_CONTRACTS = {
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT,
}


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _scale(changed_facility_count: int) -> float:
    count = int(changed_facility_count)
    if count <= 0 or count > 109:
        raise ValueError("first-move normalized admission requires changed facility count in [1,109]")
    return math.sqrt(float(count))


def derive_first_move_admission(
    *,
    panel_contract: str,
    panel_step3_contract: str,
    panel_records: Sequence[Mapping[str, Any]],
    expected_rainfall_groups: Sequence[str],
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
    rainfall_scenario_contract: str | None = None,
) -> dict[str, Any]:
    """Derive a rainfall-group normalized margin for the exact query distribution supplied."""
    if str(panel_contract) != DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT:
        raise ValueError("first-move calibration panel has the wrong contract")
    query_contract = str(panel_step3_contract)
    if query_contract not in _ALLOWED_QUERY_CONTRACTS:
        raise ValueError("first-move calibration panel has an unsupported optimizer contract")
    if query_contract == DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT:
        if not rainfall_scenario_contract:
            raise ValueError("scenario-mean first-move admission requires rainfall scenario lineage")
    elif rainfall_scenario_contract is not None:
        raise ValueError("single-scenario V11 admission must not claim a scenario-mean contract")
    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("first-move admission coverage must lie in (0.5,1)")
    expected = {str(value) for value in expected_rainfall_groups if str(value)}
    if len(expected) < DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS:
        raise ValueError(
            "refined first-move admission requires at least "
            f"{DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS} fresh rainfall groups; got {len(expected)}"
        )

    by_group: dict[str, list[float]] = defaultdict(list)
    plan_keys: set[tuple[str, str]] = set()
    changed_counts: list[int] = []
    raw_residuals: list[float] = []
    for row in panel_records:
        group = str(row.get("rainfall_group", ""))
        if group not in expected:
            raise ValueError(f"first-move panel contains non-calibration rainfall group: {group!r}")
        plan_sha = str(row.get("plan_sha256", ""))
        key = (group, plan_sha)
        if not plan_sha or key in plan_keys:
            raise ValueError("first-move panel requires unique (rainfall_group, plan_sha) records")
        plan_keys.add(key)
        predicted = _finite(
            row.get("predicted_refined_delta_tfv_m3"),
            label="predicted refined first-move delta TFV",
        )
        truth = _finite(
            row.get("true_refined_delta_tfv_m3"),
            label="true refined first-move delta TFV",
        )
        changed = int(row.get("first_move_changed_facility_count", -1))
        residual = truth - predicted
        by_group[group].append(residual / _scale(changed))
        raw_residuals.append(residual)
        changed_counts.append(changed)

    if set(by_group) != expected:
        raise ValueError(
            "first-move calibration must cover every expected rainfall group; "
            f"missing={sorted(expected-set(by_group))}, extra={sorted(set(by_group)-expected)}"
        )
    group_max_normalized = {group: max(values) for group, values in by_group.items() if values}
    normalized_q = max(
        0.0,
        _one_sided_conformal_upper(list(group_max_normalized.values()), float(coverage)),
    )
    ordered = sorted(float(value) for value in group_max_normalized.values())
    raw_ordered = sorted(float(value) for value in raw_residuals)
    payload: dict[str, Any] = {
        "contract": DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
        "development_only": True,
        "execution_estimand": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
        "coverage": float(coverage),
        "independent_unit": "RAINFALL_GROUP_MAX_NORMALIZED_TRUE_MINUS_PREDICTED_RESIDUAL",
        "residual_definition": "true_refined_delta_tfv_m3_minus_predicted_refined_delta_tfv_m3",
        "normalization": DIRECT_TFV_FIRST_MOVE_SCALE,
        "normalized_residual_conformal_upper": float(normalized_q),
        "admission_rule": (
            "predicted_refined_delta_tfv_m3 + normalized_q*sqrt(first_move_changed_facility_count) < 0"
        ),
        "calibration_rainfall_group_count": len(group_max_normalized),
        "calibration_rainfall_groups": sorted(group_max_normalized),
        "calibration_plan_count": len(panel_records),
        "query_step3_contract": query_contract,
        "generic_d3_floor_controls_execution": False,
        "v9_full_plan_margin_controls_execution": False,
        "v10_prefix_margin_controls_execution": False,
        "minimum_calibration_rainfall_groups": DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS,
        "normalized_group_residual_min": float(ordered[0]),
        "normalized_group_residual_median": float(ordered[len(ordered) // 2]),
        "normalized_group_residual_max": float(ordered[-1]),
        "raw_residual_min_m3": float(raw_ordered[0]),
        "raw_residual_median_m3": float(raw_ordered[len(raw_ordered) // 2]),
        "raw_residual_max_m3": float(raw_ordered[-1]),
        "changed_facility_count_min": min(changed_counts, default=0),
        "changed_facility_count_max": max(changed_counts, default=0),
        "online_swmm_called": False,
        "coverage_claim_scope": (
            "Marginal rainfall-group split-conformal claim for the fixed exact-query normalized "
            "score; no conditional-coverage claim by changed-facility count is made."
        ),
        "scientific_role": (
            "Execution uncertainty is calibrated only on fresh authoritative SWMM labels of the "
            "same refined first-move query distribution used online."
        ),
    }
    if rainfall_scenario_contract is not None:
        payload["rainfall_scenario_contract"] = str(rainfall_scenario_contract)
    return payload


def first_move_margin_m3(
    calibration: Mapping[str, Any], changed_facility_count: int
) -> float:
    if str(calibration.get("contract", "")) != DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT:
        raise ValueError("Direct-TFV refined first move received the wrong admission contract")
    if str(calibration.get("normalization", "")) != DIRECT_TFV_FIRST_MOVE_SCALE:
        raise ValueError("Direct-TFV refined first move received the wrong residual normalization")
    q = _finite(
        calibration.get("normalized_residual_conformal_upper"),
        label="normalized first-move conformal upper",
    )
    if q < 0.0:
        raise ValueError("normalized first-move conformal upper must be non-negative")
    return float(q * _scale(int(changed_facility_count)))


__all__ = [
    "DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT",
    "DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS",
    "DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT",
    "DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT",
    "DIRECT_TFV_FIRST_MOVE_SCENARIO_MEAN_QUERY_STEP3_CONTRACT",
    "DIRECT_TFV_FIRST_MOVE_SCALE",
    "derive_first_move_admission",
    "first_move_margin_m3",
]
