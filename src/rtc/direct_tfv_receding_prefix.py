"""Receding-consistent executable-prefix admission for Project7 Direct-TFV MPC.

The online controller replans every 10 minutes, so the H120/H360 plan produced by Step3 is guidance,
not a commitment. Development V9 showed that full H360 optimizer plans can be authoritatively
beneficial while the resulting 10-minute receding closed loop is still worse than No-control. This
module therefore calibrates the error of the action that is actually committed before the next
observation: execute the optimized first 10-minute block, then HOLD the current active target for the
remaining H350 evaluation window.

The raw V6 optimizer, q95 joint-sequence support, Step2 weights and TFV-only objective are unchanged.
The independent conformal unit remains the fresh Development rainfall group. The previous V2
full-plan policy margin is retained as diagnostic evidence but is not a floor on the executable-prefix
margin; otherwise the estimand mismatch would be preserved by construction.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from .direct_tfv_admission import (
    DIRECT_TFV_ADMISSION_COVERAGE,
    _minimum_conformal_sample_size,
    _one_sided_conformal_upper,
)
from .direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT


DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT = (
    "PROJECT7_DIRECT_TFV_EXECUTE_H10_THEN_HOLD_H350_CALIBRATION_PANEL_V1"
)
DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RECEDING_PREFIX_ONE_SIDED_ADMISSION_V1"
)
DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6"
)
DIRECT_TFV_RECEDING_PREFIX_EXECUTION_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V8"
)
DIRECT_TFV_RECEDING_PREFIX_SEMANTICS = "EXECUTE_H10_THEN_HOLD_ACTIVE_TARGET_H350"


def executable_prefix_sequence(
    candidate: torch.Tensor,
    active_target: torch.Tensor,
    *,
    control_block_steps: int,
) -> torch.Tensor:
    """Return the exact online commitment estimand: first block from candidate, then HOLD."""

    if candidate.ndim != 2 or candidate.shape[1] != 109:
        raise ValueError("Direct-TFV executable-prefix candidate must be [H,109]")
    if tuple(active_target.shape) != (109,):
        raise ValueError("Direct-TFV executable-prefix active target must contain 109 settings")
    steps = int(control_block_steps)
    if steps <= 0 or steps > int(candidate.shape[0]):
        raise ValueError("invalid Direct-TFV executable-prefix control block length")
    hold = active_target[None].expand_as(candidate).clone()
    hold[:steps] = candidate[:steps]
    return hold


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def derive_receding_prefix_admission(
    *,
    base_policy_admission: Mapping[str, Any],
    panel_contract: str,
    panel_step3_contract: str,
    panel_records: Sequence[Mapping[str, Any]],
    expected_rainfall_groups: Sequence[str],
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
) -> dict[str, Any]:
    """Calibrate one-sided error for the first 10-minute executable action estimand."""

    if str(base_policy_admission.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("receding-prefix admission requires the accepted V2 policy admission")
    if base_policy_admission.get("development_only") is not True:
        raise ValueError("receding-prefix base admission must be Development-only")
    if str(panel_contract) != DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT:
        raise ValueError("receding-prefix calibration panel has the wrong contract")
    if str(panel_step3_contract) != DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT:
        raise ValueError("receding-prefix panel was not generated from the frozen V6 raw optimizer")
    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("receding-prefix admission coverage must lie in (0.5,1)")

    expected = {str(value) for value in expected_rainfall_groups if str(value)}
    minimum = _minimum_conformal_sample_size(float(coverage))
    if len(expected) < minimum:
        raise ValueError(
            f"{float(coverage):.3f} receding-prefix calibration needs at least {minimum} rainfall "
            f"groups; got {len(expected)}"
        )

    density_floor = int(base_policy_admission.get("density_floor_changed_facilities", 2))
    by_group: dict[str, list[float]] = defaultdict(list)
    dense_by_group: dict[str, list[float]] = defaultdict(list)
    plan_hashes: set[str] = set()
    for row in panel_records:
        group = str(row.get("rainfall_group", ""))
        if group not in expected:
            raise ValueError(f"receding-prefix panel contains non-calibration rainfall group: {group!r}")
        plan_sha = str(row.get("plan_sha256", ""))
        if not plan_sha:
            raise ValueError("receding-prefix panel record lacks plan_sha256")
        if plan_sha in plan_hashes:
            raise ValueError("receding-prefix panel repeats a plan SHA")
        plan_hashes.add(plan_sha)
        predicted = _finite(
            row.get("predicted_prefix_delta_tfv_m3"), label="predicted executable-prefix delta TFV"
        )
        truth = _finite(
            row.get("true_prefix_delta_tfv_m3"), label="true executable-prefix delta TFV"
        )
        residual = truth - predicted
        by_group[group].append(residual)
        changed = int(
            row.get(
                "prefix_changed_facility_count",
                row.get("first_move_changed_facility_count", row.get("active_facility_count", -1)),
            )
        )
        if changed < 0:
            raise ValueError("receding-prefix panel lacks first-move changed-facility count")
        if changed >= density_floor:
            dense_by_group[group].append(residual)

    if set(by_group) != expected:
        raise ValueError(
            "receding-prefix calibration must cover every fresh rainfall group; "
            f"missing={sorted(expected - set(by_group))}, extra={sorted(set(by_group) - expected)}"
        )
    group_max = {group: max(values) for group, values in by_group.items() if values}
    dense_group_max = {group: max(values) for group, values in dense_by_group.items() if values}
    prefix_q = _one_sided_conformal_upper(list(group_max.values()), float(coverage))
    dense_used = len(dense_group_max) >= minimum
    prefix_dense_q = (
        _one_sided_conformal_upper(list(dense_group_max.values()), float(coverage))
        if dense_used
        else prefix_q
    )

    generic_q = _finite(
        base_policy_admission.get("d3_rainfall_group_residual_conformal_upper_m3"),
        label="generic D3 conformal upper",
    )
    generic_dense_q = _finite(
        base_policy_admission.get("d3_dense_rainfall_group_residual_conformal_upper_m3", generic_q),
        label="generic dense D3 conformal upper",
    )
    global_margin = max(0.0, generic_q, prefix_q)
    dense_margin = max(global_margin, generic_dense_q, prefix_dense_q)
    full_global = _finite(
        base_policy_admission.get("global_margin_m3"), label="V2 full-plan margin"
    )
    full_dense = _finite(
        base_policy_admission.get("dense_margin_m3"), label="V2 dense full-plan margin"
    )

    return {
        "contract": DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT,
        "development_only": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "execution_estimand": DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
        "coverage": float(coverage),
        "independent_unit": "RAINFALL_GROUP_MAX_TRUE_MINUS_PREDICTED_RESIDUAL",
        "residual_definition": "true_prefix_delta_tfv_m3_minus_predicted_prefix_delta_tfv_m3",
        "admission_rule": "prefix_delta_tfv_m3 + receding_prefix_margin_m3 < 0",
        "full_plan_requirement": "raw_full_plan_predicted_delta_tfv_m3 < 0",
        "density_floor_changed_facilities": density_floor,
        "density_classification_variable": "FIRST_MOVE_CHANGED_FACILITY_COUNT",
        "global_margin_m3": float(global_margin),
        "dense_margin_m3": float(dense_margin),
        "generic_d3_conformal_upper_m3": float(generic_q),
        "generic_d3_dense_conformal_upper_m3": float(generic_dense_q),
        "receding_prefix_conformal_upper_m3": float(prefix_q),
        "receding_prefix_dense_conformal_upper_m3": float(prefix_dense_q),
        "receding_prefix_dense_conformal_used": bool(dense_used),
        "receding_prefix_calibration_rainfall_group_count": len(group_max),
        "receding_prefix_dense_calibration_rainfall_group_count": len(dense_group_max),
        "receding_prefix_calibration_rainfall_groups": sorted(group_max),
        "receding_prefix_calibration_plan_count": len(panel_records),
        "policy_query_step3_contract": DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
        "execution_step3_contract": DIRECT_TFV_RECEDING_PREFIX_EXECUTION_STEP3_CONTRACT,
        "raw_optimizer_query_distribution_unchanged": True,
        "v2_full_plan_global_margin_m3": float(full_global),
        "v2_full_plan_dense_margin_m3": float(full_dense),
        "full_plan_policy_margin_controls_execution": False,
        "online_swmm_called": False,
        "scientific_role": (
            "V8 keeps the frozen V6 raw optimizer and q95 support, but admission is calibrated on "
            "the only action committed before replanning: optimized H10 followed by HOLD H350. "
            "Dense/global prefix uncertainty is keyed to facilities actually changed in that first "
            "H10 block, not the larger active optimization set. The H120/H360 plan remains search "
            "guidance and diagnostic evidence, not executable benefit credited to the current action."
        ),
    }


def receding_prefix_margin_m3(
    calibration: Mapping[str, Any], changed_facility_count: int
) -> tuple[float, str]:
    if str(calibration.get("contract", "")) != DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT:
        raise ValueError("Direct-TFV V8 received the wrong receding-prefix admission contract")
    density_floor = int(calibration.get("density_floor_changed_facilities", 2))
    if int(changed_facility_count) >= density_floor:
        return float(calibration["dense_margin_m3"]), "receding_prefix_dense"
    return float(calibration["global_margin_m3"]), "receding_prefix_global"


__all__ = [
    "DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT",
    "DIRECT_TFV_RECEDING_PREFIX_EXECUTION_STEP3_CONTRACT",
    "DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT",
    "DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT",
    "DIRECT_TFV_RECEDING_PREFIX_SEMANTICS",
    "derive_receding_prefix_admission",
    "executable_prefix_sequence",
    "receding_prefix_margin_m3",
]
