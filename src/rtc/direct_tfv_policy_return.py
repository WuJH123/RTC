"""Receding-policy first-action value contracts for Project7 Direct-TFV control.

The historical Direct-TFV Step2 predicts an open-loop/no-further-command H360 value difference.
That remains useful for generating a coordinated H120 direction, but Development V12 evidence shows
that a locally correct H360 first-move sign is not sufficient to guarantee lower event-level TFV
under repeated 10-minute replanning. This module defines the deployed-policy estimand:

    A^pi(x_t, u_t) = J(x_t, u_t for H10, then frozen pi)
                     - J(x_t, HOLD for H10, then the same frozen pi)

Both counterfactual branches must use the same frozen continuation policy after t+10 min and must
start from an identical authoritative SWMM prefix. SWMM remains offline truth only.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .direct_tfv_admission import DIRECT_TFV_ADMISSION_COVERAGE, _one_sided_conformal_upper
from .step2_tfv_value import DirectFacilityTFVValueModel, DirectTFVValueDesign
from .step2_train_response_v60 import InputNormalizationV60


DIRECT_TFV_POLICY_RETURN_ESTIMAND = (
    "EXECUTE_CANDIDATE_H10_THEN_FROZEN_POLICY_VS_HOLD_H10_THEN_SAME_FROZEN_POLICY"
)
DIRECT_TFV_POLICY_RETURN_CHECKPOINT_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RECEDING_POLICY_RETURN_VALUE_CHECKPOINT_V1"
)
DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RECEDING_POLICY_RETURN_ONE_SIDED_ADMISSION_V1"
)
DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RECEDING_POLICY_RETURN_PAIRED_SWMM_DATASET_V1"
)
DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS = 48
DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS = 12
DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS = 24
DIRECT_TFV_POLICY_RETURN_NORMALIZATION = "SQRT_ACTUAL_FIRST_MOVE_CHANGED_FACILITY_COUNT"
_POLICY_RETURN_ROLES = {
    "policy_return_train",
    "policy_return_validation",
    "policy_return_calibration",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _scale(changed_facility_count: int) -> float:
    count = int(changed_facility_count)
    if count <= 0 or count > 109:
        raise ValueError("policy-return calibration requires changed-facility count in [1,109]")
    return math.sqrt(float(count))


def validate_policy_return_record(record: Mapping[str, Any]) -> None:
    """Fail closed on a paired offline SWMM policy-return label.

    Training/validation records are authoritative label records and need not already contain a model
    prediction. Calibration records must contain the prediction from the frozen selected critic.
    """
    if str(record.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("policy-return record has the wrong estimand")
    role = str(record.get("data_role", ""))
    if role not in _POLICY_RETURN_ROLES:
        raise ValueError("policy-return record has an invalid data role")
    if record.get("same_prefix_verified") is not True:
        raise ValueError("policy-return record requires identical authoritative prefix verification")
    if record.get("same_continuation_policy_verified") is not True:
        raise ValueError("candidate/HOLD branches must use the same frozen continuation policy")
    if record.get("future_realized_rainfall_used_online") is not False:
        raise ValueError("policy-return record indicates future rainfall leakage into the online policy")
    group = str(record.get("rainfall_group", ""))
    event = str(record.get("event_id", ""))
    if not group or not event:
        raise ValueError("policy-return record requires rainfall_group and event_id")
    _scale(int(record.get("first_move_changed_facility_count", -1)))
    if role == "policy_return_calibration" or "predicted_policy_return_delta_tfv_m3" in record:
        _finite(record.get("predicted_policy_return_delta_tfv_m3"), label="predicted policy return")
    _finite(record.get("true_policy_return_delta_tfv_m3"), label="true policy return")
    candidate_tfv = _finite(record.get("candidate_branch_tfv_m3"), label="candidate branch TFV")
    hold_tfv = _finite(record.get("hold_branch_tfv_m3"), label="HOLD branch TFV")
    truth = float(record["true_policy_return_delta_tfv_m3"])
    if abs((candidate_tfv - hold_tfv) - truth) > 1.0e-5 * max(1.0, abs(truth)):
        raise ValueError("policy-return truth is inconsistent with paired authoritative TFV")
    for key in (
        "continuation_policy_sha256",
        "prefix_sha256",
        "candidate_first_target_sha256",
        "hold_first_target_sha256",
    ):
        value = str(record.get(key, "")).lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"policy-return record lacks canonical {key}")


def assert_role_disjoint(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    calibration_records: Sequence[Mapping[str, Any]],
) -> None:
    sets = {
        "train": {str(row.get("rainfall_group", "")) for row in train_records},
        "validation": {str(row.get("rainfall_group", "")) for row in validation_records},
        "calibration": {str(row.get("rainfall_group", "")) for row in calibration_records},
    }
    for left, right in (("train", "validation"), ("train", "calibration"), ("validation", "calibration")):
        overlap = sorted(sets[left] & sets[right])
        if overlap:
            raise ValueError(f"policy-return {left}/{right} rainfall groups overlap: {overlap}")
    if len(sets["train"]) < DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS:
        raise ValueError(
            f"policy-return training requires >= {DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS} rainfall groups"
        )
    if len(sets["validation"]) < DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS:
        raise ValueError(
            "policy-return model selection requires >= "
            f"{DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS} rainfall groups"
        )
    if len(sets["calibration"]) < DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS:
        raise ValueError(
            "policy-return calibration requires >= "
            f"{DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS} rainfall groups"
        )


def derive_policy_return_admission(
    *,
    records: Sequence[Mapping[str, Any]],
    expected_rainfall_groups: Sequence[str],
    policy_return_checkpoint_sha256: str,
    continuation_policy_sha256: str,
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
) -> dict[str, Any]:
    """Derive rainfall-group split-conformal admission for the deployed-policy estimand."""
    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("policy-return admission coverage must lie in (0.5,1)")
    expected = {str(value) for value in expected_rainfall_groups if str(value)}
    if len(expected) < DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS:
        raise ValueError("insufficient policy-return calibration rainfall groups")
    by_group: dict[str, list[float]] = {group: [] for group in expected}
    raw_residuals: list[float] = []
    changed_counts: list[int] = []
    for row in records:
        validate_policy_return_record(row)
        if str(row.get("data_role")) != "policy_return_calibration":
            raise ValueError("admission received a non-calibration policy-return row")
        group = str(row["rainfall_group"])
        if group not in expected:
            raise ValueError(f"unexpected policy-return calibration group: {group}")
        if str(row["continuation_policy_sha256"]).lower() != str(continuation_policy_sha256).lower():
            raise ValueError("policy-return calibration mixes continuation-policy lineages")
        predicted = float(row["predicted_policy_return_delta_tfv_m3"])
        truth = float(row["true_policy_return_delta_tfv_m3"])
        changed = int(row["first_move_changed_facility_count"])
        residual = truth - predicted
        by_group[group].append(residual / _scale(changed))
        raw_residuals.append(residual)
        changed_counts.append(changed)
    missing = sorted(group for group, values in by_group.items() if not values)
    if missing:
        raise ValueError(f"policy-return calibration missing rainfall groups: {missing}")
    group_max = {group: max(values) for group, values in by_group.items()}
    q = max(0.0, _one_sided_conformal_upper(list(group_max.values()), float(coverage)))
    ordered = sorted(group_max.values())
    raw = sorted(raw_residuals)
    return {
        "contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
        "development_only": True,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "coverage": float(coverage),
        "independent_unit": "RAINFALL_GROUP_MAX_NORMALIZED_TRUE_MINUS_PREDICTED_RESIDUAL",
        "normalization": DIRECT_TFV_POLICY_RETURN_NORMALIZATION,
        "normalized_residual_conformal_upper": float(q),
        "calibration_rainfall_group_count": len(group_max),
        "calibration_rainfall_groups": sorted(group_max),
        "calibration_record_count": len(records),
        "normalized_group_residual_min": float(ordered[0]),
        "normalized_group_residual_median": float(ordered[len(ordered) // 2]),
        "normalized_group_residual_max": float(ordered[-1]),
        "raw_residual_min_m3": float(raw[0]),
        "raw_residual_median_m3": float(raw[len(raw) // 2]),
        "raw_residual_max_m3": float(raw[-1]),
        "changed_facility_count_min": min(changed_counts),
        "changed_facility_count_max": max(changed_counts),
        "policy_return_checkpoint_sha256": str(policy_return_checkpoint_sha256).lower(),
        "continuation_policy_sha256": str(continuation_policy_sha256).lower(),
        "online_swmm_called": False,
        "generic_d3_floor_controls_execution": False,
        "open_loop_first_move_margin_controls_execution": False,
        "coverage_claim_scope": (
            "Marginal rainfall-group split-conformal coverage for the fixed policy-return score; "
            "no conditional coverage claim is made."
        ),
    }


def policy_return_margin_m3(calibration: Mapping[str, Any], changed_facility_count: int) -> float:
    if str(calibration.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
        raise ValueError("wrong policy-return admission contract")
    if str(calibration.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("wrong policy-return execution estimand")
    if str(calibration.get("normalization", "")) != DIRECT_TFV_POLICY_RETURN_NORMALIZATION:
        raise ValueError("wrong policy-return residual normalization")
    q = _finite(calibration.get("normalized_residual_conformal_upper"), label="policy-return conformal upper")
    if q < 0.0:
        raise ValueError("policy-return conformal upper must be non-negative")
    return float(q * _scale(int(changed_facility_count)))


def _design_from_payload(payload: Mapping[str, Any]) -> DirectTFVValueDesign:
    allowed = {field.name for field in fields(DirectTFVValueDesign)}
    design = DirectTFVValueDesign(**{key: value for key, value in dict(payload).items() if key in allowed})
    design.validate()
    return design


def _normalization(payload: Mapping[str, Any]) -> InputNormalizationV60:
    required = ("state_mean", "state_std", "rainfall_mean", "rainfall_std", "flow_mean", "flow_std")
    arrays = {name: np.asarray(payload[name], dtype=np.float32) for name in required}
    if any(value.size == 0 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("policy-return checkpoint normalization is invalid")
    return InputNormalizationV60(**arrays)


def load_policy_return_checkpoint(
    path: str | Path,
    *,
    graph: Any,
    device: torch.device,
    expected_base_step2_sha256: str,
) -> tuple[DirectFacilityTFVValueModel, InputNormalizationV60, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_CHECKPOINT_CONTRACT:
        raise ValueError("runtime requires a current receding-policy-return checkpoint")
    if payload.get("development_only") is not True:
        raise ValueError("policy-return checkpoint must be Development-only")
    if str(payload.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("policy-return checkpoint has the wrong target estimand")
    if str(payload.get("base_step2_sha256", "")).lower() != str(expected_base_step2_sha256).lower():
        raise ValueError("policy-return checkpoint was initialized from a different Step2 checkpoint")
    if int(payload.get("train_rainfall_group_count", 0)) < DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS:
        raise ValueError("policy-return checkpoint has insufficient independent training rainfall groups")
    if int(payload.get("validation_rainfall_group_count", 0)) < DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS:
        raise ValueError("policy-return checkpoint has insufficient model-selection rainfall groups")
    design = _design_from_payload(payload["model_design"])
    model = DirectFacilityTFVValueModel(
        state_dim=int(payload["state_dim"]), rainfall_dim=int(payload["rainfall_dim"]),
        actuator_physics_dim=int(payload["actuator_physics_dim"]),
        target_scale_m3=float(payload["target_scale_m3"]), design=design,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    normalization = _normalization(payload["normalization"])
    if tuple(str(value) for value in payload.get("actuator_ids", ())) != tuple(str(value) for value in graph.actuator_ids):
        raise ValueError("policy-return checkpoint actuator order differs from graph")
    return model, normalization, payload


__all__ = [
    "DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT",
    "DIRECT_TFV_POLICY_RETURN_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT",
    "DIRECT_TFV_POLICY_RETURN_ESTIMAND",
    "DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS",
    "DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS",
    "DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS",
    "assert_role_disjoint",
    "derive_policy_return_admission",
    "load_policy_return_checkpoint",
    "policy_return_margin_m3",
    "sha256_file",
    "validate_policy_return_record",
]
