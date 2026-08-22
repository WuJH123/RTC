"""Publication-facing reuse contracts for Project7 V23 without new empirical truth.

This module deliberately does *not* manufacture labels or repartition already frozen evidence.
Completed policy-return records keep their original train/validation/calibration rainfall-group roles.
The closed-loop Final cohort remains the separately preregistered Project7 v0.6.9 Final split.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT = (
    "PROJECT7_V23_EXISTING_AUTHORITATIVE_TRUTH_EXACT_MATCH_AUDIT_V2_PREREGISTERED_ROLES"
)
V23_PUBLICATION_ROLE_MANIFEST_CONTRACT = (
    "PROJECT7_V23_PUBLICATION_EXISTING_DATA_ROLE_MANIFEST_V2_PRESERVE_SOURCE_ROLES"
)
V23_FORMAL_PROTOCOL_CONTRACT = "PROJECT7_V23_FORMAL_REUSE_ONLY_PROTOCOL_V2_PREREGISTERED_FINAL"

LEARNING_ROLES = (
    "policy_return_train",
    "policy_return_validation",
    "policy_return_calibration",
)


@dataclass(frozen=True)
class ExactTargetMatch:
    matched: bool
    maximum_absolute_difference: float
    exact_float32_sha256: str


def float32_target_sha256(value: np.ndarray) -> str:
    target = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    if target.shape != (109,):
        raise ValueError("V23 formal target hashing requires 109 settings")
    if not np.isfinite(target).all():
        raise ValueError("V23 formal target hashing received non-finite values")
    return hashlib.sha256(target.tobytes(order="C")).hexdigest()


def compare_candidate_targets(
    generated: np.ndarray,
    recorded: np.ndarray,
    *,
    atol: float = 1.0e-7,
) -> ExactTargetMatch:
    left = np.asarray(generated, dtype=np.float64).reshape(-1)
    right = np.asarray(recorded, dtype=np.float64).reshape(-1)
    if left.shape != (109,) or right.shape != (109,):
        raise ValueError("V23 exact-match audit requires 109-dimensional targets")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("V23 exact-match audit received non-finite target values")
    maximum = float(np.max(np.abs(left - right), initial=0.0))
    return ExactTargetMatch(
        matched=bool(maximum <= float(atol)),
        maximum_absolute_difference=maximum,
        exact_float32_sha256=float32_target_sha256(generated),
    )


def learning_groups_by_role(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = {role: set() for role in LEARNING_ROLES}
    for row in records:
        role = str(row.get("data_role", ""))
        if role not in grouped:
            raise ValueError(f"unexpected existing learning role: {role}")
        group = str(row.get("rainfall_group", ""))
        if not group:
            raise ValueError("existing learning record lacks rainfall_group")
        grouped[role].add(group)
    result = {role: tuple(sorted(values)) for role, values in grouped.items()}
    validate_learning_role_disjointness(result)
    return result


def validate_learning_role_disjointness(roles: Mapping[str, Iterable[str]]) -> None:
    seen: set[str] = set()
    for role in LEARNING_ROLES:
        current = {str(value) for value in roles.get(role, ())}
        if not current:
            raise ValueError(f"existing learning role is empty: {role}")
        overlap = seen & current
        if overlap:
            raise ValueError(
                f"rainfall-group leakage across existing learning roles: {sorted(overlap)[:5]}"
            )
        seen.update(current)


def assert_excluded_events_absent_from_learning(
    records: Sequence[Mapping[str, Any]],
    excluded_event_ids: Iterable[str],
) -> None:
    excluded = {str(value) for value in excluded_event_ids if str(value)}
    collisions = sorted(
        {
            str(row.get("event_id"))
            for row in records
            if str(row.get("event_id")) in excluded
        }
    )
    if collisions:
        raise ValueError(
            "development-steering events leaked into formal learning evidence: "
            f"{collisions}"
        )


def validate_frozen_final_split(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if str(payload.get("contract", "")) != "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1":
        raise ValueError("V23 formal protocol requires the frozen Project7 v0.6.9 split contract")
    invariants = payload.get("invariants")
    if not isinstance(invariants, Mapping):
        raise ValueError("frozen Project7 split lacks invariants")
    required = (
        "rainfall_groups_cross_scientific_splits",
        "final_untouched_before_policy_lock",
        "final_used_for_tuning",
        "validation_used_for_training",
    )
    expected = {
        "rainfall_groups_cross_scientific_splits": False,
        "final_untouched_before_policy_lock": True,
        "final_used_for_tuning": False,
        "validation_used_for_training": False,
    }
    for key in required:
        if invariants.get(key) is not expected[key]:
            raise ValueError(f"frozen Project7 split invariant changed: {key}")
    final = tuple(str(value) for value in payload.get("final", ()))
    if len(final) != 6 or len(set(final)) != 6:
        raise ValueError("frozen Project7 split must contain exactly six unique Final events")
    return final


__all__ = [
    "ExactTargetMatch",
    "LEARNING_ROLES",
    "V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT",
    "V23_FORMAL_PROTOCOL_CONTRACT",
    "V23_PUBLICATION_ROLE_MANIFEST_CONTRACT",
    "assert_excluded_events_absent_from_learning",
    "compare_candidate_targets",
    "float32_target_sha256",
    "learning_groups_by_role",
    "validate_frozen_final_split",
    "validate_learning_role_disjointness",
]
