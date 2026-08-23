"""Publication-facing Project7 V23 Formal reuse and split-authority contracts.

The original v0.6.9 18/6/6 forcing-only split remains immutable provenance. If its Final cohort is
shown to have pre-lock historical exposure, publication Final authority may move only to the explicit
contamination-remediated fixed-policy successor split. The successor preserves six independent,
exposure-free forcing strata without requiring the contaminated source-Final duration cells to be
recreated. Historical rows are never deleted/relabelled and no helper here manufactures labels or
opens Final hydraulic evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .project7_v23_final_remediation import (
    REMEDIATED_SPLIT_CONTRACT,
    validate_remediated_split,
)


FROZEN_SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"
V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT = (
    "PROJECT7_V23_EXISTING_AUTHORITATIVE_TRUTH_EXACT_MATCH_AUDIT_V4_CONTAMINATION_REMEDIATION"
)
V23_PUBLICATION_ROLE_MANIFEST_CONTRACT = (
    "PROJECT7_V23_PUBLICATION_ROLE_MANIFEST_V4_CONTAMINATION_REMEDIATION"
)
V23_FORMAL_PROTOCOL_CONTRACT = "PROJECT7_V23_FORMAL_PROTOCOL_V4_REBLIND_FIXED_POLICY_CAPABLE"

SCIENTIFIC_ROLES = ("development_train", "development_validation", "final")
FORMAL_LEARNING_ROLES = ("development_train", "development_validation")
ARCHIVAL_ROLE = "archival_only"


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


def _validate_original_frozen_split(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    counts = payload.get("counts")
    invariants = payload.get("invariants")
    if not isinstance(counts, Mapping) or not isinstance(invariants, Mapping):
        raise ValueError("frozen Project7 split lacks counts/invariants")
    expected_counts = {"development_train": 18, "development_validation": 6, "final": 6}
    for role, expected in expected_counts.items():
        values = tuple(str(value) for value in payload.get(role, ()))
        if int(counts.get(role, -1)) != expected or len(values) != expected:
            raise ValueError(f"frozen Project7 split changed: {role}")
        if len(set(values)) != expected:
            raise ValueError(f"frozen Project7 split contains duplicate {role} events")
    expected_invariants = {
        "calibration_role_removed": True,
        "safety_audit_role_removed": True,
        "rainfall_groups_cross_scientific_splits": False,
        "development_groups_cross_train_validation": False,
        "final_untouched_before_policy_lock": True,
        "final_used_for_tuning": False,
        "validation_used_for_training": False,
    }
    for key, expected in expected_invariants.items():
        if invariants.get(key) is not expected:
            raise ValueError(f"frozen Project7 split invariant changed: {key}")
    roles = {
        role: tuple(str(value) for value in payload.get(role, ())) for role in SCIENTIFIC_ROLES
    }
    seen: set[str] = set()
    for role in SCIENTIFIC_ROLES:
        overlap = seen & set(roles[role])
        if overlap:
            raise ValueError(f"frozen Project7 split role overlap: {sorted(overlap)}")
        seen.update(roles[role])
    return roles


def validate_frozen_split(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Validate either the original frozen split or its explicit contamination remediation."""
    contract = str(payload.get("contract", ""))
    if contract == FROZEN_SPLIT_CONTRACT:
        return _validate_original_frozen_split(payload)
    if contract == REMEDIATED_SPLIT_CONTRACT:
        return validate_remediated_split(payload)
    raise ValueError(
        "V23 Formal requires either the original v0.6.9 split or its explicit contamination-remediated successor"
    )


def validate_frozen_final_split(payload: Mapping[str, Any]) -> tuple[str, ...]:
    return validate_frozen_split(payload)["final"]


def _normalise_record_identity(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).name.removesuffix(".inp")


def scientific_role_for_record(
    row: Mapping[str, Any],
    split_payload: Mapping[str, Any],
) -> str:
    """Map a historical record to the active frozen scientific role, never its old data_role."""
    roles = validate_frozen_split(split_payload)
    identities = {
        _normalise_record_identity(row.get("event_id")),
        _normalise_record_identity(row.get("rainfall_group")),
    }
    identities.discard("")
    matches = [
        role
        for role in SCIENTIFIC_ROLES
        if identities & {_normalise_record_identity(value) for value in roles[role]}
    ]
    if len(matches) > 1:
        raise ValueError("historical record maps to multiple frozen scientific roles")
    return matches[0] if matches else ARCHIVAL_ROLE


def split_role_coverage(
    records: list[Mapping[str, Any]],
    split_payload: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    roles = validate_frozen_split(split_payload)
    covered: dict[str, set[str]] = {role: set() for role in SCIENTIFIC_ROLES}
    for row in records:
        role = scientific_role_for_record(row, split_payload)
        if role == ARCHIVAL_ROLE:
            continue
        event = _normalise_record_identity(row.get("event_id"))
        group = _normalise_record_identity(row.get("rainfall_group"))
        for expected in roles[role]:
            canonical = _normalise_record_identity(expected)
            if canonical in {event, group}:
                covered[role].add(expected)
    return {role: tuple(sorted(values)) for role, values in covered.items()}


__all__ = [
    "ARCHIVAL_ROLE",
    "ExactTargetMatch",
    "FORMAL_LEARNING_ROLES",
    "FROZEN_SPLIT_CONTRACT",
    "SCIENTIFIC_ROLES",
    "V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT",
    "V23_FORMAL_PROTOCOL_CONTRACT",
    "V23_PUBLICATION_ROLE_MANIFEST_CONTRACT",
    "compare_candidate_targets",
    "float32_target_sha256",
    "scientific_role_for_record",
    "split_role_coverage",
    "validate_frozen_final_split",
    "validate_frozen_split",
]
