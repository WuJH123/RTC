"""Publication-facing reuse contracts for Project7 V23 without new empirical truth.

This module deliberately does *not* manufacture labels.  It provides fail-closed helpers for
reusing already completed authoritative policy-return records when, and only when, the exact V23
candidate executed at the same frozen query is numerically identical to the recorded candidate.
Formal roles are assigned by rainfall group rather than decision row.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT = (
    "PROJECT7_V23_EXISTING_AUTHORITATIVE_TRUTH_EXACT_MATCH_AUDIT_V1"
)
V23_PUBLICATION_ROLE_MANIFEST_CONTRACT = (
    "PROJECT7_V23_PUBLICATION_EXISTING_DATA_ROLE_MANIFEST_V1"
)
V23_FORMAL_PROTOCOL_CONTRACT = "PROJECT7_V23_FORMAL_REUSE_ONLY_PROTOCOL_V1"

PUBLICATION_ROLES = ("train", "validation", "calibration", "final_test")
DEFAULT_ROLE_FRACTIONS = (0.70, 0.10, 0.10, 0.10)


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


def deterministic_publication_split(
    groups: Sequence[str],
    *,
    seed: int = 42,
    fractions: Sequence[float] = DEFAULT_ROLE_FRACTIONS,
) -> dict[str, tuple[str, ...]]:
    """Split complete rainfall groups into publication roles without row-level leakage."""
    unique = sorted({str(value) for value in groups if str(value)})
    if len(unique) < 40:
        raise ValueError("publication role split requires at least 40 complete rainfall groups")
    values = tuple(float(value) for value in fractions)
    if len(values) != 4 or any(value <= 0.0 for value in values):
        raise ValueError("publication role fractions must contain four positive values")
    if abs(sum(values) - 1.0) > 1.0e-9:
        raise ValueError("publication role fractions must sum to one")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest(),
    )
    total = len(ordered)
    counts = [int(np.floor(total * value)) for value in values]
    for index in range(total - sum(counts)):
        counts[index % len(counts)] += 1
    if min(counts) <= 0:
        raise ValueError("publication role split produced an empty role")
    result: dict[str, tuple[str, ...]] = {}
    offset = 0
    for role, count in zip(PUBLICATION_ROLES, counts, strict=True):
        result[role] = tuple(sorted(ordered[offset : offset + count]))
        offset += count
    assert offset == total
    validate_role_disjointness(result)
    return result


def validate_role_disjointness(roles: Mapping[str, Iterable[str]]) -> None:
    seen: set[str] = set()
    for role in PUBLICATION_ROLES:
        current = {str(value) for value in roles.get(role, ())}
        if not current:
            raise ValueError(f"publication role is empty: {role}")
        overlap = seen & current
        if overlap:
            raise ValueError(
                f"rainfall-group leakage across publication roles: {sorted(overlap)[:5]}"
            )
        seen.update(current)


def assert_excluded_events_absent(
    event_ids_by_group: Mapping[str, Iterable[str]],
    roles: Mapping[str, Iterable[str]],
    excluded_event_ids: Iterable[str],
) -> None:
    excluded = {str(value) for value in excluded_event_ids if str(value)}
    if not excluded:
        return
    for role in PUBLICATION_ROLES:
        for group in roles.get(role, ()):
            events = {str(value) for value in event_ids_by_group.get(str(group), ())}
            collision = events & excluded
            if collision:
                raise ValueError(
                    f"development-steering event leaked into publication role {role}: "
                    f"{sorted(collision)}"
                )


__all__ = [
    "DEFAULT_ROLE_FRACTIONS",
    "ExactTargetMatch",
    "PUBLICATION_ROLES",
    "V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT",
    "V23_FORMAL_PROTOCOL_CONTRACT",
    "V23_PUBLICATION_ROLE_MANIFEST_CONTRACT",
    "assert_excluded_events_absent",
    "compare_candidate_targets",
    "deterministic_publication_split",
    "float32_target_sha256",
    "validate_role_disjointness",
]
