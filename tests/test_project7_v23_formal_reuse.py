from __future__ import annotations

import numpy as np
import pytest

from rtc.project7_v23_formal_reuse import (
    PUBLICATION_ROLES,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    V23_FORMAL_PROTOCOL_CONTRACT,
    V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
    assert_excluded_events_absent,
    compare_candidate_targets,
    deterministic_publication_split,
    float32_target_sha256,
    validate_role_disjointness,
)


def test_exact_target_match_is_strict_and_float32_hashed() -> None:
    target = np.linspace(0.0, 1.0, 109, dtype=np.float32)
    same = target.astype(np.float64)
    result = compare_candidate_targets(target, same)
    assert result.matched is True
    assert result.maximum_absolute_difference == 0.0
    assert result.exact_float32_sha256 == float32_target_sha256(target)

    shifted = same.copy()
    shifted[8] += 2.0e-6
    result = compare_candidate_targets(target, shifted)
    assert result.matched is False
    assert result.maximum_absolute_difference > 1.0e-6


def test_publication_split_is_deterministic_and_rainfall_group_disjoint() -> None:
    groups = [f"RAIN_{index:03d}" for index in range(80)]
    first = deterministic_publication_split(groups, seed=42)
    second = deterministic_publication_split(list(reversed(groups)), seed=42)
    assert first == second
    assert tuple(first) == PUBLICATION_ROLES
    validate_role_disjointness(first)
    flattened = [group for values in first.values() for group in values]
    assert len(flattened) == len(set(flattened)) == 80
    assert len(first["train"]) == 56
    assert len(first["validation"]) == 8
    assert len(first["calibration"]) == 8
    assert len(first["final_test"]) == 8


def test_publication_split_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError, match="at least 40"):
        deterministic_publication_split([f"G{i}" for i in range(39)])


def test_development_events_are_forbidden_from_every_publication_role() -> None:
    groups = [f"G{i:02d}" for i in range(40)]
    roles = deterministic_publication_split(groups)
    mapping = {group: {f"EVENT_{group}"} for group in groups}
    final_group = roles["final_test"][0]
    excluded = f"EVENT_{final_group}"
    with pytest.raises(ValueError, match="development-steering event"):
        assert_excluded_events_absent(mapping, roles, [excluded])


def test_formal_contract_names_are_versioned_and_explicit() -> None:
    assert V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT.endswith("_V1")
    assert V23_PUBLICATION_ROLE_MANIFEST_CONTRACT.endswith("_V1")
    assert V23_FORMAL_PROTOCOL_CONTRACT.endswith("_V1")
