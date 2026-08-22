from __future__ import annotations

import numpy as np
import pytest

from rtc.project7_v23_formal_reuse import (
    LEARNING_ROLES,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    V23_FORMAL_PROTOCOL_CONTRACT,
    V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
    assert_excluded_events_absent_from_learning,
    compare_candidate_targets,
    float32_target_sha256,
    learning_groups_by_role,
    validate_frozen_final_split,
)


def test_exact_target_match_is_strict_and_float32_hashed() -> None:
    target = np.linspace(0.0, 1.0, 109, dtype=np.float32)
    result = compare_candidate_targets(target, target.astype(np.float64))
    assert result.matched is True
    assert result.maximum_absolute_difference == 0.0
    assert result.exact_float32_sha256 == float32_target_sha256(target)
    shifted = target.astype(np.float64)
    shifted[8] += 2.0e-6
    assert compare_candidate_targets(target, shifted).matched is False


def _records() -> list[dict[str, str]]:
    return [
        {"data_role": "policy_return_train", "rainfall_group": "T1", "event_id": "E1"},
        {"data_role": "policy_return_validation", "rainfall_group": "V1", "event_id": "E2"},
        {"data_role": "policy_return_calibration", "rainfall_group": "C1", "event_id": "E3"},
    ]


def test_existing_learning_roles_are_preserved_and_group_disjoint() -> None:
    roles = learning_groups_by_role(_records())
    assert tuple(roles) == LEARNING_ROLES
    assert roles["policy_return_train"] == ("T1",)
    assert roles["policy_return_validation"] == ("V1",)
    assert roles["policy_return_calibration"] == ("C1",)


def test_existing_learning_roles_reject_group_leakage() -> None:
    rows = _records()
    rows[1]["rainfall_group"] = "T1"
    with pytest.raises(ValueError, match="leakage"):
        learning_groups_by_role(rows)


def test_development_events_are_forbidden_from_formal_learning() -> None:
    with pytest.raises(ValueError, match="development-steering events"):
        assert_excluded_events_absent_from_learning(_records(), ["E2"])


def test_frozen_v069_final_split_is_required() -> None:
    payload = {
        "contract": "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1",
        "final": [f"FINAL_{index}" for index in range(6)],
        "invariants": {
            "rainfall_groups_cross_scientific_splits": False,
            "final_untouched_before_policy_lock": True,
            "final_used_for_tuning": False,
            "validation_used_for_training": False,
        },
    }
    assert validate_frozen_final_split(payload) == tuple(payload["final"])
    payload["invariants"]["final_used_for_tuning"] = True
    with pytest.raises(ValueError, match="invariant changed"):
        validate_frozen_final_split(payload)


def test_formal_contracts_are_explicitly_versioned() -> None:
    assert "EXACT_MATCH_AUDIT_V2" in V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT
    assert "ROLE_MANIFEST_V2" in V23_PUBLICATION_ROLE_MANIFEST_CONTRACT
    assert "PROTOCOL_V2" in V23_FORMAL_PROTOCOL_CONTRACT
