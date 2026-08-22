from __future__ import annotations

import numpy as np
import pytest

from rtc.project7_v23_formal_reuse import (
    ARCHIVAL_ROLE,
    FROZEN_SPLIT_CONTRACT,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    V23_FORMAL_PROTOCOL_CONTRACT,
    V23_PUBLICATION_ROLE_MANIFEST_CONTRACT,
    compare_candidate_targets,
    scientific_role_for_record,
    validate_frozen_final_split,
    validate_frozen_split,
)


def _split() -> dict[str, object]:
    train = [f"TRAIN_{i}" for i in range(18)]
    validation = [f"VALID_{i}" for i in range(6)]
    final = [f"FINAL_{i}" for i in range(6)]
    return {
        "contract": FROZEN_SPLIT_CONTRACT,
        "counts": {"development_train": 18, "development_validation": 6, "final": 6},
        "development_train": train,
        "development_validation": validation,
        "final": final,
        "invariants": {
            "calibration_role_removed": True,
            "safety_audit_role_removed": True,
            "rainfall_groups_cross_scientific_splits": False,
            "development_groups_cross_train_validation": False,
            "final_untouched_before_policy_lock": True,
            "final_used_for_tuning": False,
            "validation_used_for_training": False,
        },
    }


def test_exact_target_match_is_strict_and_float32_hashed() -> None:
    target = np.linspace(0.0, 1.0, 109, dtype=np.float32)
    assert compare_candidate_targets(target, target.astype(np.float64)).matched is True
    shifted = target.astype(np.float64)
    shifted[8] += 2.0e-6
    assert compare_candidate_targets(target, shifted).matched is False


def test_frozen_split_requires_removed_calibration_role_and_disjoint_18_6_6() -> None:
    payload = _split()
    roles = validate_frozen_split(payload)
    assert len(roles["development_train"]) == 18
    assert len(roles["development_validation"]) == 6
    assert len(roles["final"]) == 6
    assert validate_frozen_final_split(payload) == tuple(payload["final"])
    payload["invariants"]["calibration_role_removed"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="calibration_role_removed"):
        validate_frozen_split(payload)


def test_historical_data_role_never_overrides_frozen_scientific_role() -> None:
    split = _split()
    row = {
        "event_id": "TRAIN_3",
        "rainfall_group": "TRAIN_3",
        "data_role": "policy_return_calibration",
    }
    assert scientific_role_for_record(row, split) == "development_train"
    archival = {
        "event_id": "OLD_CALIBRATION_EVENT",
        "rainfall_group": "OLD_CALIBRATION_EVENT",
        "data_role": "policy_return_calibration",
    }
    assert scientific_role_for_record(archival, split) == ARCHIVAL_ROLE


def test_final_record_is_identified_before_any_truth_reuse() -> None:
    split = _split()
    row = {"event_id": "FINAL_2", "data_role": "policy_return_train"}
    assert scientific_role_for_record(row, split) == "final"


def test_split_rejects_cross_role_overlap_and_final_tuning() -> None:
    split = _split()
    split["development_validation"] = ["TRAIN_0", *[f"VALID_{i}" for i in range(5)]]
    with pytest.raises(ValueError, match="role overlap"):
        validate_frozen_split(split)
    split = _split()
    split["invariants"]["final_used_for_tuning"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="final_used_for_tuning"):
        validate_frozen_split(split)


def test_formal_contracts_are_explicitly_versioned_and_split_authoritative() -> None:
    assert "V3_SPLIT_AUTHORITY" in V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT
    assert "V3_SPLIT_AUTHORITY" in V23_PUBLICATION_ROLE_MANIFEST_CONTRACT
    assert "V3_FIXED_POLICY_FALLBACK" in V23_FORMAL_PROTOCOL_CONTRACT
