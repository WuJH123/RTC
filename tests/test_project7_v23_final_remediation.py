from __future__ import annotations

import pytest

from rtc.project7_v23_final_remediation import (
    FinalCandidate,
    ORIGINAL_SPLIT_CONTRACT,
    REMEDIATED_SPLIT_CONTRACT,
    select_reblind_final,
    validate_remediated_split,
)
from rtc.project7_v23_formal_reuse import validate_frozen_split


def _candidate(event_id: str, rp: int, duration: int, *, exposed: bool = False) -> FinalCandidate:
    return FinalCandidate(
        event_id=event_id,
        return_period_year=rp,
        duration_minutes=duration,
        inp_path=f"/tmp/{event_id}.inp",
        prepared_inp_sha256="a" * 64,
        original_role="outside_original_split",
        exposed_prelock=exposed,
    )


def _remediated() -> dict[str, object]:
    return {
        "contract": REMEDIATED_SPLIT_CONTRACT,
        "source_split_contract": ORIGINAL_SPLIT_CONTRACT,
        "formal_mode": "FIXED_POLICY_NO_RETRAIN",
        "formal_retraining_allowed": False,
        "source_final_contamination_detected": True,
        "final_exposure_count_prelock": 0,
        "contaminated_records_deleted_or_relabeled": False,
        "selection_basis": {
            "hydraulic_outcomes_used": False,
            "controller_performance_used": False,
            "forcing_descriptors_used": ["return_period_year", "duration_minutes"],
            "prelock_exposure_status_used": True,
        },
        "development_train": [f"TRAIN_{i}" for i in range(13)],
        "development_validation": [f"VALID_{i}" for i in range(6)],
        "final": [f"FINAL_NEW_{i}" for i in range(6)],
        "quarantined_prelock_exposed_final": [f"FINAL_OLD_{i}" for i in range(5)],
        "counts": {
            "development_train": 13,
            "development_validation": 6,
            "final": 6,
            "quarantined_final": 5,
        },
        "invariants": {
            "calibration_role_removed": True,
            "safety_audit_role_removed": True,
            "rainfall_groups_cross_scientific_splits": False,
            "development_groups_cross_train_validation": False,
            "final_untouched_before_policy_lock": True,
            "final_used_for_tuning": False,
            "validation_used_for_training": False,
            "contaminated_source_final_excluded_from_publication_final": True,
        },
    }


def test_reblind_selector_excludes_exposed_and_validation_and_covers_six_durations() -> None:
    candidates = []
    for duration in (60, 120, 180, 240, 300, 360):
        candidates.append(_candidate(f"SAFE_{duration}", 20, duration))
        candidates.append(_candidate(f"EXPOSED_{duration}", 5, duration, exposed=True))
        candidates.append(_candidate(f"VALID_{duration}", 10, duration))
    selected = select_reblind_final(
        candidates,
        protected_validation_events=[f"VALID_{duration}" for duration in (60, 120, 180, 240, 300, 360)],
        original_final_events=[],
    )
    assert [row.duration_minutes for row in selected] == [60, 120, 180, 240, 300, 360]
    assert all(row.event_id.startswith("SAFE_") for row in selected)


def test_reblind_selector_prefers_still_clean_source_final_without_using_performance() -> None:
    candidates = []
    original = []
    for duration in (60, 120, 180, 240, 300, 360):
        candidates.append(_candidate(f"OTHER_{duration}", 5, duration))
        clean = _candidate(f"ORIGINAL_FINAL_{duration}", 100, duration)
        candidates.append(clean)
        original.append(clean.event_id)
    selected = select_reblind_final(
        candidates,
        protected_validation_events=[],
        original_final_events=original,
    )
    assert [row.event_id for row in selected] == original


def test_reblind_selector_fails_when_a_duration_has_no_clean_candidate() -> None:
    candidates = [_candidate(f"SAFE_{duration}", 20, duration) for duration in (60, 120, 180, 240, 300)]
    with pytest.raises(ValueError, match="missing durations"):
        select_reblind_final(candidates, protected_validation_events=[], original_final_events=[])


def test_remediated_split_is_fixed_policy_only_and_supported_by_formal_validator() -> None:
    payload = _remediated()
    roles = validate_remediated_split(payload)
    assert len(roles["development_train"]) == 13
    assert len(roles["development_validation"]) == 6
    assert len(roles["final"]) == 6
    assert validate_frozen_split(payload) == roles
    payload["formal_retraining_allowed"] = True
    with pytest.raises(ValueError, match="forbid Formal retraining"):
        validate_remediated_split(payload)


def test_quarantined_source_final_cannot_reenter_any_scientific_role() -> None:
    payload = _remediated()
    payload["final"] = ["FINAL_OLD_0", *[f"FINAL_NEW_{i}" for i in range(5)]]
    with pytest.raises(ValueError, match="quarantined"):
        validate_remediated_split(payload)
