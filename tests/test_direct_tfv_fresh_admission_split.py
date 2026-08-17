from __future__ import annotations

from dataclasses import dataclass

import pytest

from rtc.direct_tfv_admission_split import (
    DIRECT_TFV_FRESH_ADMISSION_DATA_CONTRACT,
    minimum_calibration_rainfall_groups,
    validate_fresh_admission_partition,
)


@dataclass
class _Entry:
    rainfall_group: str
    event_id: str


class _Cache:
    def __init__(self, entries: dict[str, _Entry]) -> None:
        self.entries = entries

    def entry(self, name: str) -> _Entry:
        return self.entries[name]


def _names(prefix: str, count: int) -> tuple[list[str], _Cache]:
    names: list[str] = []
    entries: dict[str, _Entry] = {}
    for index in range(count):
        name = f"D3::{prefix}-rain-{index}::{prefix}-event-{index}::cp"
        names.append(name)
        entries[name] = _Entry(
            rainfall_group=f"{prefix}-rain-{index}",
            event_id=f"{prefix}-event-{index}",
        )
    return names, _Cache(entries)


def _combined_cache(*caches: _Cache) -> _Cache:
    merged: dict[str, _Entry] = {}
    for cache in caches:
        merged.update(cache.entries)
    return _Cache(merged)


def test_ninety_percent_fresh_calibration_needs_nine_independent_rainfall_groups() -> None:
    assert minimum_calibration_rainfall_groups(0.90) == 9


def test_fresh_nine_group_calibration_can_keep_four_group_audit_wholly_independent() -> None:
    train, train_cache = _names("train", 14)
    audit, audit_cache = _names("audit", 4)
    fresh, fresh_cache = _names("fresh", 9)
    partition = validate_fresh_admission_partition(
        base_cache=_combined_cache(train_cache, audit_cache),
        step2_trainfit_d3_names=train,
        original_audit_d3_names=audit,
        fresh_cache=fresh_cache,
        fresh_calibration_d3_names=fresh,
        coverage=0.90,
        optimizer_replay_events=("T5_D120_chicago",),
    )
    assert partition["contract"] == DIRECT_TFV_FRESH_ADMISSION_DATA_CONTRACT
    assert partition["fresh_calibration_rainfall_group_count"] == 9
    assert partition["original_audit_rainfall_group_count"] == 4
    assert partition["ready_for_admission_calibration"] is True
    assert partition["train_calibration_rainfall_overlap_count"] == 0
    assert partition["calibration_audit_rainfall_overlap_count"] == 0


def test_fresh_eight_group_calibration_fails_closed_at_ninety_percent() -> None:
    train, train_cache = _names("train", 14)
    audit, audit_cache = _names("audit", 4)
    fresh, fresh_cache = _names("fresh", 8)
    with pytest.raises(ValueError, match="at least 9"):
        validate_fresh_admission_partition(
            base_cache=_combined_cache(train_cache, audit_cache),
            step2_trainfit_d3_names=train,
            original_audit_d3_names=audit,
            fresh_cache=fresh_cache,
            fresh_calibration_d3_names=fresh,
            coverage=0.90,
        )


def test_fresh_calibration_cannot_reuse_step2_trainfit_or_audit_rainfall_groups() -> None:
    train, train_cache = _names("train", 14)
    audit, audit_cache = _names("audit", 4)
    fresh, fresh_cache = _names("fresh", 9)
    fresh_cache.entries[fresh[0]].rainfall_group = "train-rain-0"
    with pytest.raises(ValueError, match="role leakage"):
        validate_fresh_admission_partition(
            base_cache=_combined_cache(train_cache, audit_cache),
            step2_trainfit_d3_names=train,
            original_audit_d3_names=audit,
            fresh_cache=fresh_cache,
            fresh_calibration_d3_names=fresh,
            coverage=0.90,
        )

    fresh, fresh_cache = _names("fresh", 9)
    fresh_cache.entries[fresh[0]].rainfall_group = "audit-rain-0"
    with pytest.raises(ValueError, match="role leakage"):
        validate_fresh_admission_partition(
            base_cache=_combined_cache(train_cache, audit_cache),
            step2_trainfit_d3_names=train,
            original_audit_d3_names=audit,
            fresh_cache=fresh_cache,
            fresh_calibration_d3_names=fresh,
            coverage=0.90,
        )


def test_fresh_calibration_rejects_optimizer_replay_and_reserved_postcalibration_events() -> None:
    train, train_cache = _names("train", 14)
    audit, audit_cache = _names("audit", 4)
    base = _combined_cache(train_cache, audit_cache)

    fresh, fresh_cache = _names("fresh", 9)
    fresh_cache.entries[fresh[0]].event_id = "T5_D120_chicago"
    # Exact replay-event reuse is caught by the stable structured role-overlap key before
    # the secondary substring/token guard.  Assert that machine-readable contract rather
    # than prose from the later (unreachable for an exact ID match) error branch.
    with pytest.raises(ValueError, match="calibration_optimizer_replay_event"):
        validate_fresh_admission_partition(
            base_cache=base,
            step2_trainfit_d3_names=train,
            original_audit_d3_names=audit,
            fresh_cache=fresh_cache,
            fresh_calibration_d3_names=fresh,
            coverage=0.90,
            optimizer_replay_events=("T5_D120_chicago",),
        )

    fresh, fresh_cache = _names("fresh", 9)
    fresh_cache.entries[fresh[1]].event_id = "my_T10_D180_chicago_event"
    with pytest.raises(ValueError, match="reserved post-calibration"):
        validate_fresh_admission_partition(
            base_cache=base,
            step2_trainfit_d3_names=train,
            original_audit_d3_names=audit,
            fresh_cache=fresh_cache,
            fresh_calibration_d3_names=fresh,
            coverage=0.90,
        )


def test_fresh_calibration_rejects_untouched_evaluation_identifiers() -> None:
    train, train_cache = _names("train", 14)
    audit, audit_cache = _names("audit", 4)
    fresh, fresh_cache = _names("fresh", 9)
    fresh_cache.entries[fresh[0]].event_id = "Validation_event_001"
    with pytest.raises(ValueError, match="untouched-evaluation"):
        validate_fresh_admission_partition(
            base_cache=_combined_cache(train_cache, audit_cache),
            step2_trainfit_d3_names=train,
            original_audit_d3_names=audit,
            fresh_cache=fresh_cache,
            fresh_calibration_d3_names=fresh,
            coverage=0.90,
        )