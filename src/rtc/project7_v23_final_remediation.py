"""Fail-closed governance for Project7 V23 Final-cohort contamination remediation.

This module never opens hydraulic outcomes and never changes controller parameters.  It supports a
single scientifically defensible remediation: quarantine any preregistered Final event that had
pre-lock exposure, then freeze a replacement Final cohort from already prepared events that have no
recorded model/development exposure.  Selection uses forcing descriptors only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ORIGINAL_SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"
FINAL_EXPOSURE_LEDGER_CONTRACT = "PROJECT7_V23_FINAL_PRELOCK_EXPOSURE_LEDGER_V1"
REMEDIATED_SPLIT_CONTRACT = (
    "PROJECT7_V069R1_CONTAMINATION_REMEDIATED_FIXED_POLICY_SPLIT_V1"
)
TARGET_DURATIONS_MINUTES = (60, 120, 180, 240, 300, 360)


@dataclass(frozen=True)
class FinalCandidate:
    event_id: str
    return_period_year: int
    duration_minutes: int
    inp_path: str
    prepared_inp_sha256: str
    original_role: str
    exposed_prelock: bool


def normalise_event_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).name.removesuffix(".inp")


def validate_exposure_ledger(payload: Mapping[str, Any]) -> tuple[FinalCandidate, ...]:
    if str(payload.get("contract", "")) != FINAL_EXPOSURE_LEDGER_CONTRACT:
        raise ValueError("wrong Project7 V23 Final exposure-ledger contract")
    if payload.get("hydraulic_outcomes_used_for_eligibility") is not False:
        raise ValueError("Final eligibility must not use hydraulic outcomes")
    if payload.get("controller_performance_used_for_eligibility") is not False:
        raise ValueError("Final eligibility must not use controller performance")
    rows = payload.get("events")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Final exposure ledger contains no candidate events")
    result: list[FinalCandidate] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Final exposure-ledger event must be an object")
        event_id = normalise_event_id(row.get("event_id"))
        if not event_id or event_id in seen:
            raise ValueError("Final exposure ledger contains missing/duplicate event_id")
        seen.add(event_id)
        digest = str(row.get("prepared_inp_sha256", "")).strip().lower()
        if len(digest) != 64:
            raise ValueError(f"candidate event lacks prepared INP SHA256: {event_id}")
        result.append(
            FinalCandidate(
                event_id=event_id,
                return_period_year=int(row["return_period_year"]),
                duration_minutes=int(row["duration_minutes"]),
                inp_path=str(row.get("inp_path", "")),
                prepared_inp_sha256=digest,
                original_role=str(row.get("original_role", "outside_original_split")),
                exposed_prelock=bool(row.get("exposed_prelock", False)),
            )
        )
    return tuple(result)


def select_reblind_final(
    candidates: Sequence[FinalCandidate],
    *,
    protected_validation_events: Iterable[str],
    original_final_events: Iterable[str],
) -> tuple[FinalCandidate, ...]:
    """Select six unexposed events using only forcing descriptors and provenance status.

    The final cohort keeps one event at each of the six frozen durations.  Within a duration, the
    deterministic tie-break first prefers a still-clean member of the original Final cohort, then a
    return period not yet represented, then the least-used return period, then event_id.  No hydraulic
    or controller metric is accepted by this function.
    """
    protected = {normalise_event_id(value) for value in protected_validation_events}
    original_final = {normalise_event_id(value) for value in original_final_events}
    by_duration: dict[int, list[FinalCandidate]] = {value: [] for value in TARGET_DURATIONS_MINUTES}
    for candidate in candidates:
        if candidate.exposed_prelock or candidate.event_id in protected:
            continue
        if candidate.duration_minutes in by_duration:
            by_duration[candidate.duration_minutes].append(candidate)
    missing = [duration for duration, rows in by_duration.items() if not rows]
    if missing:
        raise ValueError(
            "insufficient untouched prepared events for a six-duration reblind Final cohort; "
            f"missing durations={missing}"
        )

    selected: list[FinalCandidate] = []
    rp_counts: Counter[int] = Counter()
    for duration in TARGET_DURATIONS_MINUTES:
        rows = by_duration[duration]
        rows = sorted(
            rows,
            key=lambda row: (
                0 if row.event_id in original_final else 1,
                0 if rp_counts[row.return_period_year] == 0 else 1,
                rp_counts[row.return_period_year],
                row.return_period_year,
                row.event_id,
            ),
        )
        chosen = rows[0]
        selected.append(chosen)
        rp_counts[chosen.return_period_year] += 1
    if len({row.event_id for row in selected}) != 6:
        raise ValueError("reblind Final selector produced duplicate events")
    return tuple(selected)


def validate_remediated_split(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if str(payload.get("contract", "")) != REMEDIATED_SPLIT_CONTRACT:
        raise ValueError("wrong contamination-remediated split contract")
    if str(payload.get("source_split_contract", "")) != ORIGINAL_SPLIT_CONTRACT:
        raise ValueError("remediated split must descend from the frozen v0.6.9 split")
    if str(payload.get("formal_mode", "")) != "FIXED_POLICY_NO_RETRAIN":
        raise ValueError("contamination remediation is fixed-policy only")
    if payload.get("formal_retraining_allowed") is not False:
        raise ValueError("contamination-remediated split must forbid Formal retraining")
    if payload.get("source_final_contamination_detected") is not True:
        raise ValueError("remediated split requires documented source Final contamination")
    if int(payload.get("final_exposure_count_prelock", -1)) != 0:
        raise ValueError("every remediated Final event must be unexposed pre-lock")
    if payload.get("contaminated_records_deleted_or_relabeled") is not False:
        raise ValueError("contaminated historical records must be quarantined, not rewritten")

    basis = payload.get("selection_basis")
    if not isinstance(basis, Mapping):
        raise ValueError("remediated split lacks selection_basis")
    required_basis = {
        "hydraulic_outcomes_used": False,
        "controller_performance_used": False,
        "forcing_descriptors_used": ["return_period_year", "duration_minutes"],
        "prelock_exposure_status_used": True,
    }
    for key, expected in required_basis.items():
        if basis.get(key) != expected:
            raise ValueError(f"remediated Final selection basis changed: {key}")

    counts = payload.get("counts")
    invariants = payload.get("invariants")
    if not isinstance(counts, Mapping) or not isinstance(invariants, Mapping):
        raise ValueError("remediated split lacks counts/invariants")
    roles = {
        "development_train": tuple(str(value) for value in payload.get("development_train", ())),
        "development_validation": tuple(
            str(value) for value in payload.get("development_validation", ())
        ),
        "final": tuple(str(value) for value in payload.get("final", ())),
    }
    if len(roles["development_validation"]) != 6 or int(
        counts.get("development_validation", -1)
    ) != 6:
        raise ValueError("remediated split must preserve six Development-Validation events")
    if len(roles["final"]) != 6 or int(counts.get("final", -1)) != 6:
        raise ValueError("remediated split must contain exactly six Final events")
    if len(roles["development_train"]) != int(counts.get("development_train", -1)):
        raise ValueError("remediated development_train count mismatch")
    for role, values in roles.items():
        if len(values) != len(set(values)):
            raise ValueError(f"remediated split contains duplicate {role} events")

    quarantine = tuple(str(value) for value in payload.get("quarantined_prelock_exposed_final", ()))
    if not quarantine or len(quarantine) != int(counts.get("quarantined_final", -1)):
        raise ValueError("remediated split must explicitly quarantine contaminated source-Final events")
    seen: set[str] = set()
    for role in ("development_train", "development_validation", "final"):
        overlap = seen & set(roles[role])
        if overlap:
            raise ValueError(f"remediated split role overlap: {sorted(overlap)}")
        seen.update(roles[role])
    if seen & set(quarantine):
        raise ValueError("quarantined contaminated Final event was reused in a scientific role")

    expected_invariants = {
        "calibration_role_removed": True,
        "safety_audit_role_removed": True,
        "rainfall_groups_cross_scientific_splits": False,
        "development_groups_cross_train_validation": False,
        "final_untouched_before_policy_lock": True,
        "final_used_for_tuning": False,
        "validation_used_for_training": False,
        "contaminated_source_final_excluded_from_publication_final": True,
    }
    for key, expected in expected_invariants.items():
        if invariants.get(key) is not expected:
            raise ValueError(f"remediated split invariant changed: {key}")
    return roles


__all__ = [
    "FINAL_EXPOSURE_LEDGER_CONTRACT",
    "FinalCandidate",
    "ORIGINAL_SPLIT_CONTRACT",
    "REMEDIATED_SPLIT_CONTRACT",
    "TARGET_DURATIONS_MINUTES",
    "normalise_event_id",
    "select_reblind_final",
    "validate_exposure_ledger",
    "validate_remediated_split",
]
