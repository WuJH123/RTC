"""Fresh Development rainfall-group split contract for Direct-TFV admission.

The accepted Step2 V5 model was trained on its TrainFit rainfall groups, so those groups cannot be
reused as the split-conformal calibration sample without weakening the intended held-out finite-
sample argument.  Conversely, the historical Step2 holdout contains only four independent D3
rainfall groups: useful as an audit, but insufficient for 90% one-sided split conformal calibration.

This module therefore enforces three mutually disjoint Development roles:

* Step2 TrainFit rainfall groups: model fitting only;
* fresh D3-HOLD rainfall groups: admission calibration only;
* original Step2 D3 holdout rainfall groups: independent admission audit only.

The fresh calibration set is also kept disjoint from optimizer-replay calibration events and reserved
post-calibration Development events.  No Validation/Final/Formal/Policy-Lock asset is permitted.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


DIRECT_TFV_FRESH_ADMISSION_DATA_CONTRACT = (
    "PROJECT7_DIRECT_TFV_FRESH_D3_HOLD_ADMISSION_CALIBRATION_V1"
)
DIRECT_TFV_FRESH_ADMISSION_SOURCE = "FRESH_DEVELOPMENT_D3_HOLD_NOT_STEP2_TRAIN"
DIRECT_TFV_ADMISSION_AUDIT_SOURCE = "ORIGINAL_STEP2_D3_HOLDOUT"
DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS = ("T10_D180", "T20_D300")
FORBIDDEN_EVALUATION_TOKENS = ("validation", "final", "formal", "policy_lock", "policylock")


def minimum_calibration_rainfall_groups(coverage: float) -> int:
    """Minimum independent units with a finite conformal order statistic at ``coverage``."""

    value = float(coverage)
    if not 0.5 < value < 1.0:
        raise ValueError("one-sided admission coverage must lie in (0.5,1)")
    return int(math.ceil(value / (1.0 - value) - 1.0e-12))


def _rainfall_groups(cache: Any, names: Sequence[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def _event_ids(cache: Any, names: Sequence[str]) -> set[str]:
    return {str(cache.entry(name).event_id) for name in names}


def _contains_token(value: str, tokens: Iterable[str]) -> bool:
    lowered = str(value).lower()
    return any(str(token).strip() and str(token).lower() in lowered for token in tokens)


def _require_development_only_identifiers(
    *,
    rainfall_groups: set[str],
    event_ids: set[str],
) -> None:
    bad = sorted(
        value
        for value in rainfall_groups | event_ids
        if _contains_token(value, FORBIDDEN_EVALUATION_TOKENS)
    )
    if bad:
        raise ValueError(
            "fresh Direct-TFV admission data contains untouched-evaluation identifiers: "
            f"{bad}"
        )


def optimizer_replay_event_ids(report: dict[str, Any]) -> set[str]:
    rows = report.get("results")
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("event_id", ""))
        for row in rows
        if isinstance(row, dict) and str(row.get("event_id", "")).strip()
    }


def validate_fresh_admission_partition(
    *,
    base_cache: Any,
    step2_trainfit_d3_names: Sequence[str],
    original_audit_d3_names: Sequence[str],
    fresh_cache: Any,
    fresh_calibration_d3_names: Sequence[str],
    coverage: float,
    optimizer_replay_events: Sequence[str] = (),
    reserved_postcalibration_event_tokens: Sequence[str] = DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS,
) -> dict[str, Any]:
    """Fail closed unless calibration is fresh, large enough and role-disjoint by rainfall group.

    Rainfall group is the independent conformal unit. Event IDs are checked as an additional leakage
    guard, including substring matching for reserved T10/T20 identifiers because prepared event names
    may carry suffixes such as ``_chicago``.
    """

    train_names = sorted(str(name) for name in step2_trainfit_d3_names)
    audit_names = sorted(str(name) for name in original_audit_d3_names)
    fresh_names = sorted(str(name) for name in fresh_calibration_d3_names)
    if not train_names:
        raise ValueError("fresh admission partition has no Step2 D3 TrainFit groups")
    if not audit_names:
        raise ValueError("fresh admission partition has no original D3 holdout audit groups")
    if not fresh_names:
        raise ValueError("fresh admission partition has no fresh D3 calibration groups")
    if any(not name.startswith("D3::") for name in train_names + audit_names + fresh_names):
        raise ValueError("Direct-TFV admission partition accepts D3 groups only")

    minimum = minimum_calibration_rainfall_groups(float(coverage))
    train_rain = _rainfall_groups(base_cache, train_names)
    audit_rain = _rainfall_groups(base_cache, audit_names)
    fresh_rain = _rainfall_groups(fresh_cache, fresh_names)
    train_events = _event_ids(base_cache, train_names)
    audit_events = _event_ids(base_cache, audit_names)
    fresh_events = _event_ids(fresh_cache, fresh_names)
    replay_events = {str(value) for value in optimizer_replay_events if str(value).strip()}

    if len(fresh_rain) < minimum:
        raise ValueError(
            f"{float(coverage):.3f} fresh rainfall-group conformal calibration needs at least "
            f"{minimum} independent rainfall groups; got {len(fresh_rain)}"
        )

    overlaps = {
        "train_calibration_rainfall": sorted(train_rain & fresh_rain),
        "calibration_audit_rainfall": sorted(fresh_rain & audit_rain),
        "train_audit_rainfall": sorted(train_rain & audit_rain),
        "train_calibration_event": sorted(train_events & fresh_events),
        "calibration_audit_event": sorted(fresh_events & audit_events),
        "calibration_optimizer_replay_event": sorted(fresh_events & replay_events),
    }
    nonzero = {key: value for key, value in overlaps.items() if value}
    if nonzero:
        raise ValueError(f"Direct-TFV admission role leakage detected: {nonzero}")

    reserved_hits = sorted(
        event
        for event in fresh_events
        if _contains_token(event, reserved_postcalibration_event_tokens)
    )
    if reserved_hits:
        raise ValueError(
            "fresh Direct-TFV admission calibration uses reserved post-calibration events: "
            f"{reserved_hits}"
        )
    replay_token_hits = sorted(
        event
        for event in fresh_events
        if _contains_token(event, replay_events)
    )
    if replay_token_hits:
        raise ValueError(
            "fresh Direct-TFV admission calibration reuses optimizer-replay calibration events: "
            f"{replay_token_hits}"
        )

    _require_development_only_identifiers(rainfall_groups=fresh_rain, event_ids=fresh_events)
    return {
        "contract": DIRECT_TFV_FRESH_ADMISSION_DATA_CONTRACT,
        "development_only": True,
        "calibration_source_semantics": DIRECT_TFV_FRESH_ADMISSION_SOURCE,
        "audit_source_semantics": DIRECT_TFV_ADMISSION_AUDIT_SOURCE,
        "requested_conformal_coverage": float(coverage),
        "minimum_calibration_rainfall_groups": int(minimum),
        "step2_trainfit_group_count": len(train_names),
        "step2_trainfit_rainfall_group_count": len(train_rain),
        "fresh_calibration_group_count": len(fresh_names),
        "fresh_calibration_rainfall_group_count": len(fresh_rain),
        "original_audit_group_count": len(audit_names),
        "original_audit_rainfall_group_count": len(audit_rain),
        "step2_trainfit_rainfall_groups": sorted(train_rain),
        "fresh_calibration_rainfall_groups": sorted(fresh_rain),
        "original_audit_rainfall_groups": sorted(audit_rain),
        "fresh_calibration_event_ids": sorted(fresh_events),
        "optimizer_replay_event_ids": sorted(replay_events),
        "reserved_postcalibration_event_tokens": [
            str(value) for value in reserved_postcalibration_event_tokens
        ],
        "train_calibration_rainfall_overlap_count": 0,
        "calibration_audit_rainfall_overlap_count": 0,
        "train_audit_rainfall_overlap_count": 0,
        "train_calibration_event_overlap_count": 0,
        "calibration_audit_event_overlap_count": 0,
        "calibration_optimizer_replay_event_overlap_count": 0,
        "ready_for_admission_calibration": True,
        "scientific_role": (
            "Fresh Development D3-HOLD rainfall groups calibrate the one-sided residual bound; "
            "Step2 TrainFit groups remain model-fitting only and the original D3 holdout remains "
            "an independent cached admission audit."
        ),
    }


__all__ = [
    "DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS",
    "DIRECT_TFV_ADMISSION_AUDIT_SOURCE",
    "DIRECT_TFV_FRESH_ADMISSION_DATA_CONTRACT",
    "DIRECT_TFV_FRESH_ADMISSION_SOURCE",
    "minimum_calibration_rainfall_groups",
    "optimizer_replay_event_ids",
    "validate_fresh_admission_partition",
]
