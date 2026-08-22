"""Fixed five-event Development benchmark for Project7 operational RTC iteration.

This module deliberately separates *development steering* from Formal/Policy-Lock evidence.  The
five events are frozen once, fixed baselines are computed once and cached, and each later Proposed
controller reruns only those five events.  Authoritative SWMM TFV is the primary development signal;
Priority8 PFV remains the secondary no-control non-inferiority safety check and Global Peak is report
only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .baselines import COMPETITIVE_BASELINE_IDS
from .event_clock import inspect_prepared_event_clock


OPERATIONAL_BENCHMARK5_CONTRACT = "PROJECT7_OPERATIONAL_DEVELOPMENT_BENCHMARK5_V1"
OPERATIONAL_BASELINE_CACHE_CONTRACT = "PROJECT7_OPERATIONAL_FIXED_BASELINE_CACHE5_V1"
OPERATIONAL_COMPARISON_CONTRACT = "PROJECT7_OPERATIONAL_PROPOSED_VS_FIXED_BASELINES5_V1"
OPERATIONAL_COMPARATORS = tuple(COMPETITIVE_BASELINE_IDS)
EVENT_COUNT = 5


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class OperationalEvent:
    event_id: str
    inp_path: Path
    inp_sha256: str
    prepared_event_clock: dict[str, Any]
    rainfall_family: str | None = None
    return_period_code: int | None = None
    duration_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "inp_path": str(self.inp_path),
            "inp_sha256": self.inp_sha256,
            "prepared_event_clock": self.prepared_event_clock,
            "rainfall_family": self.rainfall_family,
            "return_period_code": self.return_period_code,
            "duration_code": self.duration_code,
        }


def build_event(
    *,
    event_id: str,
    inp_path: str | Path,
    rainfall_family: str | None = None,
    return_period_code: int | None = None,
    duration_code: int | None = None,
) -> OperationalEvent:
    path = Path(inp_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    event = str(event_id).strip()
    if not event:
        raise ValueError("operational benchmark event_id must be non-empty")
    return OperationalEvent(
        event_id=event,
        inp_path=path,
        inp_sha256=sha256_file(path),
        prepared_event_clock=inspect_prepared_event_clock(path),
        rainfall_family=(None if rainfall_family in (None, "") else str(rainfall_family)),
        return_period_code=(None if return_period_code is None else int(return_period_code)),
        duration_code=(None if duration_code is None else int(duration_code)),
    )


def write_benchmark_manifest(events: Iterable[OperationalEvent], out: str | Path, *, selection_basis: str) -> dict[str, Any]:
    values = tuple(events)
    if len(values) != EVENT_COUNT:
        raise ValueError(f"operational benchmark requires exactly {EVENT_COUNT} events")
    ids = tuple(event.event_id for event in values)
    if len(set(ids)) != EVENT_COUNT:
        raise ValueError("operational benchmark event IDs must be unique")
    shas = tuple(event.inp_sha256 for event in values)
    if len(set(shas)) != EVENT_COUNT:
        raise ValueError("operational benchmark INPs must be distinct")
    payload = {
        "contract": OPERATIONAL_BENCHMARK5_CONTRACT,
        "development_only": True,
        "event_count": EVENT_COUNT,
        "selection_basis": str(selection_basis),
        "selection_uses_proposed_performance": False,
        "events_frozen_before_iterative_proposed_comparison": True,
        "competitive_baselines": list(OPERATIONAL_COMPARATORS),
        "diagnostic_extremes_excluded_from_primary_comparison": True,
        "tfv_primary": True,
        "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
        "global_peak_role": "report_only",
        "formal_evidence": False,
        "ready_for_policy_lock": False,
        "events": [event.as_dict() for event in values],
    }
    destination = Path(out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != OPERATIONAL_BENCHMARK5_CONTRACT:
        raise ValueError("wrong Project7 operational benchmark contract")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != EVENT_COUNT:
        raise ValueError("operational benchmark must contain exactly five events")
    ids: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("operational benchmark event must be an object")
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in ids:
            raise ValueError("operational benchmark contains missing/duplicate event ID")
        ids.add(event_id)
        inp = Path(str(event.get("inp_path", ""))).resolve()
        if not inp.is_file():
            raise FileNotFoundError(inp)
        if sha256_file(inp).lower() != str(event.get("inp_sha256", "")).lower():
            raise ValueError(f"operational event INP changed after freeze: {event_id}")
    if tuple(payload.get("competitive_baselines", ())) != OPERATIONAL_COMPARATORS:
        raise ValueError("operational benchmark comparator set drifted")
    return payload


__all__ = [
    "EVENT_COUNT",
    "OPERATIONAL_BASELINE_CACHE_CONTRACT",
    "OPERATIONAL_BENCHMARK5_CONTRACT",
    "OPERATIONAL_COMPARATORS",
    "OPERATIONAL_COMPARISON_CONTRACT",
    "OperationalEvent",
    "build_event",
    "load_benchmark_manifest",
    "sha256_file",
    "write_benchmark_manifest",
]
