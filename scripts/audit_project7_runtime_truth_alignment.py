"""Join Project7 runtime decisions to existing exact-return truth without running SWMM.

Formal-strength matches require the same event/decision time, causal-context fingerprint, selected
action SHA and continuation-policy SHA.  Historical V27-V29 logs lack some of these identities; the
auditor may report an event/time/action-only legacy match, but never upgrades it to formal evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from rtc.project7_policy_consistency import normalize_sha256
from rtc.project7_v26_historical_supervision import action_sha256, exact_truth


CONTRACT = "PROJECT7_RUNTIME_TO_EXACT_RETURN_ALIGNMENT_AUDIT_V1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _runtime_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runtime must be EVENT_ID=DECISIONS_JSONL")
    event_id, raw_path = value.split("=", 1)
    event_id = event_id.strip()
    path = Path(raw_path.strip()).resolve()
    if not event_id or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid runtime spec: {value}")
    return event_id, path


def _truth_continuation(row: dict[str, Any]) -> str:
    return normalize_sha256(
        row.get("resolved_continuation_policy_sha256", row.get("continuation_policy_sha256"))
    )


def _prefix(row: dict[str, Any]) -> str:
    for key in (
        "recorded_prefix_action_sha256",
        "prefix_sha256",
        "runtime_recorded_prefix_action_sha256",
    ):
        value = normalize_sha256(row.get(key))
        if value:
            return value
    return ""


def _runtime_action_class(diagnostics: dict[str, Any]) -> str:
    """Return the actual executed class, preferring versioned structured telemetry.

    Older V27/V28 logs can retain a stale calibrated/fallback field after the
    runtime has already selected an action.  A versioned action-class field is
    emitted by the runtime at the point where the executed decision is known,
    so it has precedence.  Unknown values are not coerced into ACTION/HOLD.
    """
    for key in (
        "v28_action_class",
        "v27_action_class",
        "action_class",
        "calibrated_runtime_action_class",
    ):
        value = diagnostics.get(key)
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"ACTION", "HOLD"}:
                return normalized
    return "UNKNOWN"


def _runtime_action_hash(row: dict[str, Any], actuator_ids: tuple[str, ...]) -> str:
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict):
        candidate_telemetry = diagnostics.get("v28_candidate_telemetry")
        if isinstance(candidate_telemetry, list):
            for candidate in candidate_telemetry:
                if not isinstance(candidate, dict) or not candidate.get("candidate_selected"):
                    continue
                for key in (
                    "supported_target_sha256",
                    "candidate_first_target_sha256",
                    "action_sha256",
                ):
                    value = normalize_sha256(candidate.get(key))
                    if value:
                        return value
        value = normalize_sha256(diagnostics.get("runtime_selected_action_sha256"))
        if value:
            return value
    settings = row.get("settings")
    if not isinstance(settings, dict) or set(settings) != set(actuator_ids):
        return ""
    target = np.asarray([float(settings[aid]) for aid in actuator_ids], dtype=np.float32)
    return action_sha256(target)


def _truth_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(row.get("event_id", "")).strip().casefold(),
        int(row.get("decision_elapsed_seconds", -1)),
        normalize_sha256(row.get("causal_context_fingerprint_sha256")),
        normalize_sha256(row.get("candidate_first_target_sha256")),
        _truth_continuation(row),
    )


def _legacy_key(event_id: str, elapsed: int, action_hash: str) -> tuple[str, int, str]:
    return event_id.strip().casefold(), int(elapsed), action_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-jsonl", required=True)
    parser.add_argument("--runtime", action="append", required=True, type=_runtime_spec)
    parser.add_argument("--supervisory-control", required=True)
    parser.add_argument("--target-continuation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target = normalize_sha256(args.target_continuation)
    if not target:
        raise ValueError("target continuation must be one SHA256")
    control = json.loads(Path(args.supervisory_control).read_text(encoding="utf-8"))
    actuator_ids = tuple(str(value) for value in control.get("actuator_ids", ()))
    if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
        raise ValueError("supervisory-control artifact must provide the canonical 109 actuator order")

    truth_rows = [
        row
        for row in _read_jsonl(Path(args.truth_jsonl).resolve())
        if _truth_continuation(row) == target
    ]
    full_index: defaultdict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    legacy_index: defaultdict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in truth_rows:
        key = _truth_key(row)
        if key[0] and key[1] >= 0 and key[2] and key[3] and key[4]:
            full_index[key].append(row)
        if key[0] and key[1] >= 0 and key[3]:
            legacy_index[(key[0], key[1], key[3])].append(row)

    detail: list[dict[str, Any]] = []
    match_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    exact_returns_by_match: defaultdict[str, list[float]] = defaultdict(list)

    for event_id, path in args.runtime:
        for decision_index, row in enumerate(_read_jsonl(path)):
            elapsed = int(row.get("elapsed_seconds", -1))
            diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
            action_hash = _runtime_action_hash(row, actuator_ids)
            context_hash = normalize_sha256(
                diagnostics.get("runtime_causal_context_fingerprint_sha256")
            )
            continuation = normalize_sha256(diagnostics.get("runtime_continuation_policy_sha256"))
            runtime_prefix = normalize_sha256(
                diagnostics.get("runtime_recorded_prefix_action_sha256")
            )
            action_class = _runtime_action_class(diagnostics)
            class_counts[action_class] += 1

            full_candidates: list[dict[str, Any]] = []
            if action_hash and context_hash and continuation:
                full_candidates = full_index.get(
                    (
                        event_id.casefold(),
                        elapsed,
                        context_hash,
                        action_hash,
                        continuation,
                    ),
                    [],
                )
            legacy_candidates = legacy_index.get(
                _legacy_key(event_id, elapsed, action_hash), []
            ) if action_hash else []

            match_kind = "NO_MATCH"
            selected_truth: dict[str, Any] | None = None
            if len(full_candidates) == 1:
                candidate = full_candidates[0]
                truth_prefix = _prefix(candidate)
                if runtime_prefix and truth_prefix and runtime_prefix != truth_prefix:
                    match_kind = "FULL_IDENTITY_PREFIX_MISMATCH"
                else:
                    match_kind = "FULL_IDENTITY_MATCH"
                    selected_truth = candidate
            elif len(full_candidates) > 1:
                match_kind = "AMBIGUOUS_FULL_IDENTITY"
            elif len(legacy_candidates) == 1:
                match_kind = "LEGACY_EVENT_TIME_ACTION_ONLY"
                selected_truth = legacy_candidates[0]
            elif len(legacy_candidates) > 1:
                match_kind = "AMBIGUOUS_LEGACY_EVENT_TIME_ACTION"

            match_counts[match_kind] += 1
            truth_value = None if selected_truth is None else exact_truth(selected_truth)
            if truth_value is not None:
                exact_returns_by_match[match_kind].append(float(truth_value))
            detail.append(
                {
                    "event_id": event_id,
                    "decision_index": decision_index,
                    "elapsed_seconds": elapsed,
                    "runtime_action_class": action_class,
                    "runtime_action_sha256": action_hash,
                    "runtime_context_sha256": context_hash,
                    "runtime_continuation_sha256": continuation,
                    "runtime_prefix_sha256": runtime_prefix,
                    "match_kind": match_kind,
                    "formal_identity_match": match_kind == "FULL_IDENTITY_MATCH",
                    "legacy_match_is_not_formal_evidence": match_kind.startswith("LEGACY_"),
                    "true_policy_return_delta_tfv_m3": truth_value,
                    "truth_candidate_source": (
                        "" if selected_truth is None else str(selected_truth.get("candidate_source", ""))
                    ),
                    "truth_prefix_sha256": "" if selected_truth is None else _prefix(selected_truth),
                }
            )

    summaries: dict[str, Any] = {}
    for kind, values in sorted(exact_returns_by_match.items()):
        array = np.asarray(values, dtype=np.float64)
        summaries[kind] = {
            "count_with_truth": int(array.size),
            "mean_exact_return_m3": float(array.mean()) if array.size else None,
            "beneficial_count": int(np.sum(array < 0.0)),
            "nonbeneficial_count": int(np.sum(array >= 0.0)),
        }

    report = {
        "contract": CONTRACT,
        "target_continuation_policy_sha256": target,
        "truth_rows_in_target_continuation": len(truth_rows),
        "runtime_decisions": len(detail),
        "runtime_action_class_counts": dict(sorted(class_counts.items())),
        "match_counts": dict(sorted(match_counts.items())),
        "matched_truth_summaries": summaries,
        "formal_match_requires_context_action_continuation": True,
        "prefix_is_verified_when_present_on_both_sides": True,
        "legacy_event_time_action_match_is_formal_evidence": False,
        "candidate_coverage_regret_identified": False,
        "hold_boundary_regret_identified": False,
        "new_swmm_runs": 0,
        "training_performed": False,
        "details": detail,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
