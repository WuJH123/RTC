"""Historical exact-return supervision discovery for Project7 Step3 V26.

The data contract is intentionally permissive about *provenance* and strict about *semantics*:
old Train/Validation/Calibration labels, prior Step3 versions, and Step1/Step2 exposure are metadata,
not exclusion rules.  Reuse is controlled by a newly frozen leakage-group Train/Validation/Test split.
Only authoritative full candidate-vs-HOLD policy-return truth is promoted to the Step3 target; H120
or Step2 auxiliary targets are never silently relabelled as full-return truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


HISTORICAL_SUPERVISION_CONTRACT = "PROJECT7_STEP3_V26_HISTORICAL_EXACT_RETURN_SUPERVISION_V2"
SUPPORTED_SUFFIXES = {".jsonl", ".json", ".npz"}
EXACT_TRUTH_FIELD = "true_policy_return_delta_tfv_m3"
CONTEXT_KEYS = ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow")
TARGET_KEYS = ("candidate_target", "candidate_first_target")


@dataclass
class HistoricalCandidateRecord:
    row: dict[str, Any]
    source_path: Path
    source_index: int
    embedded_context: dict[str, np.ndarray] | None = None
    embedded_target: np.ndarray | None = None


@dataclass
class CanonicalCandidateRecord:
    row: dict[str, Any]
    context: dict[str, np.ndarray]
    target: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _array_bytes(value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return (
        str(array.dtype).encode("utf-8")
        + b"|"
        + json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8")
        + b"|"
        + array.tobytes(order="C")
    )


def causal_context_sha256(context: dict[str, np.ndarray]) -> str:
    return _sha_bytes(
        key.encode("utf-8") + b"\0" + _array_bytes(np.asarray(context[key]))
        for key in CONTEXT_KEYS
    )


def action_sha256(target: np.ndarray) -> str:
    target32 = np.ascontiguousarray(np.asarray(target, dtype=np.float32).reshape(-1))
    return hashlib.sha256(target32.tobytes(order="C")).hexdigest()


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar(value.item())
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(value: Any) -> str:
    value = _scalar(value)
    if value is None:
        return ""
    return str(value).strip()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(_scalar(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def exact_truth(row: dict[str, Any]) -> float | None:
    direct = _finite_float(row.get(EXACT_TRUTH_FIELD))
    if direct is not None:
        return direct
    candidate = _finite_float(row.get("candidate_branch_tfv_m3"))
    hold = _finite_float(row.get("hold_branch_tfv_m3"))
    if candidate is not None and hold is not None:
        return candidate - hold
    return None


def _looks_like_candidate_row(row: dict[str, Any]) -> bool:
    if exact_truth(row) is None:
        return False
    return any(key in row for key in TARGET_KEYS) or bool(_text(row.get("context_npz"))) or bool(
        _text(row.get("candidate_source"))
    )


def _walk_json(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if _looks_like_candidate_row(value):
            yield dict(value)
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _json_records(path: Path) -> list[HistoricalCandidateRecord]:
    records: list[HistoricalCandidateRecord] = []
    if path.suffix.lower() == ".jsonl":
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if isinstance(value, dict) and _looks_like_candidate_row(value):
                records.append(HistoricalCandidateRecord(dict(value), path, line_number))
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    for index, row in enumerate(_walk_json(payload)):
        records.append(HistoricalCandidateRecord(row, path, index))
    return records


def _npz_row_value(array: np.ndarray, index: int, count: int) -> Any:
    if array.ndim == 0:
        return _scalar(array)
    if int(array.shape[0]) == count:
        return _scalar(array[index])
    if array.size == 1:
        return _scalar(array.reshape(-1)[0])
    return None


def _npz_records(path: Path) -> list[HistoricalCandidateRecord]:
    data = np.load(path, allow_pickle=False)
    try:
        if EXACT_TRUTH_FIELD in data:
            truth_array = np.asarray(data[EXACT_TRUTH_FIELD]).reshape(-1)
        elif "candidate_branch_tfv_m3" in data and "hold_branch_tfv_m3" in data:
            truth_array = (
                np.asarray(data["candidate_branch_tfv_m3"], dtype=np.float64).reshape(-1)
                - np.asarray(data["hold_branch_tfv_m3"], dtype=np.float64).reshape(-1)
            )
        else:
            return []
        count = int(truth_array.size)
        if count <= 0 or "candidate_target" not in data:
            return []
        targets = np.asarray(data["candidate_target"])
        if targets.ndim < 2 or int(targets.shape[0]) != count:
            return []
        if any(key not in data for key in CONTEXT_KEYS):
            return []

        metadata_keys = (
            "rainfall_group",
            "event_id",
            "query_set_id",
            "candidate_source",
            "data_role",
            "source_data_role",
            "decision_index",
            "decision_elapsed_seconds",
            "continuation_policy_sha256",
            "prefix_sha256",
            "supervisory_mask_sha256",
            "candidate_portfolio_contract",
            "first_move_changed_facility_count",
            "base_step2_h10_score_m3",
        )
        out: list[HistoricalCandidateRecord] = []
        for index in range(count):
            truth = _finite_float(truth_array[index])
            if truth is None:
                continue
            row: dict[str, Any] = {EXACT_TRUTH_FIELD: truth}
            for key in metadata_keys:
                if key not in data:
                    continue
                value = _npz_row_value(np.asarray(data[key]), index, count)
                if value is not None and not isinstance(value, np.ndarray):
                    row[key] = value
            context = {
                key: np.asarray(data[key][index] if np.asarray(data[key]).shape[0] == count else data[key]).copy()
                for key in CONTEXT_KEYS
            }
            target = np.asarray(targets[index], dtype=np.float32).reshape(-1).copy()
            out.append(
                HistoricalCandidateRecord(
                    row=row,
                    source_path=path,
                    source_index=index,
                    embedded_context=context,
                    embedded_target=target,
                )
            )
        return out
    finally:
        data.close()


def read_candidate_records(path: str | Path) -> list[HistoricalCandidateRecord]:
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return []
    if suffix in {".json", ".jsonl"}:
        return _json_records(path)
    return _npz_records(path)


def discover_candidate_assets(root: str | Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    assets: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                if read_candidate_records(path):
                    assets.append(path.resolve())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
    return assets


def _candidate_target_from_row(row: dict[str, Any]) -> np.ndarray | None:
    for key in TARGET_KEYS:
        value = row.get(key)
        if value is None:
            continue
        try:
            target = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if target.shape == (109,) and np.isfinite(target).all():
            return target.copy()
    return None


def _valid_context(context: dict[str, np.ndarray]) -> bool:
    try:
        state = np.asarray(context["current_state"])
        rain = np.asarray(context["rainfall_scenarios"])
        active = np.asarray(context["active_target"]).reshape(-1)
        flow = np.asarray(context["previous_actuator_flow"]).reshape(-1)
    except (KeyError, ValueError):
        return False
    return (
        state.ndim in (2, 3)
        and rain.ndim in (4, 5)
        and active.shape == (109,)
        and flow.shape == (109,)
        and all(np.isfinite(np.asarray(value)).all() for value in (state, rain, active, flow))
    )


def _load_context(path: Path) -> dict[str, np.ndarray] | None:
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        if any(key not in data for key in CONTEXT_KEYS):
            return None
        context = {key: np.asarray(data[key]).copy() for key in CONTEXT_KEYS}
        if context["current_state"].ndim == 3 and context["current_state"].shape[0] == 1:
            context["current_state"] = context["current_state"][0]
        if context["rainfall_scenarios"].ndim == 5 and context["rainfall_scenarios"].shape[0] == 1:
            context["rainfall_scenarios"] = context["rainfall_scenarios"][0]
        context["active_target"] = context["active_target"].reshape(-1)
        context["previous_actuator_flow"] = context["previous_actuator_flow"].reshape(-1)
        return context if _valid_context(context) else None
    finally:
        data.close()


def _target_from_context(path: Path) -> np.ndarray | None:
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        if "candidate_target" not in data:
            return None
        target = np.asarray(data["candidate_target"], dtype=np.float32).reshape(-1)
        return target.copy() if target.shape == (109,) and np.isfinite(target).all() else None
    finally:
        data.close()


class ContextResolver:
    """Resolve historical context references without treating stale absolute paths as lost data."""

    def __init__(self, *, study_root: str | Path | None) -> None:
        self.study_root = Path(study_root).resolve() if study_root else None
        self._basename_cache: dict[str, list[Path]] = {}

    def _basename_matches(self, name: str) -> list[Path]:
        if name in self._basename_cache:
            return self._basename_cache[name]
        if self.study_root is None:
            matches: list[Path] = []
        else:
            matches = [path.resolve() for path in self.study_root.rglob(name) if path.is_file()]
        self._basename_cache[name] = matches
        return matches

    def resolve(self, row: dict[str, Any], source_path: Path) -> Path | None:
        raw = _text(row.get("context_npz"))
        if not raw:
            return None
        requested = Path(raw)
        candidates: list[Path] = []
        if requested.is_absolute():
            candidates.append(requested)
        else:
            candidates.extend((source_path.parent / requested, requested))
        if requested.name:
            candidates.extend(self._basename_matches(requested.name))
        expected = _text(row.get("context_npz_sha256")).lower()
        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            if len(expected) == 64 and sha256_file(candidate).lower() != expected:
                continue
            return candidate
        return None


def canonicalize_record(
    record: HistoricalCandidateRecord,
    *,
    resolver: ContextResolver,
) -> tuple[CanonicalCandidateRecord | None, str]:
    truth = exact_truth(record.row)
    if truth is None:
        return None, "missing_exact_full_policy_return_truth"

    context: dict[str, np.ndarray] | None = None
    target: np.ndarray | None = record.embedded_target
    original_context_path = ""
    original_context_sha = _text(record.row.get("context_npz_sha256")).lower()
    if record.embedded_context is not None:
        context = {key: np.asarray(value).copy() for key, value in record.embedded_context.items()}
        if context["current_state"].ndim == 3 and context["current_state"].shape[0] == 1:
            context["current_state"] = context["current_state"][0]
        if context["rainfall_scenarios"].ndim == 5 and context["rainfall_scenarios"].shape[0] == 1:
            context["rainfall_scenarios"] = context["rainfall_scenarios"][0]
        context["active_target"] = context["active_target"].reshape(-1)
        context["previous_actuator_flow"] = context["previous_actuator_flow"].reshape(-1)
    else:
        context_path = resolver.resolve(record.row, record.source_path)
        if context_path is None:
            return None, "missing_causal_context"
        context = _load_context(context_path)
        if context is None:
            return None, "invalid_causal_context"
        original_context_path = str(context_path)
        if target is None:
            target = _target_from_context(context_path)

    if not _valid_context(context):
        return None, "invalid_causal_context"
    if target is None:
        target = _candidate_target_from_row(record.row)
    if target is None or target.shape != (109,) or not np.isfinite(target).all():
        return None, "missing_or_invalid_candidate_target"

    direct = _finite_float(record.row.get(EXACT_TRUTH_FIELD))
    candidate_branch = _finite_float(record.row.get("candidate_branch_tfv_m3"))
    hold_branch = _finite_float(record.row.get("hold_branch_tfv_m3"))
    if direct is not None and candidate_branch is not None and hold_branch is not None:
        if abs((candidate_branch - hold_branch) - direct) > 1.0e-6:
            return None, "exact_return_arithmetic_mismatch"

    for flag in (
        "same_prefix_verified",
        "same_continuation_policy_verified",
        "candidate_target_write_readback_verified",
        "target_write_readback_verified",
    ):
        if flag in record.row and record.row.get(flag) is False:
            return None, f"failed_{flag}"

    group = _text(record.row.get("rainfall_group"))
    event = _text(record.row.get("event_id"))
    query = _text(record.row.get("query_set_id"))
    if not group and not event:
        return None, "missing_rainfall_or_event_group"
    if not query:
        query = hashlib.sha256(
            ("|".join((group, event, causal_context_sha256(context)))).encode("utf-8")
        ).hexdigest()

    source = _text(record.row.get("candidate_source")) or "HISTORICAL_EXACT_ACTION"
    context_fingerprint = causal_context_sha256(context)
    target_hash = action_sha256(target)
    continuation = _text(record.row.get("continuation_policy_sha256")).lower()
    dedup_scope = continuation or query
    out = dict(record.row)
    out.update(
        {
            "historical_supervision_contract": HISTORICAL_SUPERVISION_CONTRACT,
            EXACT_TRUTH_FIELD: float(truth),
            "rainfall_group": group or event,
            "event_id": event,
            "query_set_id": query,
            "candidate_source": source,
            "candidate_target": np.asarray(target, dtype=np.float32).tolist(),
            "candidate_first_target_sha256": target_hash,
            "causal_context_fingerprint_sha256": context_fingerprint,
            "dedup_continuation_or_query": dedup_scope,
            "historical_source_path": str(record.source_path),
            "historical_source_index": int(record.source_index),
            "historical_source_format": record.source_path.suffix.lower().lstrip("."),
            "historical_original_data_role": _text(
                record.row.get("data_role", record.row.get("source_data_role", ""))
            ),
            "historical_original_context_path": original_context_path,
            "historical_original_context_sha256": original_context_sha,
            "h120_truth_used_for_training": False,
            "prior_version_visibility_excludes_training": False,
            "step1_step2_prior_exposure_excludes_training": False,
        }
    )
    return CanonicalCandidateRecord(out, context, np.asarray(target, dtype=np.float32)), "eligible"


def leakage_components(records: list[CanonicalCandidateRecord]) -> dict[int, str]:
    """Build connected leakage groups from rainfall/event/context identities."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    identity_owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        row = record.row
        identities = []
        for kind, value in (
            ("rainfall", _text(row.get("rainfall_group"))),
            ("event", _text(row.get("event_id"))),
            ("context", _text(row.get("causal_context_fingerprint_sha256"))),
        ):
            if value:
                identities.append((kind, value.lower()))
        for identity in identities:
            previous = identity_owner.get(identity)
            if previous is None:
                identity_owner[identity] = index
            else:
                union(index, previous)

    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(find(index), []).append(index)
    result: dict[int, str] = {}
    for root, indices in members.items():
        labels: list[str] = []
        for index in indices:
            row = records[index].row
            labels.extend(
                value
                for value in (
                    _text(row.get("rainfall_group")),
                    _text(row.get("event_id")),
                    _text(row.get("causal_context_fingerprint_sha256")),
                )
                if value
            )
        component = hashlib.sha256("|".join(sorted(set(labels))).encode("utf-8")).hexdigest()
        for index in indices:
            result[index] = component
    return result


def deterministic_split(
    component_ids: Iterable[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    groups = sorted(
        set(component_ids),
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode("utf-8")).hexdigest(),
    )
    if len(groups) < 3:
        raise ValueError("at least three independent leakage groups are required for Train/Validation/Test")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must lie in (0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train + Validation must leave a non-empty Test split")
    n = len(groups)
    n_train = min(n - 2, max(1, int(round(n * train_fraction))))
    n_validation = min(n - n_train - 1, max(1, int(round(n * validation_fraction))))
    result: dict[str, str] = {}
    for group in groups[:n_train]:
        result[group] = "train"
    for group in groups[n_train : n_train + n_validation]:
        result[group] = "validation"
    for group in groups[n_train + n_validation :]:
        result[group] = "test"
    return result


def canonical_dedup_key(record: CanonicalCandidateRecord) -> tuple[str, str, str]:
    row = record.row
    return (
        _text(row["causal_context_fingerprint_sha256"]).lower(),
        _text(row["candidate_first_target_sha256"]).lower(),
        _text(row["dedup_continuation_or_query"]).lower(),
    )


__all__ = [
    "CanonicalCandidateRecord",
    "ContextResolver",
    "EXACT_TRUTH_FIELD",
    "HISTORICAL_SUPERVISION_CONTRACT",
    "HistoricalCandidateRecord",
    "SUPPORTED_SUFFIXES",
    "action_sha256",
    "canonical_dedup_key",
    "canonicalize_record",
    "causal_context_sha256",
    "deterministic_split",
    "discover_candidate_assets",
    "exact_truth",
    "leakage_components",
    "read_candidate_records",
    "sha256_file",
]
