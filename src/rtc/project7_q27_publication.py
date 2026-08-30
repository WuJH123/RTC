"""Publication evidence utilities for the policy-consistent Project7 Q27 controller.

The module is deliberately independent of SWMM.  It defines controller-outcome exposure semantics,
freezes an outcome-unexposed event panel, and computes paired event-level TFV statistics.  Prepared
rainfall, Step1 training and Step2 training are not controller-outcome exposure by themselves.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PUBLICATION_PROTOCOL_CONTRACT = "PROJECT7_Q27_POLICY_CONSISTENT_PUBLICATION_PROTOCOL_V1"
POLICY_LOCK_CONTRACT = "PROJECT7_Q27_PUBLICATION_POLICY_LOCK_V1"
STATISTICS_CONTRACT = "PROJECT7_Q27_PUBLICATION_STATISTICS_V1"
BLOCKING_OUTCOME_EXPOSURES = frozenset(
    {
        "POLICY_RETURN_SUPERVISION",
        "MODEL_SELECTION",
        "DEVELOPMENT_STEERING",
        "OPENED_FORMAL_OUTCOME",
        "PRIOR_FINAL_OUTCOME",
    }
)
NONBLOCKING_EXPOSURES = frozenset({"PREPARED_ONLY", "STEP1_TRAINING", "STEP2_TRAINING"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_locked_event_provenance(
    rows: Sequence[Mapping[str, Any]],
    selected_events: Sequence[str],
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Bind each locked event to its registry row and verified prepared INP."""
    indexed: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        eid = event_id(raw_row)
        if not eid:
            continue
        row = dict(raw_row)
        previous = indexed.get(eid)
        if previous is not None and canonical_sha256(previous) != canonical_sha256(row):
            raise ValueError(f"prepared registry has conflicting rows for event {eid}")
        indexed[eid] = row

    root = Path(base_dir).resolve() if base_dir is not None else None
    provenance: list[dict[str, Any]] = []
    for eid in selected_events:
        if eid not in indexed:
            raise ValueError(f"selected event missing from prepared registry: {eid}")
        row = indexed[eid]
        raw_path = str(row.get("inp_path", "")).strip()
        if not raw_path:
            raise ValueError(f"prepared registry lacks INP path for event {eid}")
        inp_path = Path(raw_path)
        if not inp_path.is_absolute() and root is not None:
            inp_path = root / inp_path
        inp_path = inp_path.resolve()
        if not inp_path.is_file():
            raise FileNotFoundError(f"prepared INP missing for event {eid}: {inp_path}")
        actual_sha = sha256_file(inp_path)
        recorded_sha = str(
            row.get("prepared_inp_sha256", row.get("inp_sha256", ""))
        ).strip().lower()
        if recorded_sha and recorded_sha != actual_sha:
            raise ValueError(f"INP SHA256 mismatch for event {eid}")
        bound = dict(row)
        bound["event_id"] = eid
        bound["inp_path"] = str(inp_path)
        bound["inp_sha256"] = actual_sha
        bound["registry_inp_sha256"] = recorded_sha or actual_sha
        bound["forcing_metadata"] = {
            key: value
            for key, value in row.items()
            if key not in {"event_id", "event", "rainfall_group", "inp_path", "prepared_inp_sha256", "inp_sha256"}
        }
        provenance.append(bound)
    return provenance


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def event_id(row: Mapping[str, Any]) -> str:
    for key in ("event_id", "event", "rainfall_group"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def exposure_index(exposures: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in exposures:
        eid = event_id(row)
        kind = str(row.get("kind", row.get("exposure_kind", ""))).strip().upper()
        if not eid or not kind:
            continue
        out.setdefault(eid, set()).add(kind)
    return out


def select_outcome_unexposed_events(
    prepared_event_ids: Sequence[str], exposures: Iterable[Mapping[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    prepared = sorted({str(value).strip() for value in prepared_event_ids if str(value).strip()})
    index = exposure_index(exposures)
    selected: list[str] = []
    blocked: dict[str, list[str]] = {}
    for eid in prepared:
        reasons = sorted(index.get(eid, set()) & BLOCKING_OUTCOME_EXPOSURES)
        if reasons:
            blocked[eid] = reasons
        else:
            selected.append(eid)
    return selected, {
        "prepared_event_count": len(prepared),
        "eligible_event_count": len(selected),
        "blocked_event_count": len(blocked),
        "blocked_events": blocked,
        "selection_used_hydraulic_or_controller_performance": False,
        "prepared_step1_step2_exposure_is_not_a_blocking_outcome": True,
    }


def paired_reduction_pct(proposed: Sequence[float], baseline: Sequence[float]) -> np.ndarray:
    p = np.asarray(proposed, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if p.size < 1 or p.size != b.size or not np.isfinite(p).all() or not np.isfinite(b).all():
        raise ValueError("paired TFV arrays must be aligned finite non-empty vectors")
    if np.any(b <= 0.0):
        raise ValueError("baseline TFV must be positive for percentage reduction")
    return 100.0 * (b - p) / b


def bootstrap_mean_ci(
    values: Sequence[float], *, repetitions: int = 10000, seed: int = 20260830
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 1 or not np.isfinite(x).all() or repetitions < 100:
        raise ValueError("bootstrap requires finite values and at least 100 repetitions")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, x.size, size=(int(repetitions), x.size))
    means = x[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_tfv_statistics(
    proposed: Sequence[float],
    baseline: Sequence[float],
    *,
    repetitions: int = 10000,
    seed: int = 20260830,
) -> dict[str, Any]:
    p = np.asarray(proposed, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    reduction = paired_reduction_pct(p, b)
    low, high = bootstrap_mean_ci(reduction, repetitions=repetitions, seed=seed)
    mean = float(reduction.mean())
    if low > 0.0:
        classification = "SUPERIOR"
    elif high < 0.0:
        classification = "INFERIOR"
    else:
        classification = "INCONCLUSIVE_OR_NEUTRAL"
    return {
        "event_count": int(p.size),
        "wins": int(np.sum(p < b)),
        "event_balanced_mean_reduction_pct": mean,
        "median_reduction_pct": float(np.median(reduction)),
        "aggregate_volume_reduction_pct": float(100.0 * (b.sum() - p.sum()) / b.sum()),
        "paired_bootstrap_mean_reduction_95ci_pct": [low, high],
        "claim_classification": classification,
    }


def validate_final_table_event_set(rows: Sequence[Mapping[str, Any]], locked_events: Sequence[str]) -> None:
    actual = [event_id(row) for row in rows]
    if any(not value for value in actual):
        raise ValueError("final result table contains a row without event identity")
    if len(actual) != len(set(actual)):
        raise ValueError("final result table contains duplicate events")
    if set(actual) != set(locked_events):
        missing = sorted(set(locked_events) - set(actual))
        extra = sorted(set(actual) - set(locked_events))
        raise ValueError(f"final result event set differs from Policy Lock; missing={missing}, extra={extra}")


def exposure_counts(exposures: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in exposures:
        kind = str(row.get("kind", row.get("exposure_kind", ""))).strip().upper()
        if kind:
            counts[kind] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "BLOCKING_OUTCOME_EXPOSURES",
    "NONBLOCKING_EXPOSURES",
    "POLICY_LOCK_CONTRACT",
    "PUBLICATION_PROTOCOL_CONTRACT",
    "STATISTICS_CONTRACT",
    "build_locked_event_provenance",
    "bootstrap_mean_ci",
    "canonical_sha256",
    "event_id",
    "exposure_counts",
    "exposure_index",
    "paired_reduction_pct",
    "paired_tfv_statistics",
    "select_outcome_unexposed_events",
    "sha256_file",
    "validate_final_table_event_set",
]
