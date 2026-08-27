"""Composite-lineage causal context recovery for Project7 V27.

V26 stopped at the first ambiguous historical identity (often query_set_id), leaving 94 exact-return
rows unusable even when a more specific prefix/event/action identity could disambiguate them.  V27
intersects all available strong identities before falling back to weaker query/group identities.
It never copies a peer candidate action: only the causal context is recovered.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .project7_v26_historical_supervision import (
    ContextResolver,
    HistoricalCandidateRecord,
    action_sha256,
    causal_context_sha256,
    normalize_context,
)


V27_CONTEXT_RECOVERY_CONTRACT = "PROJECT7_V27_COMPOSITE_LINEAGE_CAUSAL_CONTEXT_RECOVERY_V1"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _text(value.reshape(-1)[0])
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    return str(value).strip()


def _row_target_sha(record: HistoricalCandidateRecord) -> str:
    explicit = _text(record.row.get("candidate_first_target_sha256")).lower()
    if len(explicit) == 64:
        return explicit
    value = record.embedded_target
    if value is None:
        value = record.row.get("candidate_target", record.row.get("candidate_first_target"))
    if value is None:
        return ""
    try:
        target = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return ""
    if target.shape != (109,) or not np.isfinite(target).all():
        return ""
    return action_sha256(target)


def _context_for(record: HistoricalCandidateRecord, resolver: ContextResolver) -> dict[str, np.ndarray] | None:
    if record.embedded_context is not None:
        return normalize_context(record.embedded_context)
    context, _ = resolver.load_context(record.row, record.source_path)
    return context


def _identities(record: HistoricalCandidateRecord) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    row = record.row
    query = _text(row.get("query_set_id")).lower()
    prefix = _text(row.get("prefix_sha256")).lower()
    continuation = _text(row.get("continuation_policy_sha256")).lower()
    event = _text(row.get("event_id")).lower()
    group = _text(row.get("rainfall_group")).lower()
    decision = _text(row.get("decision_index"))
    elapsed = _text(row.get("decision_elapsed_seconds"))
    context_file_sha = _text(row.get("context_npz_sha256")).lower()
    action = _row_target_sha(record)

    strong: list[tuple[str, str]] = []
    weak: list[tuple[str, str]] = []
    if len(context_file_sha) == 64:
        strong.append(("context_file_sha", context_file_sha))
    if prefix and continuation:
        strong.append(("prefix_continuation", f"{prefix}|{continuation}"))
    if prefix:
        strong.append(("prefix", prefix))
    if event and decision and elapsed:
        strong.append(("event_decision_elapsed", f"{event}|{decision}|{elapsed}"))
    if event and decision:
        strong.append(("event_decision", f"{event}|{decision}"))
    if event and elapsed:
        strong.append(("event_elapsed", f"{event}|{elapsed}"))
    if query and action:
        strong.append(("query_action", f"{query}|{action}"))
    if group and decision and elapsed:
        strong.append(("group_decision_elapsed", f"{group}|{decision}|{elapsed}"))
    if query:
        weak.append(("query", query))
    if group and query:
        weak.append(("group_query", f"{group}|{query}"))
    if event and query:
        weak.append(("event_query", f"{event}|{query}"))
    return strong, weak


def _intersection(
    identities: Sequence[tuple[str, str]],
    index: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]],
) -> tuple[set[str] | None, list[str]]:
    candidate_sets: list[set[str]] = []
    used: list[str] = []
    for identity in identities:
        candidates = index.get(identity)
        if not candidates:
            continue
        candidate_sets.append(set(candidates))
        used.append(identity[0])
    if not candidate_sets:
        return None, used
    result = set(candidate_sets[0])
    for values in candidate_sets[1:]:
        result &= values
    return result, used


def recover_missing_contexts_v27(
    records: Sequence[HistoricalCandidateRecord],
    *,
    resolver: ContextResolver,
    references: Sequence[HistoricalCandidateRecord] = (),
) -> dict[str, Any]:
    """Recover missing causal contexts by composite lineage intersection.

    A unique context can be recovered from strong identities even when query_set_id alone maps to
    several historical contexts.  If strong identities conflict or still leave multiple fingerprints,
    the row remains ambiguous rather than being guessed.
    """
    all_sources = list(records) + list(references)
    context_cache: dict[int, dict[str, np.ndarray] | None] = {}
    strong_index: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    weak_index: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for source in all_sources:
        context = _context_for(source, resolver)
        context_cache[id(source)] = context
        if context is None:
            continue
        fingerprint = causal_context_sha256(context)
        strong, weak = _identities(source)
        for identity in strong:
            strong_index.setdefault(identity, {})[fingerprint] = context
        for identity in weak:
            weak_index.setdefault(identity, {})[fingerprint] = context

    repaired = 0
    ambiguous = 0
    no_reference = 0
    already_available = 0
    repaired_by: dict[str, int] = {}
    for record in records:
        if context_cache.get(id(record)) is not None:
            already_available += 1
            continue
        # Recovery is not allowed to invent/borrow the candidate action.  Require the record itself
        # to identify a valid action before context backfill is attempted.
        if not _row_target_sha(record):
            no_reference += 1
            continue
        strong, weak = _identities(record)
        strong_candidates, strong_used = _intersection(strong, strong_index)
        chosen_fingerprint = ""
        chosen_context: dict[str, np.ndarray] | None = None
        used_kind = ""
        if strong_candidates is not None:
            if len(strong_candidates) == 1:
                chosen_fingerprint = next(iter(strong_candidates))
                # Any strong identity containing the chosen fingerprint can supply identical context.
                for identity in strong:
                    candidates = strong_index.get(identity, {})
                    if chosen_fingerprint in candidates:
                        chosen_context = candidates[chosen_fingerprint]
                        used_kind = "+".join(strong_used) or identity[0]
                        break
            else:
                ambiguous += 1
                continue
        else:
            weak_candidates, weak_used = _intersection(weak, weak_index)
            if weak_candidates is None:
                no_reference += 1
                continue
            if len(weak_candidates) != 1:
                ambiguous += 1
                continue
            chosen_fingerprint = next(iter(weak_candidates))
            for identity in weak:
                candidates = weak_index.get(identity, {})
                if chosen_fingerprint in candidates:
                    chosen_context = candidates[chosen_fingerprint]
                    used_kind = "+".join(weak_used) or identity[0]
                    break
        if chosen_context is None:
            ambiguous += 1
            continue
        record.embedded_context = {
            key: np.asarray(value).copy() for key, value in chosen_context.items()
        }
        repaired += 1
        repaired_by[used_kind] = repaired_by.get(used_kind, 0) + 1

    return {
        "contract": V27_CONTEXT_RECOVERY_CONTRACT,
        "repaired": int(repaired),
        "ambiguous": int(ambiguous),
        "no_reference": int(no_reference),
        "already_available": int(already_available),
        "repaired_by_identity": dict(sorted(repaired_by.items())),
        "candidate_action_borrowed_from_peer": False,
    }


__all__ = ["V27_CONTEXT_RECOVERY_CONTRACT", "recover_missing_contexts_v27"]
