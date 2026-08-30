"""Policy-consistency utilities for Project7 exact-return supervision.

This module is deliberately zero-SWMM and non-mutating with respect to historical evidence.  It
creates a derived, auditable view in which every usable row has one resolved continuation-policy
identity.  Missing continuation hashes may be recovered only from an exact causal-context peer when
that context contains exactly one explicit 64-hex continuation hash.  Ambiguous or unsupported rows
remain unresolved and must not enter policy-consistent pairwise supervision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence


POLICY_CONSISTENCY_CONTRACT = "PROJECT7_POLICY_CONSISTENT_EXACT_RETURN_VIEW_V1"
RESOLVED_CONTINUATION_FIELD = "resolved_continuation_policy_sha256"
CONTINUATION_PROVENANCE_FIELD = "resolved_continuation_provenance"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_sha256(value: Any) -> str:
    text = _text(value).lower()
    return text if _HEX64.fullmatch(text) else ""


def context_id(row: Mapping[str, Any]) -> str:
    return _text(row.get("causal_context_fingerprint_sha256")).lower()


def explicit_continuation(row: Mapping[str, Any]) -> str:
    return normalize_sha256(row.get("continuation_policy_sha256"))


@dataclass(frozen=True)
class ContinuationResolution:
    rows: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def resolve_continuations(rows: Sequence[Mapping[str, Any]]) -> ContinuationResolution:
    """Resolve missing continuation hashes without inventing cross-context provenance.

    Recovery is permitted only when the *same exact causal-context fingerprint* contains one and only
    one explicit continuation hash.  This is intentionally stricter than event/time/query matching.
    """
    known_by_context: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        cid = context_id(row)
        continuation = explicit_continuation(row)
        if cid and continuation:
            known_by_context[cid].add(continuation)

    ambiguous_contexts = {
        cid: tuple(sorted(values)) for cid, values in known_by_context.items() if len(values) > 1
    }
    resolved: list[dict[str, Any]] = []
    provenance = Counter()
    unresolved_reason = Counter()
    hash_counts = Counter()

    for source in rows:
        row = dict(source)
        cid = context_id(row)
        explicit = explicit_continuation(row)
        value = ""
        origin = ""
        if explicit:
            value = explicit
            origin = "explicit"
        elif not cid:
            unresolved_reason["missing_causal_context_fingerprint"] += 1
        elif cid in ambiguous_contexts:
            unresolved_reason["ambiguous_same_context_continuation"] += 1
        else:
            peers = known_by_context.get(cid, set())
            if len(peers) == 1:
                value = next(iter(peers))
                origin = "inferred_unique_same_context_peer"
            else:
                unresolved_reason["no_explicit_same_context_peer"] += 1

        row[RESOLVED_CONTINUATION_FIELD] = value
        row[CONTINUATION_PROVENANCE_FIELD] = origin or "unresolved"
        row["policy_consistency_contract"] = POLICY_CONSISTENCY_CONTRACT
        row["policy_consistency_usable"] = bool(value)
        provenance[origin or "unresolved"] += 1
        if value:
            hash_counts[value] += 1
        resolved.append(row)

    report = {
        "contract": POLICY_CONSISTENCY_CONTRACT,
        "input_rows": len(rows),
        "resolved_rows": int(sum(bool(row[RESOLVED_CONTINUATION_FIELD]) for row in resolved)),
        "unresolved_rows": int(sum(not bool(row[RESOLVED_CONTINUATION_FIELD]) for row in resolved)),
        "provenance_counts": dict(sorted(provenance.items())),
        "resolved_hash_counts": dict(sorted(hash_counts.items())),
        "ambiguous_context_count": len(ambiguous_contexts),
        "ambiguous_contexts": ambiguous_contexts,
        "unresolved_reason_counts": dict(sorted(unresolved_reason.items())),
        "missing_hash_is_never_treated_as_equivalent": True,
        "cross_context_imputation_forbidden": True,
        "historical_evidence_mutated": False,
        "new_swmm_truth_generated": False,
    }
    return ContinuationResolution(tuple(resolved), report)


def policy_consistent_decision_unit(row: Mapping[str, Any]) -> str:
    """Identity for same-state pairwise learning under one continuation policy."""
    group = _text(row.get("leakage_group_id")).lower()
    context = context_id(row)
    continuation = normalize_sha256(row.get(RESOLVED_CONTINUATION_FIELD))
    if not group or not context or not continuation:
        raise ValueError(
            "policy-consistent decision unit requires leakage group, causal context and resolved continuation"
        )
    return hashlib.sha256(f"{group}|{context}|{continuation}".encode("utf-8")).hexdigest()


def select_target_continuation(
    rows: Iterable[Mapping[str, Any]], *, target_continuation_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = normalize_sha256(target_continuation_sha256)
    if not target:
        raise ValueError("target continuation must be one 64-hex SHA256")
    selected: list[dict[str, Any]] = []
    other = Counter()
    unresolved = 0
    for source in rows:
        row = dict(source)
        value = normalize_sha256(row.get(RESOLVED_CONTINUATION_FIELD))
        if not value:
            unresolved += 1
        elif value == target:
            selected.append(row)
        else:
            other[value] += 1
    return selected, {
        "contract": POLICY_CONSISTENCY_CONTRACT,
        "target_continuation_policy_sha256": target,
        "selected_rows": len(selected),
        "unresolved_rows": unresolved,
        "other_continuation_counts": dict(sorted(other.items())),
        "mixed_continuation_training_forbidden": True,
    }


__all__ = [
    "CONTINUATION_PROVENANCE_FIELD",
    "ContinuationResolution",
    "POLICY_CONSISTENCY_CONTRACT",
    "RESOLVED_CONTINUATION_FIELD",
    "explicit_continuation",
    "normalize_sha256",
    "policy_consistent_decision_unit",
    "resolve_continuations",
    "select_target_continuation",
]
