from __future__ import annotations

import pytest

from rtc.project7_policy_consistency import (
    RESOLVED_CONTINUATION_FIELD,
    policy_consistent_decision_unit,
    resolve_continuations,
    select_target_continuation,
)


def _row(context: str, continuation: str = "", *, target: str = "a", group: str = "g"):
    return {
        "causal_context_fingerprint_sha256": context,
        "continuation_policy_sha256": continuation,
        "candidate_first_target_sha256": target,
        "leakage_group_id": group,
    }


def test_missing_hash_recovers_only_from_unique_same_context_peer() -> None:
    h = "a" * 64
    rows = [_row("ctx", h, target="a"), _row("ctx", "", target="b")]
    result = resolve_continuations(rows)
    assert result.report["resolved_rows"] == 2
    assert result.rows[1][RESOLVED_CONTINUATION_FIELD] == h
    assert result.rows[1]["resolved_continuation_provenance"] == "inferred_unique_same_context_peer"


def test_ambiguous_context_is_not_imputed() -> None:
    a, b = "a" * 64, "b" * 64
    rows = [_row("ctx", a), _row("ctx", b), _row("ctx", "", target="c")]
    result = resolve_continuations(rows)
    assert result.report["ambiguous_context_count"] == 1
    assert result.rows[2][RESOLVED_CONTINUATION_FIELD] == ""
    assert result.rows[2]["policy_consistency_usable"] is False


def test_decision_unit_changes_with_continuation() -> None:
    base = _row("ctx", "a" * 64)
    result = resolve_continuations([base]).rows[0]
    first = policy_consistent_decision_unit(result)
    changed = dict(result)
    changed[RESOLVED_CONTINUATION_FIELD] = "b" * 64
    second = policy_consistent_decision_unit(changed)
    assert first != second


def test_target_selection_never_accepts_unresolved_or_other_policy() -> None:
    target, other = "a" * 64, "b" * 64
    resolved = resolve_continuations([
        _row("ctx1", target),
        _row("ctx2", other),
        _row("ctx3", ""),
    ])
    selected, report = select_target_continuation(
        resolved.rows, target_continuation_sha256=target
    )
    assert len(selected) == 1
    assert report["selected_rows"] == 1
    assert report["unresolved_rows"] == 1
    assert report["other_continuation_counts"] == {other: 1}


def test_decision_unit_fails_closed_without_resolved_continuation() -> None:
    row = _row("ctx", "")
    with pytest.raises(ValueError):
        policy_consistent_decision_unit(row)
