from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.build_step2_v125_anchor_override_evidence import _candidate_indices


def test_candidate_indices_exclude_reference_and_preserve_order() -> None:
    entry = SimpleNamespace(indices=(3, 7, 9, 11), reference_index=7)
    assert _candidate_indices(entry) == [3, 9, 11]


def test_candidate_indices_fail_closed_without_valid_reference() -> None:
    with pytest.raises(RuntimeError):
        _candidate_indices(SimpleNamespace(indices=(1, 2), reference_index=5))
