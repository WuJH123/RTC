from __future__ import annotations

import numpy as np

from rtc.step2_d4_action_support_v125 import (
    D4ActionSupportContractV125,
    action_sequence_sha256_v125,
    action_support_gap_v125,
    common_anchor_continuation_sequence_v125,
    deterministic_d4_rainfall_roles_v125,
    knowledge_neighbourhood_first_moves_v125,
    select_gap_balanced_checkpoints_v125,
)


def test_knowledge_neighbourhood_is_bounded_and_contains_anchor() -> None:
    current = np.asarray([0.2, 0.4, 0.5, 0.6], dtype=np.float32)
    anchor = np.asarray([0.6, 0.2, 0.7, 0.6], dtype=np.float32)
    groups = np.asarray([0, 0, 1, 2], dtype=np.int64)
    lower = np.zeros(4, dtype=np.float32)
    upper = np.ones(4, dtype=np.float32)
    contract = D4ActionSupportContractV125(max_active_groups=2)
    plans = knowledge_neighbourhood_first_moves_v125(
        current, anchor, groups, lower, upper, contract=contract
    )
    families = [family for family, _ in plans]
    targets = np.stack([target for _, target in plans])
    assert "hold" in families
    assert "anchor_scale_1.00" in families
    assert len(families) == len(set(families))
    assert np.all(targets >= lower[None, :] - 1e-7)
    assert np.all(targets <= upper[None, :] + 1e-7)
    assert np.max(np.abs(targets - current[None, :])) <= 0.5 + 1e-7
    np.testing.assert_allclose(targets[families.index("anchor_scale_1.00")], anchor, atol=1e-7)


def test_common_continuation_changes_only_first_executable_block() -> None:
    anchor = np.arange(18, dtype=np.float32).reshape(6, 3) / 20.0
    first = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    seq = common_anchor_continuation_sequence_v125(first, anchor, control_block_steps=2)
    np.testing.assert_array_equal(seq[:2], np.repeat(first[None, :], 2, axis=0))
    np.testing.assert_array_equal(seq[2:], anchor[2:])
    assert action_sequence_sha256_v125(seq) == action_sequence_sha256_v125(seq.copy())


def test_rainfall_split_is_outcome_blind_deterministic_and_disjoint() -> None:
    groups = [f"r{i:02d}" for i in range(14)]
    first = deterministic_d4_rainfall_roles_v125(groups, audit_fraction=0.25)
    second = deterministic_d4_rainfall_roles_v125(list(reversed(groups)), audit_fraction=0.25)
    assert first == second
    audit = {g for g, role in first.items() if role == "audit"}
    fit = {g for g, role in first.items() if role == "fit"}
    assert len(audit) == 4
    assert len(fit) == 10
    assert not (audit & fit)
    assert audit | fit == set(groups)


def test_support_gap_is_zero_when_anchor_is_labelled() -> None:
    current = np.asarray([0.2, 0.4], dtype=np.float32)
    anchor = np.asarray([0.6, 0.1], dtype=np.float32)
    candidates = np.asarray([[0.2, 0.4], [0.6, 0.1], [0.3, 0.3]], dtype=np.float32)
    gap = action_support_gap_v125(current, anchor, candidates)
    assert gap["nearest_anchor_l1_normalized"] <= 1e-7
    assert gap["nearest_direction_agreement"] == 1.0


def test_gap_balanced_selection_is_deterministic_and_round_robin() -> None:
    records = [
        {"rainfall_group": "r1", "event_id": "e1", "checkpoint_id": "a", "nearest_anchor_l1_normalized": 0.9},
        {"rainfall_group": "r1", "event_id": "e1", "checkpoint_id": "b", "nearest_anchor_l1_normalized": 0.8},
        {"rainfall_group": "r2", "event_id": "e2", "checkpoint_id": "c", "nearest_anchor_l1_normalized": 0.7},
        {"rainfall_group": "r2", "event_id": "e2", "checkpoint_id": "d", "nearest_anchor_l1_normalized": 0.6},
    ]
    first = select_gap_balanced_checkpoints_v125(records, max_checkpoints=3)
    second = select_gap_balanced_checkpoints_v125(records, max_checkpoints=3)
    assert first == second
    assert [x["checkpoint_id"] for x in first] == ["a", "c", "b"]
