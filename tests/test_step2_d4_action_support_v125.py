from __future__ import annotations

import numpy as np

from rtc.step2_d4_action_support_v125 import (
    D4ActionSupportContractV125,
    action_support_gap_v125,
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
        current,
        anchor,
        groups,
        lower,
        upper,
        contract=contract,
    )
    families = [family for family, _ in plans]
    targets = np.stack([target for _, target in plans])

    assert "hold" in families
    assert "anchor_scale_1.00" in families
    assert len(families) == len(set(families))
    assert np.all(targets >= lower[None, :] - 1.0e-7)
    assert np.all(targets <= upper[None, :] + 1.0e-7)
    assert np.max(np.abs(targets - current[None, :])) <= 0.5 + 1.0e-7
    np.testing.assert_allclose(targets[families.index("anchor_scale_1.00")], anchor, atol=1.0e-7)


def test_support_gap_is_zero_when_anchor_is_labelled() -> None:
    current = np.asarray([0.2, 0.4], dtype=np.float32)
    anchor = np.asarray([0.6, 0.1], dtype=np.float32)
    candidates = np.asarray([[0.2, 0.4], [0.6, 0.1], [0.3, 0.3]], dtype=np.float32)
    gap = action_support_gap_v125(current, anchor, candidates)
    assert gap["nearest_anchor_l1_normalized"] <= 1.0e-7
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
