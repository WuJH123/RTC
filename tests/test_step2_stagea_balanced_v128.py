from __future__ import annotations

import numpy as np

from rtc.step2_stagea_balanced_v128 import (
    D2_DIRECT_PAIR_BUDGET,
    OTHER_DIRECT_PAIR_BUDGET,
    select_balanced_direct_specs,
)


def _spec(candidate: int, actuator: int, effect: float) -> dict[str, float | int]:
    return {
        "candidate_position": candidate,
        "step": 0,
        "actuator_index": actuator,
        "setting_delta": 0.5,
        "true_flow_delta": effect,
        "prefix_state_max_abs": 0.0,
        "prefix_flow_max_abs": 0.0,
    }


def test_balanced_selector_prefers_informative_and_undercovered_actuators() -> None:
    specs = [
        _spec(1, 0, 0.0),
        _spec(2, 1, 2.0),
        _spec(3, 2, 1.0),
        _spec(4, 3, 0.5),
    ]
    counts = np.asarray([0, 4, 0, 0], dtype=np.int64)

    selected = select_balanced_direct_specs(
        specs,
        source="D4",
        group_name="g",
        epoch=1,
        seed=42,
        actuator_counts=counts,
    )

    assert len(selected) == len(specs)
    # Zero-response pairs are ranked after informative pairs even if their actuator is uncovered.
    assert selected[-1]["actuator_index"] == 0
    # Within informative pairs, less-covered actuators outrank an already overrepresented one.
    assert selected[0]["actuator_index"] in {2, 3}
    assert counts.sum() == 8


def test_balanced_selector_uses_larger_d2_budget() -> None:
    specs = [_spec(i + 1, i, float(i + 1)) for i in range(20)]

    d2_counts = np.zeros(32, dtype=np.int64)
    other_counts = np.zeros(32, dtype=np.int64)
    d2 = select_balanced_direct_specs(
        specs,
        source="D2",
        group_name="d2",
        epoch=7,
        seed=42,
        actuator_counts=d2_counts,
    )
    other = select_balanced_direct_specs(
        specs,
        source="D3",
        group_name="d3",
        epoch=7,
        seed=42,
        actuator_counts=other_counts,
    )

    assert len(d2) == D2_DIRECT_PAIR_BUDGET
    assert len(other) == OTHER_DIRECT_PAIR_BUDGET
    assert len(d2) > len(other)
    assert np.count_nonzero(d2_counts) == D2_DIRECT_PAIR_BUDGET
    assert np.count_nonzero(other_counts) == OTHER_DIRECT_PAIR_BUDGET


def test_balanced_selector_is_deterministic() -> None:
    specs = [_spec(i + 1, i % 4, 1.0) for i in range(16)]
    first_counts = np.zeros(8, dtype=np.int64)
    second_counts = np.zeros(8, dtype=np.int64)

    first = select_balanced_direct_specs(
        specs,
        source="D2",
        group_name="same",
        epoch=11,
        seed=123,
        actuator_counts=first_counts,
    )
    second = select_balanced_direct_specs(
        specs,
        source="D2",
        group_name="same",
        epoch=11,
        seed=123,
        actuator_counts=second_counts,
    )

    assert first == second
    assert np.array_equal(first_counts, second_counts)
