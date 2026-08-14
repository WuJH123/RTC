from __future__ import annotations

import numpy as np

from rtc.step3_knowledge_seeds_v123 import (
    build_knowledge_guided_seed_settings_v123,
    build_sparse_state_auto_rbc_anchor_v123,
    knowledge_guided_seed_delta_v123,
    sparse_state_auto_rbc_target_v123,
)


class _Graph:
    node_ids = ("n0", "n1", "n2", "n3")
    actuator_upstream = np.asarray([0, 1, 2], dtype=np.int64)
    actuator_downstream = np.asarray([1, 2, 3], dtype=np.int64)
    static_node_features = np.asarray(
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32
    )
    static_node_feature_names = (
        "invert_elevation_m", "max_depth_m", "is_junction", "is_outfall",
        "is_storage", "is_divider", "init_depth_m", "surcharge_depth_m",
        "ponded_area_m2", "storage_capacity_m3",
    )


def test_knowledge_seed_uses_current_state_and_has_bounded_signed_direction() -> None:
    state = np.zeros((4, 6), dtype=np.float32)
    state[0, 0] = 0.9
    state[1, 0] = 0.1
    state[2, 0] = 0.1
    state[3, 0] = 0.1
    delta = knowledge_guided_seed_delta_v123(
        state, np.zeros((1, 4, 1), dtype=np.float32), _Graph()
    )
    assert delta.shape == (3,)
    assert delta[0] > 0.0
    assert np.all(np.abs(delta) <= 0.5)


def test_knowledge_seed_is_exact_hold_for_equal_state_and_engineering_bounded() -> None:
    state = np.zeros((4, 6), dtype=np.float32)
    reference = np.full((72, 3), 0.5, dtype=np.float32)
    seed = build_knowledge_guided_seed_settings_v123(
        state,
        np.zeros((72, 4, 1), dtype=np.float32),
        reference,
        _Graph(),
    )
    assert seed.shape == (72, 3)
    assert np.allclose(seed, reference)
    assert np.max(np.abs(np.diff(seed, axis=0))) <= 0.5 + 1e-7


def test_sparse_state_rbc_matches_local_fill_logic() -> None:
    state = np.zeros((4, 6), dtype=np.float32)
    current = np.full(3, 0.5, dtype=np.float32)
    # Actuator 0: strongly filled upstream and empty downstream -> open.
    state[0, 0] = 0.90
    state[1, 0] = 0.10
    # Actuator 1: low upstream -> close relative to current.
    state[2, 0] = 0.10
    target = sparse_state_auto_rbc_target_v123(state, current, _Graph())
    assert target.shape == (3,)
    assert target[0] > current[0]
    assert target[1] < current[1]
    assert np.max(np.abs(target - current)) <= 0.5 + 1e-7


def test_sparse_state_rbc_anchor_only_commits_current_feedback_move() -> None:
    state = np.zeros((4, 6), dtype=np.float32)
    state[0, 0] = 0.90
    state[1, 0] = 0.10
    current = np.full(3, 0.5, dtype=np.float32)
    reference = np.full((72, 3), 0.5, dtype=np.float32)
    anchor = build_sparse_state_auto_rbc_anchor_v123(
        state,
        current,
        reference,
        _Graph(),
        control_block_steps=2,
    )
    assert anchor.shape == reference.shape
    # Receding-horizon semantics: compute one current feedback target and hold it in the
    # scoring tail; the next decision epoch will recompute it from a fresh Step1 state.
    assert np.allclose(anchor, anchor[0][None, :])
    assert not np.allclose(anchor[0], reference[0])
    assert np.max(np.abs(anchor[0] - current)) <= 0.5 + 1e-7
