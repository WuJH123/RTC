from __future__ import annotations

import numpy as np

from rtc.step3_knowledge_seeds_v123 import (
    build_knowledge_guided_seed_settings_v123,
    knowledge_guided_seed_delta_v123,
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
    state[0, 0] = 0.9  # upstream full, downstream empty: open actuator 0
    state[1, 0] = 0.1
    state[2, 0] = 0.1
    state[3, 0] = 0.1
    delta = knowledge_guided_seed_delta_v123(state, np.zeros((1, 4, 1), dtype=np.float32), _Graph())
    assert delta.shape == (3,)
    assert delta[0] > 0.0
    assert np.all(np.abs(delta) <= 0.5)


def test_knowledge_seed_is_exact_hold_for_equal_state_and_engineering_bounded() -> None:
    state = np.zeros((4, 6), dtype=np.float32)
    reference = np.full((72, 3), 0.5, dtype=np.float32)
    seed = build_knowledge_guided_seed_settings_v123(state, np.zeros((72, 4, 1), dtype=np.float32), reference, _Graph())
    assert seed.shape == (72, 3)
    assert np.allclose(seed, reference)
    assert np.max(np.abs(np.diff(seed, axis=0))) <= 0.5 + 1e-7
