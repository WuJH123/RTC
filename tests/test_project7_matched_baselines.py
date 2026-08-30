from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.project7_matched_baselines import (
    MATCHED_ACTIVE_BASELINES,
    MATCHED_AUTO_RBC,
    MATCHED_EFD,
    _raw_auto_rbc_target,
    _raw_efd_target,
)


def _graph():
    node_ids = ("S1", "J2", "J3")
    static_names = ("max_depth_m", "storage_capacity_m3")
    static = np.asarray([
        [4.0, 100.0],
        [4.0, 0.0],
        [4.0, 0.0],
    ], dtype=np.float32)
    physics_names = ("is_pump", "is_orifice", "is_weir", "is_outlet")
    physics = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)
    return SimpleNamespace(
        node_ids=node_ids,
        static_node_feature_names=static_names,
        static_node_features=static,
        actuator_ids=("P1", "O1"),
        actuator_upstream=np.asarray([0, 0], dtype=np.int64),
        actuator_downstream=np.asarray([1, 2], dtype=np.int64),
        actuator_physics_feature_names=physics_names,
        actuator_physics=physics,
    )


def _state():
    state = torch.zeros((3, 4), dtype=torch.float32)
    state[:, 0] = torch.tensor([3.0, 0.5, 0.25])
    state[:, 3] = torch.tensor([75.0, 0.0, 0.0])
    return state


def test_matched_baseline_ids_are_explicit() -> None:
    assert MATCHED_ACTIVE_BASELINES == (MATCHED_AUTO_RBC, MATCHED_EFD)


def test_auto_rbc_uses_reconstructed_state_tensor() -> None:
    raw, mean_up, max_down = _raw_auto_rbc_target(
        graph=_graph(), current_state=_state(), active_target=torch.zeros(2)
    )
    assert tuple(raw.shape) == (2,)
    assert mean_up > 0.0
    assert max_down >= 0.0
    assert torch.isfinite(raw).all()


def test_efd_uses_reconstructed_storage_volume() -> None:
    raw, mean_fill, std_fill = _raw_efd_target(
        graph=_graph(), current_state=_state(), active_target=torch.zeros(2)
    )
    assert tuple(raw.shape) == (2,)
    assert 0.0 < mean_fill <= 1.5
    assert std_fill >= 0.0
    assert torch.isfinite(raw).all()
