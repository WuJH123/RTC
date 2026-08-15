from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_train_v127_streaming import (
    _select_to_device,
    derive_residual_scales_streaming_v127,
)


class _Cache:
    def __init__(self) -> None:
        rows, horizon, nodes, state_dim, actuators = 3, 4, 5, 2, 3
        states = np.zeros((rows, horizon, nodes, state_dim), dtype=np.float32)
        flows = np.zeros((rows, horizon, actuators), dtype=np.float32)
        initial = np.zeros((rows, nodes, state_dim), dtype=np.float32)
        previous = np.zeros((rows, actuators), dtype=np.float32)
        for row in range(rows):
            for t in range(horizon):
                states[row, t, :, 0] = row + t + 1
                states[row, t, :, 1] = 2 * (row + t + 1)
                flows[row, t] = row + t + 0.5
        self._entry = SimpleNamespace(
            reference_index=0,
            indices=(0, 1, 2),
            arrays={
                "target_states": states,
                "initial_state": initial,
                "target_actuator_flows": flows,
                "previous_actuator_flow": previous,
            },
        )

    def entry(self, name: str):
        assert name == "G"
        return self._entry


def test_streaming_residual_sampling_is_deterministic_and_bounded() -> None:
    cache = _Cache()
    a_state, a_flow, a_meta = derive_residual_scales_streaming_v127(
        ((cache, ["G"]),), sample_rows=6
    )
    b_state, b_flow, b_meta = derive_residual_scales_streaming_v127(
        ((cache, ["G"]),), sample_rows=6
    )
    np.testing.assert_allclose(a_state, b_state)
    np.testing.assert_allclose(a_flow, b_flow)
    assert a_meta == b_meta
    assert a_meta["sample_rows_per_branch"] == 2
    assert a_meta["state_sample_rows"] == 6
    assert a_meta["flow_sample_rows"] == 6
    assert np.all(a_state > 0)
    assert np.all(a_flow > 0)


def test_select_to_device_materializes_only_requested_branches() -> None:
    branches, horizon, nodes, state_dim, actuators = 5, 4, 3, 2, 2
    data = {
        "initial": torch.ones(1, nodes, state_dim),
        "rainfall": torch.ones(1, horizon, nodes, 1),
        "previous_flow": torch.ones(1, actuators),
        "settings": torch.arange(branches * horizon * actuators, dtype=torch.float32).reshape(
            branches, horizon, actuators
        ),
        "states": torch.arange(
            branches * horizon * nodes * state_dim, dtype=torch.float32
        ).reshape(branches, horizon, nodes, state_dim),
        "flows": torch.arange(branches * horizon * actuators, dtype=torch.float32).reshape(
            branches, horizon, actuators
        ),
    }
    selected = _select_to_device(
        data, [1, 4], device=torch.device("cpu"), horizon=3, include_truth=True
    )
    assert selected["settings"].shape == (2, 3, actuators)
    assert selected["states"].shape == (2, 3, nodes, state_dim)
    assert selected["flows"].shape == (2, 3, actuators)
    torch.testing.assert_close(selected["settings"][0], data["settings"][1, :3])
    torch.testing.assert_close(selected["settings"][1], data["settings"][4, :3])
    assert selected["initial"].shape[0] == 2
    assert selected["rainfall"].shape[0] == 2
