from __future__ import annotations

import numpy as np
import torch

from rtc.step2_counterfactual_training_v5 import _direct_specs_lazy
from rtc.step2_oracle_isolation_v128 import (
    ORACLE_FLOW_ISOLATION_CONTRACT,
    oracle_flow_transition_prediction,
    shared_reference_setting,
)


class _Actuator:
    def prepare_static(self, physics, *, batch_size):
        return physics.expand(batch_size, -1, -1), None

    def forward_prepared(
        self,
        upstream_state,
        downstream_state,
        setting,
        previous_flow,
        physics_norm,
        identity,
    ):
        del upstream_state, downstream_state, previous_flow, physics_norm, identity
        return torch.zeros_like(setting), torch.zeros_like(setting)


class _Transition:
    def prepare_static(self, static, edges, *, batch_size, dtype):
        del edges, dtype
        return static.expand(batch_size, -1, -1), torch.zeros(2, 0, dtype=torch.long), torch.ones(2)

    def forward_prepared(self, state, rainfall, static, injection, edges, inv, action_context):
        del rainfall, static, edges, inv
        return state + injection + action_context


class _Model:
    def __init__(self) -> None:
        self.actuator = _Actuator()
        self.transition = _Transition()

    def _typed_action_context(
        self,
        *,
        state,
        setting,
        previous_flow,
        predicted_flow,
        responsiveness,
        upstream,
        downstream,
        physics_norm,
        identity_embedding,
    ):
        del previous_flow, predicted_flow, responsiveness, upstream, downstream, physics_norm
        del identity_embedding
        return setting[:, 0].reshape(-1, 1, 1).expand(-1, state.shape[1], 1)


class _LazyTruth:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.select_calls = 0

    @property
    def shape(self):
        return self.value.shape

    def select(self, positions: np.ndarray, *, horizon: int) -> np.ndarray:
        self.select_calls += 1
        return self.value[np.asarray(positions, dtype=np.int64), :horizon]

    def __array__(self, *args, **kwargs):
        raise AssertionError("lazy truth must not be materialized as a full ndarray")


def test_shared_reference_setting_blocks_setting_side_channel() -> None:
    setting = torch.tensor([[0.0], [1.0]])
    shared = shared_reference_setting(setting)
    torch.testing.assert_close(shared, torch.tensor([[0.0], [0.0]]))

    model = _Model()
    static = {
        "up": torch.tensor([0], dtype=torch.long),
        "down": torch.tensor([1], dtype=torch.long),
        "physics": torch.zeros(1, 1, 1),
        "static": torch.zeros(1, 2, 1),
        "edges": torch.zeros(2, 0, dtype=torch.long),
    }
    state = torch.zeros(2, 2, 1)
    previous = torch.zeros(2, 1)
    oracle_flow = torch.tensor([[1.0], [2.0]])
    rainfall = torch.zeros(2, 2, 1)

    bypass = oracle_flow_transition_prediction(
        model,
        prev_state=state,
        previous_flow=previous,
        setting=setting,
        oracle_flow=oracle_flow,
        rainfall=rainfall,
        static=static,
    )
    isolated = oracle_flow_transition_prediction(
        model,
        prev_state=state,
        previous_flow=previous,
        setting=setting,
        oracle_flow=oracle_flow,
        rainfall=rainfall,
        static=static,
        action_context_setting=shared,
    )

    # Authoritative q alone implies branch delta [-1, +1] at upstream/downstream nodes.
    torch.testing.assert_close(isolated[1] - isolated[0], torch.tensor([[-1.0], [1.0]]))
    # Without isolation the candidate setting leaks through the typed action message.
    torch.testing.assert_close(bypass[1] - bypass[0], torch.tensor([[0.0], [2.0]]))
    assert ORACLE_FLOW_ISOLATION_CONTRACT.endswith("_V1")


def test_direct_spec_discovery_keeps_truth_lazy() -> None:
    settings = torch.zeros(2, 4, 1)
    settings[1, 1:, 0] = 1.0
    states_np = np.zeros((2, 4, 2, 1), dtype=np.float32)
    flows_np = np.zeros((2, 4, 1), dtype=np.float32)
    flows_np[0, :, 0] = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    flows_np[1, :, 0] = np.asarray([1.0, 1.6, 50.0, -40.0], dtype=np.float32)
    states = _LazyTruth(states_np)
    flows = _LazyTruth(flows_np)
    cpu = {
        "settings": settings,
        "states": states,
        "flows": flows,
        "initial": torch.zeros(1, 2, 1),
        "previous_flow": torch.zeros(1, 1),
        "rainfall": torch.zeros(1, 4, 2),
    }

    specs = _direct_specs_lazy(cpu)
    assert len(specs) == 1
    assert specs[0]["step"] == 1
    assert specs[0]["actuator_index"] == 0
    assert np.isclose(specs[0]["true_flow_delta"], 0.5)
    assert states.select_calls >= 1
    assert flows.select_calls >= 1
