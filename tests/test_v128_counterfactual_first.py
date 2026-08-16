from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rtc.step2_counterfactual_first_v128 import (
    CounterfactualFirstActuatorFlowModelV128,
    derive_direct_response_scales_v128,
    first_direct_response_spec_numpy,
)
from rtc.step2_counterfactual_training_v4 import _direct_specs_lazy, _zero_based_spec_order
from rtc.step2_lazy_stream_v128 import LazyBranchArrayV128


@dataclass
class _Entry:
    arrays: dict[str, np.ndarray]
    indices: tuple[int, ...]
    reference_index: int = 0


class _Cache:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._entry = _Entry(arrays=arrays, indices=tuple(range(arrays["settings"].shape[0])))

    def entry(self, name: str) -> _Entry:
        assert name == "FIT"
        return self._entry


def _arrays(*, prefix_mismatch: bool = False) -> dict[str, np.ndarray]:
    settings = np.zeros((2, 4, 2), dtype=np.float32)
    settings[1, 1:, 0] = 1.0
    flows = np.zeros((2, 4, 2), dtype=np.float32)
    flows[0, :, 0] = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    flows[1, :, 0] = np.asarray([1.0, 1.6, 80.0, -70.0], dtype=np.float32)
    states = np.zeros((2, 4, 1, 2), dtype=np.float32)
    initial = np.zeros((2, 1, 2), dtype=np.float32)
    previous = np.zeros((2, 2), dtype=np.float32)
    if prefix_mismatch:
        states[1, 0, 0, 0] = 0.1
    return {
        "settings": settings,
        "target_actuator_flows": flows,
        "target_states": states,
        "initial_state": initial,
        "previous_actuator_flow": previous,
    }


def test_first_direct_response_uses_only_same_prefix_first_divergence() -> None:
    arrays = _arrays()
    spec = first_direct_response_spec_numpy(arrays, reference=0, candidate=1)
    assert spec is not None
    assert spec["step"] == 1
    assert spec["actuator_index"] == 0
    assert np.isclose(spec["true_flow_delta"], 0.5)
    assert spec["prefix_state_max_abs"] == 0.0
    assert spec["prefix_flow_max_abs"] == 0.0
    assert first_direct_response_spec_numpy(
        _arrays(prefix_mismatch=True), reference=0, candidate=1
    ) is None


def test_direct_action_scale_excludes_later_feedback(monkeypatch) -> None:
    cache = _Cache(_arrays())

    def temporal(*args, **kwargs):
        return (
            np.asarray([0.1, 0.1], dtype=np.float32),
            np.asarray([10.0, 0.2], dtype=np.float32),
            {"temporal": True},
        )

    monkeypatch.setattr(
        "rtc.step2_counterfactual_first_v128.derive_residual_scales_streaming_v127", temporal
    )
    state, temporal_scale, direct_scale, telemetry = derive_direct_response_scales_v128(
        ((cache, ["FIT"]),), sample_rows=16
    )
    np.testing.assert_allclose(state, [0.1, 0.1])
    np.testing.assert_allclose(temporal_scale, [10.0, 0.2])
    assert np.isclose(direct_scale[0], 0.5)
    assert np.isclose(direct_scale[1], 1.0e-5)
    assert telemetry["temporal_and_action_scales_separate"] is True
    assert telemetry["holdout_used_for_scale"] is False
    assert telemetry["feedback_horizon_samples_excluded_from_direct_scale"] >= 2


def test_counterfactual_actuator_separates_temporal_and_setting_scales() -> None:
    model = CounterfactualFirstActuatorFlowModelV128(
        state_dim=2,
        physics_dim=1,
        hidden_dim=4,
        actuator_count=1,
        actuator_embedding_dim=1,
        delta_flow_scale=torch.tensor([100.0]),
        direct_action_flow_scale=torch.tensor([0.5]),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.setting_linear_gain.bias.fill_(3.0)
    state = torch.zeros(1, 1, 2)
    previous = torch.zeros(1, 1)
    physics = torch.zeros(1, 1, 1)
    physics_norm, identity = model.prepare_static(physics, batch_size=1)
    q0, _ = model.forward_prepared(
        state, state, torch.zeros(1, 1), previous, physics_norm, identity
    )
    q1, _ = model.forward_prepared(
        state, state, torch.ones(1, 1), previous, physics_norm, identity
    )
    assert 0.9 < float(q1 - q0) <= 1.0


def test_direct_spec_permutation_is_zero_based_and_complete() -> None:
    order = _zero_based_spec_order(8, group_name="G", epoch=1, seed=42)
    assert sorted(order.tolist()) == list(range(8))


def test_direct_specs_support_lazy_mmap_truth_without_full_tensor_materialization() -> None:
    arrays = _arrays()
    order = np.asarray([0, 1], dtype=np.int64)
    cpu = {
        "settings": torch.as_tensor(arrays["settings"], dtype=torch.float32),
        "initial": torch.as_tensor(arrays["initial_state"][:1], dtype=torch.float32),
        "previous_flow": torch.as_tensor(arrays["previous_actuator_flow"][:1], dtype=torch.float32),
        "states": LazyBranchArrayV128(arrays["target_states"], order),
        "flows": LazyBranchArrayV128(arrays["target_actuator_flows"], order),
    }
    specs = _direct_specs_lazy(cpu)
    assert len(specs) == 1
    assert specs[0]["candidate_position"] == 1
    assert specs[0]["step"] == 1
    assert np.isclose(specs[0]["true_flow_delta"], 0.5)
