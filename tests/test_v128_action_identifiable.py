from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rtc.step2_action_identifiable_v128 import (
    ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
    ActionIdentifiableActuatorFlowModelV128,
    _response_weighted_effect_loss,
    derive_action_conditioned_residual_scales_v128,
)


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


def _fit_cache() -> _Cache:
    # Reference plus one candidate.  Actuator 0 has a 2 m3/s counterfactual response while
    # actuator 1 is unchanged.  Temporal changes are intentionally much smaller.
    settings = np.zeros((2, 3, 2), dtype=np.float32)
    settings[1, :, 0] = 1.0
    flow = np.zeros((2, 3, 2), dtype=np.float32)
    flow[0, :, 0] = np.asarray([1.00, 1.01, 1.02], dtype=np.float32)
    flow[1, :, 0] = np.asarray([3.00, 3.01, 3.02], dtype=np.float32)
    states = np.zeros((2, 3, 1, 2), dtype=np.float32)
    states[:, 1:, 0, 0] = 0.01
    initial = np.zeros((2, 1, 2), dtype=np.float32)
    previous = np.zeros((2, 2), dtype=np.float32)
    return _Cache(
        {
            "settings": settings,
            "target_actuator_flows": flow,
            "target_states": states,
            "initial_state": initial,
            "previous_actuator_flow": previous,
        }
    )


def test_response_weighted_effect_loss_penalizes_magnitude_collapse() -> None:
    true = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    scale = torch.tensor([2.0, 1.0], dtype=torch.float32)
    exact = _response_weighted_effect_loss(true, true, scale)
    collapsed = _response_weighted_effect_loss(torch.zeros_like(true), true, scale)
    assert float(exact) == 0.0
    assert float(collapsed) > 0.0


def test_action_identifiable_actuator_has_explicit_setting_response() -> None:
    model = ActionIdentifiableActuatorFlowModelV128(
        state_dim=2,
        physics_dim=1,
        hidden_dim=4,
        actuator_count=1,
        actuator_embedding_dim=1,
        delta_flow_scale=torch.tensor([2.0]),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Make the linear setting branch non-zero independent of context.
        model.setting_linear_gain.bias.fill_(0.5)
    state = torch.zeros(1, 1, 2)
    previous = torch.zeros(1, 1)
    physics = torch.zeros(1, 1, 1)
    physics_norm, identity = model.prepare_static(physics, batch_size=1)
    q0, _ = model.forward_prepared(
        state,
        state,
        torch.zeros(1, 1),
        previous,
        physics_norm,
        identity,
    )
    q1, _ = model.forward_prepared(
        state,
        state,
        torch.ones(1, 1),
        previous,
        physics_norm,
        identity,
    )
    assert float(q1.item() - q0.item()) > 1.0


def test_hybrid_scale_promotes_fit_action_response(monkeypatch) -> None:
    cache = _fit_cache()

    def fake_temporal(*args, **kwargs):
        return (
            np.asarray([0.1, 0.1], dtype=np.float32),
            np.asarray([0.02, 0.03], dtype=np.float32),
            {"temporal": True},
        )

    monkeypatch.setattr(
        "rtc.step2_action_identifiable_v128.derive_residual_scales_streaming_v127",
        fake_temporal,
    )
    state_scale, flow_scale, telemetry = derive_action_conditioned_residual_scales_v128(
        ((cache, ["FIT"]),),
        sample_rows=16,
    )
    assert np.allclose(state_scale, [0.1, 0.1])
    assert flow_scale[0] >= 1.99
    assert np.isclose(flow_scale[1], 0.03)
    assert telemetry["flow_scale_contract"] == ACTION_CONDITIONED_FLOW_SCALE_CONTRACT
    assert telemetry["holdout_used_for_scale"] is False
    assert telemetry["scale_is_engineering_constraint"] is False
