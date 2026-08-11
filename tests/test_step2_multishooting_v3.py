from __future__ import annotations

import torch

from rtc.step2_counterfactual import CounterfactualLossWeights, counterfactual_action_loss
from rtc.step2_train_v4 import _pack_segment_chunk, _segment_starts, _stage


def test_counterfactual_tfv_proxy_keeps_gradient_below_zero_flood_rate() -> None:
    initial = torch.zeros(2, 1, 6)
    target_states = torch.zeros(2, 1, 1, 6)
    target_states[1, 0, 0, 2] = 1.0
    target_flows = torch.zeros(2, 1, 1)
    predicted = torch.zeros(2, 1, 1, 6, requires_grad=True)
    with torch.no_grad():
        predicted[..., 2].fill_(-0.05)
    rollout_flows = torch.zeros(2, 1, 1, requires_grad=True)
    weights = CounterfactualLossWeights(
        absolute_state=0.0,
        absolute_flow=0.0,
        delta_state=0.0,
        delta_flow=0.0,
        delta_tfv=1.0,
        ranking=1.0,
        exact_flood=0.0,
        physical=0.0,
    )
    metrics = counterfactual_action_loss(
        initial_state=initial,
        rollout_states=predicted,
        rollout_flows=rollout_flows,
        target_states=target_states,
        target_flows=target_flows,
        exact_node_flood_volume_m3=None,
        dt_seconds=torch.full((2, 1), 300.0),
        state_std=torch.ones(6),
        flow_std=torch.ones(1),
        full_horizon=False,
        weights=weights,
    )
    metrics.total.backward()
    assert predicted.grad is not None
    assert float(predicted.grad[..., 2].abs().sum()) > 0.0
    assert torch.isfinite(predicted.grad).all()


def test_multishooting_segment_pack_uses_authoritative_segment_prefix() -> None:
    b, h, n, f, a = 2, 72, 3, 6, 2
    initial = torch.zeros(b, n, f)
    target_states = torch.zeros(b, h, n, f)
    target_flows = torch.zeros(b, h, a)
    for step in range(h):
        target_states[:, step].fill_(float(step + 1))
        target_flows[:, step].fill_(float(step + 101))
    pair = {
        "initial_state": initial,
        "rainfall": torch.zeros(b, h, n, 1),
        "settings": torch.zeros(b, h, a),
        "previous_actuator_flow": torch.zeros(b, a),
        "target_states": target_states,
        "target_actuator_flows": target_flows,
        "elapsed_seconds": torch.arange(h + 1).float().view(1, -1).expand(b, -1) * 300.0,
    }
    packed = _pack_segment_chunk(pair, [0, 12, 24])
    assert packed["initial_state"].shape == (6, n, f)
    assert packed["rainfall"].shape == (6, 12, n, 1)
    torch.testing.assert_close(packed["initial_state"][0:2], initial)
    torch.testing.assert_close(packed["initial_state"][2:4], target_states[:, 11])
    torch.testing.assert_close(packed["initial_state"][4:6], target_states[:, 23])
    torch.testing.assert_close(
        packed["previous_actuator_flow"][2:4], target_flows[:, 11]
    )
    torch.testing.assert_close(
        packed["previous_actuator_flow"][4:6], target_flows[:, 23]
    )


def test_multishooting_schedule_and_segment_starts_are_fixed() -> None:
    assert _segment_starts(72) == [0, 12, 24, 36, 48, 60]
    assert _stage(9, 18)[0:2] == ("h12", 12)
    assert _stage(10, 18)[0:2] == ("h72_multishooting", 72)
    assert _stage(16, 18)[0] == "h72_multishooting_exact_diagnostic"
