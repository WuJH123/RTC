from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_control_response_v60 import PreparedStaticV60
from rtc.step2_hydraulic_objective_v111 import _balanced_direct, derive_effect_scales_v111, hydraulic_effect_loss_v111
from rtc.step2_train_response_v60 import V60GroupBatch


class _Entry:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reference_index = 0
        self.indices = tuple(range(arrays["target_states"].shape[0]))


class _Cache:
    def __init__(self, arrays):
        self._entry = _Entry(arrays)

    def entry(self, _name):
        return self._entry


def _prepared():
    return PreparedStaticV60(
        node_static=torch.zeros(3, 1), actuator_physics=torch.zeros(2, 1),
        actuator_upstream=torch.tensor([0, 1]), actuator_downstream=torch.tensor([1, 2]),
        invert_elevation_m=torch.ones(3), max_depth_m=torch.ones(3), surcharge_depth_m=torch.zeros(3),
        storage_capacity_m3=torch.tensor([0.0, 0.0, 100.0]), storage_mask=torch.tensor([False, False, True]),
        actuator_feature_names=("x",),
    )


def test_v111_active_conditional_scale_and_fallback_are_trainfit_only():
    # Six candidates: node 0/depth and actuator 0 have support; node 1 and
    # actuator 1 deliberately fall back to the global active conditional P75.
    states = np.zeros((7, 72, 3, 6), dtype=np.float32)
    flows = np.zeros((7, 72, 2), dtype=np.float32)
    states[1:, :, 0, 0] = np.linspace(.2, 1.2, 6)[:, None]
    flows[1:, :, 0] = np.linspace(.2, 1.2, 6)[:, None]
    arrays = {"target_states": states, "target_actuator_flows": flows}
    scales = derive_effect_scales_v111(_Cache(arrays), ["D2::x"], _prepared())
    assert scales.state_local_active_support[0, 0] > 0
    assert scales.state_local_active_support[1, 0] == 0
    assert scales.flow_local_active_support[0] > 0
    assert scales.flow_local_active_support[1] == 0
    assert scales.state_fallback_fraction > 0.0
    assert scales.flow_fallback_fraction > 0.0
    assert np.all(scales.state_magnitude_scale >= scales.state_active_threshold)
    assert np.all(scales.flow_magnitude_scale >= scales.flow_active_threshold)


def test_v111_balanced_direct_loss_reports_primary_components():
    b, c, t, n, a = 1, 1, 2, 3, 2
    true_ref = torch.zeros(b, t, n, 6)
    true_cand = true_ref[:, None].clone()
    true_cand[..., 0] = 0.5
    true_flow_ref = torch.zeros(b, t, a)
    true_flow_cand = true_flow_ref[:, None].clone()
    true_flow_cand[..., 0] = 0.3
    batch = V60GroupBatch(
        source_kind="D2", group_name="g", initial_state=torch.zeros(1, n, 6), rainfall=torch.zeros(1, 2, n, 1),
        reference_settings=torch.zeros(1, 2, a), candidate_settings=torch.zeros(1, c, 2, a),
        previous_actuator_flow=torch.zeros(1, a), elapsed_seconds=torch.zeros(1),
        true_reference_states=true_ref, true_candidate_states=true_cand,
        true_reference_flows=true_flow_ref, true_candidate_flows=true_flow_cand,
        true_delta_tfv_m3=torch.zeros(1, c),
    )
    output = SimpleNamespace(
        horizon_indices=torch.tensor([0, 1]), response_minutes=torch.tensor([5.0, 10.0]),
        raw_delta_states_physical=torch.zeros(b, c, t, n, 6, requires_grad=True),
        raw_delta_flows_physical=torch.zeros(b, c, t, a, requires_grad=True),
        active_state_logits=torch.zeros(b, c, t, n, 5, requires_grad=True),
        sign_state_logits=torch.zeros(b, c, t, n, 5, requires_grad=True),
        magnitude_state_normalized=torch.zeros(b, c, t, n, 5, requires_grad=True),
        active_flow_logits=torch.zeros(b, c, t, a, requires_grad=True),
        sign_flow_logits=torch.zeros(b, c, t, a, requires_grad=True),
        magnitude_flow_normalized=torch.zeros(b, c, t, a, requires_grad=True),
    )
    scales = SimpleNamespace(
        state_magnitude_scale=np.ones((n, 5), np.float32), state_active_threshold=np.full((n, 5), .1, np.float32),
        flow_magnitude_scale=np.ones(a, np.float32), flow_active_threshold=np.full(a, .1, np.float32),
    )
    loss, metrics = hydraulic_effect_loss_v111(output, batch, scales)
    assert bool(torch.isfinite(loss))
    assert metrics["direct_active"] >= 0.0
    assert metrics["direct_inactive"] >= 0.0
    assert metrics["weighted_direct"] > 0.0
    loss.backward()
    assert output.raw_delta_states_physical.grad is not None
    assert bool(torch.isfinite(output.raw_delta_states_physical.grad).all())


def test_v111_inactive_floor_does_not_create_unbounded_gradient():
    pred = torch.full((1, 1, 2, 1), 1.0e-4, requires_grad=True)
    truth = torch.zeros_like(pred)
    scale = torch.ones_like(pred)
    threshold = torch.full_like(pred, 1.0e-5)
    loss, _active, _inactive = _balanced_direct(pred, truth, scale, threshold)
    loss.backward()
    assert bool(torch.isfinite(pred.grad).all())
    assert float(pred.grad.abs().max()) < 1.0
