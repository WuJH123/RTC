from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rtc.step2_control_response_v60 import PreparedStaticV60
from rtc.step2_control_response_v110 import build_actuator_node_relations_v110
from rtc.step2_control_response_v111 import ActuatorSetHydraulicResponseV111


class _Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.horizon_indices = torch.arange(72, dtype=torch.long)

    def forward(self, initial_state, rainfall, reference_settings, candidate_settings, prepared):
        b = initial_state.shape[0]
        states = initial_state.new_zeros(b, 1, 72, 3, 6)
        states[..., 0] = 0.5
        states[..., 1] = 10.5
        states[..., 3] = 1.0
        flows = initial_state.new_zeros(b, 1, 72, 2)
        return SimpleNamespace(horizon_indices=self.horizon_indices.to(initial_state.device),
                               reference_states_physical=states,
                               reference_flows_physical=flows)


def _graph():
    return SimpleNamespace(
        node_ids=("n0", "n1", "n2"), actuator_ids=("a0", "a1"),
        actuator_upstream=np.asarray([0, 1]), actuator_downstream=np.asarray([1, 2]),
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]]),
    )


def _prepared():
    return PreparedStaticV60(
        node_static=torch.zeros(3, 3), actuator_physics=torch.zeros(2, 4),
        actuator_upstream=torch.tensor([0, 1]), actuator_downstream=torch.tensor([1, 2]),
        invert_elevation_m=torch.tensor([10.0, 10.0, 10.0]),
        max_depth_m=torch.ones(3), surcharge_depth_m=torch.zeros(3),
        storage_capacity_m3=torch.tensor([0.0, 0.0, 100.0]),
        storage_mask=torch.tensor([False, False, True]),
        actuator_feature_names=("a", "b", "c", "d"),
    )


def _model():
    g = _graph()
    return ActuatorSetHydraulicResponseV111(
        reference_model=_Reference(), state_magnitude_scale=np.ones((3, 5), np.float32),
        flow_magnitude_scale=np.ones(2, np.float32), node_static_dim=3, physics_dim=4,
        rainfall_dim=1, actuator_count=2, node_count=3,
        relations=build_actuator_node_relations_v110(g),
    )


def _inputs():
    return (torch.zeros(1, 3, 6), torch.zeros(1, 72, 3, 1),
            torch.zeros(1, 72, 2), torch.zeros(1, 2))


def test_v111_exact_zero_reference_and_frozen_reference():
    model = _model().eval()
    initial, rainfall, reference, previous = _inputs()
    with torch.no_grad():
        output = model(initial, rainfall, reference, reference[:, None], previous, _prepared())
    assert torch.equal(output.raw_delta_states_physical, torch.zeros_like(output.raw_delta_states_physical))
    assert torch.equal(output.raw_delta_flows_physical, torch.zeros_like(output.raw_delta_flows_physical))
    assert not any(parameter.requires_grad for parameter in model.reference_model.parameters())


def test_v111_future_action_causality_and_direct_gradient():
    model = _model().eval()
    initial, rainfall, reference, previous = _inputs()
    candidate = reference[:, None].clone().requires_grad_(True)
    candidate.data[:, :, 18:, 0] = 1.0
    output = model(initial, rainfall, reference, candidate, previous, _prepared())
    early = output.horizon_indices < 18
    assert torch.equal(output.raw_delta_states_physical[:, :, early], torch.zeros_like(output.raw_delta_states_physical[:, :, early]))
    assert torch.equal(output.raw_delta_flows_physical[:, :, early], torch.zeros_like(output.raw_delta_flows_physical[:, :, early]))
    score = output.raw_delta_states_physical.square().mean() + output.raw_delta_flows_physical.square().mean()
    gradient = torch.autograd.grad(score, candidate)[0]
    assert bool(torch.isfinite(gradient).all())
    assert int(torch.count_nonzero(gradient)) > 0

