from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rtc.step2_control_response_v60 import PreparedStaticV60
from rtc.step2_control_response_v110 import (
    ActuatorSetHydraulicResponseV110,
    action_prefix_features_v110,
    build_actuator_node_relations_v110,
)
from rtc.step2_hydraulic_objective_v110 import derive_effect_scales_v110
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v110_contract import HydraulicHorizonV110


class _DummyReference(nn.Module):
    def __init__(self, nodes: int, actuators: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.nodes = nodes
        self.actuators = actuators
        self.indices = torch.as_tensor(MultiResolutionHorizonV60().indices(), dtype=torch.long)

    def forward(self, initial_state, rainfall, reference_settings, candidate_settings, prepared):
        batch = initial_state.shape[0]
        t = len(self.indices)
        states = initial_state.new_zeros(batch, 1, t, self.nodes, 6)
        states[..., 0] = 0.50
        states[..., 1] = 10.50
        states[..., 3] = 2.0
        states[..., 4] = 0.20
        states[..., 5] = 0.20
        flows = initial_state.new_zeros(batch, 1, t, self.actuators)
        return SimpleNamespace(
            horizon_indices=self.indices.to(initial_state.device),
            reference_states_physical=states,
            reference_flows_physical=flows,
        )


def _prepared(nodes=3, actuators=2):
    return PreparedStaticV60(
        node_static=torch.zeros(nodes, 3),
        actuator_physics=torch.zeros(actuators, 4),
        actuator_upstream=torch.tensor([0, 1], dtype=torch.long),
        actuator_downstream=torch.tensor([1, 2], dtype=torch.long),
        invert_elevation_m=torch.tensor([10.0, 10.0, 10.0]),
        max_depth_m=torch.tensor([1.0, 2.0, 4.0]),
        surcharge_depth_m=torch.zeros(nodes),
        storage_capacity_m3=torch.tensor([0.0, 0.0, 100.0]),
        storage_mask=torch.tensor([False, False, True]),
        actuator_feature_names=("a", "b", "c", "d"),
    )


def _graph():
    return SimpleNamespace(
        node_ids=("n0", "n1", "n2"),
        actuator_ids=("a0", "a1"),
        actuator_upstream=np.asarray([0, 1], dtype=np.int64),
        actuator_downstream=np.asarray([1, 2], dtype=np.int64),
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
    )


def _model():
    relations = build_actuator_node_relations_v110(_graph())
    return ActuatorSetHydraulicResponseV110(
        reference_model=_DummyReference(3, 2),
        state_magnitude_scale=np.ones((3, 5), dtype=np.float32),
        flow_magnitude_scale=np.ones(2, dtype=np.float32),
        node_static_dim=3,
        physics_dim=4,
        rainfall_dim=1,
        actuator_count=2,
        node_count=3,
        relations=relations,
    )


def _inputs():
    initial = torch.zeros(1, 3, 6)
    rainfall = torch.zeros(1, 72, 3, 1)
    reference = torch.zeros(1, 72, 2)
    previous_flow = torch.zeros(1, 2)
    return initial, rainfall, reference, previous_flow


def test_v110_hydraulic_horizon_is_120min_and_multiresolution():
    horizon = HydraulicHorizonV110()
    assert horizon.response_minutes() == (
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
        40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
        100.0, 110.0, 120.0,
    )


def test_action_prefix_features_do_not_see_future_actions():
    _, _, reference, _ = _inputs()
    candidate = reference[:, None].clone()
    candidate[:, :, 12:, 0] = 1.0
    indices = torch.tensor([0, 5, 11, 13], dtype=torch.long)
    features, mask = action_prefix_features_v110(reference, candidate, indices)
    assert not bool(mask[:, :, :3].any())
    assert bool(mask[:, :, 3, 0].all())
    later = candidate.clone()
    later[:, :, 30:, 1] = 1.0
    features_later, mask_later = action_prefix_features_v110(reference, later, indices)
    assert torch.equal(features, features_later)
    assert torch.equal(mask, mask_later)


def test_v110_exact_zero_and_delayed_action_causality():
    model = _model().eval()
    initial, rainfall, reference, previous_flow = _inputs()
    prepared = _prepared()
    with torch.no_grad():
        zero = model(initial, rainfall, reference, reference[:, None], previous_flow, prepared)
    assert torch.equal(zero.raw_delta_states_physical, torch.zeros_like(zero.raw_delta_states_physical))
    assert torch.equal(zero.raw_delta_flows_physical, torch.zeros_like(zero.raw_delta_flows_physical))
    assert not any(p.requires_grad for p in model.reference_model.parameters())

    candidate = reference[:, None].clone()
    candidate[:, :, 18:, 0] = 1.0
    with torch.no_grad():
        delayed = model(initial, rainfall, reference, candidate, previous_flow, prepared)
    early = delayed.horizon_indices < 18
    assert torch.equal(delayed.raw_delta_states_physical[:, :, early],
                       torch.zeros_like(delayed.raw_delta_states_physical[:, :, early]))
    assert torch.equal(delayed.raw_delta_flows_physical[:, :, early],
                       torch.zeros_like(delayed.raw_delta_flows_physical[:, :, early]))


def test_v110_nonlocal_relation_has_no_hop_cutoff():
    relations = build_actuator_node_relations_v110(_graph())
    assert relations.finite_hop_cutoff is False
    assert relations.pair_features.shape == (2, 3, 9)
    assert float(relations.pair_features[0, 2, 8]) == 1.0
    assert float(relations.pair_features[0, 2, 2]) > 0.0


def test_v110_multi_actuator_joint_response_is_not_forced_to_sum_of_single_outputs():
    torch.manual_seed(7)
    model = _model().eval()
    initial, rainfall, reference, previous_flow = _inputs()
    prepared = _prepared()
    a = reference[:, None].clone()
    b = reference[:, None].clone()
    ab = reference[:, None].clone()
    a[:, :, :12, 0] = 1.0
    b[:, :, :12, 1] = 1.0
    ab[:, :, :12, :] = 1.0
    with torch.no_grad():
        out_a = model(initial, rainfall, reference, a, previous_flow, prepared)
        out_b = model(initial, rainfall, reference, b, previous_flow, prepared)
        out_ab = model(initial, rainfall, reference, ab, previous_flow, prepared)
    summed = out_a.raw_delta_states_physical + out_b.raw_delta_states_physical
    assert not torch.allclose(out_ab.raw_delta_states_physical, summed, atol=1e-7, rtol=1e-6)


class _FakeEntry:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reference_index = 0
        self.indices = (0, 1)


class _FakeCache:
    def __init__(self, arrays):
        self._entry = _FakeEntry(arrays)

    def entry(self, name):
        return self._entry


def test_v110_active_thresholds_are_local_and_physically_floored():
    target_states = np.zeros((2, 72, 3, 6), dtype=np.float32)
    target_states[1, :, :, 0] = 0.005
    target_states[1, :, :, 1] = 0.005
    target_flows = np.zeros((2, 72, 2), dtype=np.float32)
    arrays = {"target_states": target_states, "target_actuator_flows": target_flows}
    scales = derive_effect_scales_v110(_FakeCache(arrays), ["D2::x"], _prepared())
    depth_threshold = scales.state_active_threshold[:, 0]
    assert depth_threshold[0] >= 0.01
    assert depth_threshold[2] >= 0.04
    assert depth_threshold[2] > depth_threshold[0]


def test_historical_v110_remains_importable_but_current_contract_is_v125():
    import rtc.step2_current as current

    assert current.CURRENT_STEP2_CONTRACT == "PROJECT7_V125_ANCHOR_DEFAULT_EVIDENCE_GATED_OVERRIDE_V1"
    assert current.CONTINUOUS_MPC_ENABLED is False
    assert current.HYDRAULIC_MODEL_REQUIRED_ONLINE is False
    assert current.CURRENT_POLICY_CLASS.__name__ == "AnchorOverridePolicyV125"
    source = current.__file__
    assert "step2_current" in source
