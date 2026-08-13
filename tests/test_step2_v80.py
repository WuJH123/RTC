from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import HydraulicResponseSurrogateV70
from rtc.step2_control_response_v80 import (
    CausalPrefixActionProjectorV80,
    DirectHydraulicEffectSurrogateV80,
    ZeroPreservingGraphBlockV80,
    prepare_static_v80,
)
from rtc.step2_v80_contract import DirectHydraulicEffectLossContractV80


def _graph():
    nodes, actuators = 14, 109
    edges = []
    for i in range(nodes - 1):
        edges.extend(((i, i + 1), (i + 1, i)))
    node_names = (
        "invert_elevation_m", "max_depth_m", "is_junction", "is_outfall", "is_storage",
        "is_divider", "init_depth_m", "surcharge_depth_m", "ponded_area_m2",
        "storage_capacity_m3", "storage_area_full_m2", "conduit_in_count",
        "conduit_out_count", "conduit_in_length_sum_m", "conduit_out_length_sum_m",
        "conduit_in_roughness_mean", "conduit_out_roughness_mean",
        "conduit_in_geom1_mean_m", "conduit_out_geom1_mean_m", "subcatchment_count",
        "subcatchment_area_m2", "subcatchment_impervious_area_m2",
        "subcatchment_width_area_weighted_m", "subcatchment_slope_area_weighted_pct",
        "infiltration_max_rate_area_weighted_mmhr", "infiltration_min_rate_area_weighted_mmhr",
    )
    static = np.zeros((nodes, len(node_names)), np.float32)
    static[:, 1] = 3.0
    static[:, 2] = 1.0
    static[:, 7] = 0.5
    static[0, 2] = 0.0
    static[0, 4] = 1.0
    static[0, 9] = 1000.0
    physics_names = (
        "is_pump", "is_orifice", "is_weir", "is_outlet", "min_setting", "max_setting",
        "pump_curve_max_flow_m3s", "pump_curve_max_x_m", "pump_curve_point_count",
        "offset_or_crest_m", "discharge_coefficient", "has_flap_gate",
        "xsection_geom1_m", "xsection_geom2_m", "xsection_geom3_m", "xsection_geom4_m",
        "xsection_is_circular", "xsection_is_rect_closed", "xsection_is_rect_open",
    )
    physics = np.zeros((actuators, len(physics_names)), np.float32)
    physics[:, 5] = 1.0
    physics[:57, 0] = 1.0
    physics[57:99, 1] = 1.0
    physics[99:, 2] = 1.0
    physics[:57, 6] = 2.0
    return SimpleNamespace(
        node_ids=tuple(f"n{i}" for i in range(nodes)),
        edge_index=np.asarray(edges, dtype=np.int64).T,
        static_node_features=static,
        static_node_feature_names=node_names,
        actuator_ids=tuple(f"a{i}" for i in range(actuators)),
        actuator_upstream=np.arange(actuators) % nodes,
        actuator_downstream=(np.arange(actuators) + 1) % nodes,
        actuator_physics=physics,
        actuator_physics_feature_names=physics_names,
        system_units="SI",
    )


def _model():
    graph = _graph()
    prepared = prepare_static_v60(graph)
    reference = HydraulicResponseSurrogateV70(
        state_scale=torch.ones(6),
        flow_scale=torch.ones(109),
        state_dim=6,
        rainfall_dim=1,
        node_static_dim=graph.static_node_features.shape[1],
        physics_dim=prepared.actuator_physics.shape[1],
        actuator_count=109,
        hidden_dim=16,
        latent_dim=8,
        temporal_dim=6,
    )
    basis = build_control_basis_v60(graph)
    contract = DirectHydraulicEffectLossContractV80(hidden_dim=16, graph_blocks=2)
    model = DirectHydraulicEffectSurrogateV80(
        reference_model=reference,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=torch.tensor([0.1, 0.1, 0.3, 30.0, 1.0, 1.0]),
        flow_delta_scale=torch.ones(109) * 0.1,
        physics_dim=prepared.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        contract=contract,
    )
    return graph, basis, model


def _inputs(candidate_count=3):
    initial = torch.zeros(1, 14, 6)
    initial[..., 0] = 0.3
    rain = torch.zeros(1, 72, 14, 1)
    rain[:, 10:30, :, 0] = 0.2
    reference = torch.full((1, 72, 109), 0.4)
    candidate = reference[:, None].expand(1, candidate_count, 72, 109).clone()
    if candidate_count > 1:
        candidate[:, 1, 4:20, 0] = 0.7
    if candidate_count > 2:
        candidate[:, 2, 20:50, 57] = 0.1
    previous_flow = torch.linspace(-0.5, 0.5, 109)[None]
    return initial, rain, reference, candidate, previous_flow


def test_v80_reference_is_frozen_and_same_action_is_exact_zero():
    graph, _, model = _model()
    prepared = prepare_static_v80(graph)
    initial, rain, reference, _, previous_flow = _inputs(1)
    output = model(
        initial, rain, reference, reference[:, None], previous_flow, prepared
    )
    assert not any(parameter.requires_grad for parameter in model.reference_model.parameters())
    assert torch.equal(output.delta_states_physical, torch.zeros_like(output.delta_states_physical))
    assert torch.equal(output.delta_flows_physical, torch.zeros_like(output.delta_flows_physical))
    assert torch.equal(
        output.candidate_states_physical, output.reference_states_physical
    )


def test_v80_direct_effect_has_finite_nonzero_action_gradient():
    graph, _, model = _model()
    prepared = prepare_static_v80(graph)
    initial, rain, reference, candidate, previous_flow = _inputs(2)
    action = candidate[:, 1:2].clone().requires_grad_(True)
    output = model(initial, rain, reference, action, previous_flow, prepared)
    objective = (
        output.delta_states_physical[..., 0].square().sum()
        + output.delta_states_physical[..., 2].square().sum()
        + output.delta_states_physical[..., 3].square().sum()
        + output.delta_flows_physical.square().sum()
    )
    gradient = torch.autograd.grad(objective, action)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_v80_causal_prefix_rejects_future_action_leakage():
    graph, basis, _ = _model()
    projector = CausalPrefixActionProjectorV80(
        basis.temporal_basis, control_block_steps=2
    )
    action = torch.zeros(1, 1, 72, 109)
    # A change starting at model step 40 must not affect a retained time at step 10.
    action[:, :, 40:, 0] = 0.5
    features = projector(action, torch.tensor([10, 50]))
    assert torch.equal(features[:, :, 0, 0], torch.zeros_like(features[:, :, 0, 0]))
    assert torch.count_nonzero(features[:, :, 1, 0]) > 0


def test_v80_graph_block_preserves_zero_and_propagates_local_effect():
    graph = _graph()
    prepared = prepare_static_v80(graph)
    block = ZeroPreservingGraphBlockV80(8)
    zeros = torch.zeros(1, 14, 8)
    assert torch.equal(
        block(zeros, prepared.edge_index, prepared.node_degree), zeros
    )
    local = torch.zeros(1, 14, 8)
    local[:, 0, 0] = 1.0
    propagated = block(local, prepared.edge_index, prepared.node_degree)
    assert torch.count_nonzero(propagated[:, 1]) > 0


def test_v80_previous_flow_is_an_active_causal_input():
    graph, _, model = _model()
    prepared = prepare_static_v80(graph)
    initial, rain, reference, candidate, previous_flow = _inputs(2)
    first = model(
        initial, rain, reference, candidate[:, 1:2], previous_flow, prepared
    ).delta_states_physical
    changed = previous_flow.clone()
    changed[:, 0] += 2.0
    second = model(
        initial, rain, reference, candidate[:, 1:2], changed, prepared
    ).delta_states_physical
    assert not torch.allclose(first, second)
