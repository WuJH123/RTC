from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch

from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import (
    ControlValueSurrogateV60,
    DualStep2SurrogateV60,
    HydraulicResponseSurrogateV60,
    prepare_static_v60,
)
from rtc.step2_d3_design_v60 import D3V60DesignContract, design_targeted_d3_v60
from rtc.step2_d3_lineage_v60 import stamp_d3_v60_lineage
from rtc.step2_optimization_v60 import event_balance_summary_v60
from rtc.step2_train_response_v60 import (
    TargetScalesV60,
    hydraulic_critical_weights_v60,
    listwise_loss_v60,
)


def _graph():
    nodes, actuators = 14, 109
    edges = []
    for index in range(nodes - 1):
        edges += [(index, index + 1), (index + 1, index)]
    names = (
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
    static = np.zeros((nodes, len(names)), np.float32)
    static[:, 1] = 3.0
    static[:, 2] = 1.0
    static[:, 7] = 0.5
    static[0, 4] = 1.0
    static[0, 2] = 0.0
    static[0, 9] = 1000.0
    static[1, 9] = 500.0
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
        static_node_feature_names=names,
        actuator_ids=tuple(f"a{i}" for i in range(actuators)),
        actuator_upstream=np.arange(actuators) % nodes,
        actuator_downstream=(np.arange(actuators) + 1) % nodes,
        actuator_physics=physics,
        actuator_physics_feature_names=physics_names,
        system_units="SI",
    )


def _models(graph):
    prepared = prepare_static_v60(graph)
    common = dict(
        state_dim=6,
        rainfall_dim=1,
        node_static_dim=graph.static_node_features.shape[1],
        physics_dim=prepared.actuator_physics.shape[1],
        actuator_count=109,
        hidden_dim=16,
        latent_dim=8,
        temporal_dim=6,
    )
    return (
        ControlValueSurrogateV60(tfv_rate_scale_m3s=10.0, **common),
        HydraulicResponseSurrogateV60(
            state_scale=torch.ones(6), flow_scale=torch.ones(109), **common
        ),
        prepared,
    )


def _inputs(candidate_count=3):
    initial = torch.zeros(1, 14, 6)
    initial[..., 0] = 0.3
    rainfall = torch.zeros(1, 72, 14, 1)
    rainfall[:, 10:30, :, 0] = 0.2
    reference = torch.full((1, 72, 109), 0.4)
    candidate = reference[:, None].expand(1, candidate_count, 72, 109).clone()
    candidate[:, 1, 4:20, 0] = 0.7
    if candidate_count > 2:
        candidate[:, 2, 10:40, 57] = 0.1
    return initial, rainfall, reference, candidate, torch.arange(73, dtype=torch.float32)[None] * 300


def test_v60_multi_resolution_and_low_dimensional_basis():
    horizon = MultiResolutionHorizonV60()
    assert horizon.indices()[-1] == 71
    assert 10 < len(horizon.indices()) < 72
    basis = build_control_basis_v60(_graph())
    assert basis.coefficient_dimension < 109 * 36
    reference = torch.full((1, 2, 72, 109), 0.5)
    coefficient = torch.zeros(
        1, 2, basis.temporal_basis_count, basis.group_count, requires_grad=True
    )
    with torch.no_grad():
        coefficient[0, 1, 0, 0] = 0.7
    output = basis.decode(reference, coefficient)
    blocks = output[..., ::2, :]
    assert output.shape == (1, 2, 72, 109)
    assert output.min() >= 0 and output.max() <= 1
    assert torch.max(torch.abs(blocks[..., 1:, :] - blocks[..., :-1, :])) <= 0.500001


def test_v60_targeted_d3_is_sparse_group_structured_unique_and_lineage_bound():
    graph = _graph()
    basis = build_control_basis_v60(graph)
    row = {
        "checkpoint_id": "c0", "event_id": "e0", "rainfall_group": "r0",
        "scientific_split": "development", "development_fold": "train",
        "checkpoint_minutes": 60, "inp_path": "x", "trajectory_metadata_path": "y",
    }
    for actuator_id in graph.actuator_ids:
        row[f"setting:{actuator_id}"] = 0.5
    contract = D3V60DesignContract(candidates_per_checkpoint=24)
    frame = design_targeted_d3_v60(pd.DataFrame([row]), graph, basis, contract=contract)
    frame, lineage = stamp_d3_v60_lineage(frame, basis=basis, design_contract=contract)
    assert len(frame) == 25
    assert (frame.data_role == "D3_HOLD_REFERENCE").sum() == 1
    assert frame.sequence_sha256.nunique() == 25
    candidate = frame[frame.data_role != "D3_HOLD_REFERENCE"]
    assert candidate.active_control_groups.max() <= 5
    assert candidate.active_actuators.median() < 109
    assert all(len(json.loads(value)) == 36 for value in frame.settings_sequence_json)
    assert frame.v60_control_basis_sha256.nunique() == 1
    assert frame.v60_design_contract_sha256.nunique() == 1
    assert lineage["v60_control_basis_sha256"] == frame.v60_control_basis_sha256.iloc[0]


def test_v60_targeted_d3_accepts_aligned_checkpoint_elapsed_seconds():
    graph = _graph()
    basis = build_control_basis_v60(graph)
    row = {
        "checkpoint_id": "aligned", "event_id": "e0", "rainfall_group": "r0",
        "scientific_split": "development", "development_fold": "train",
        "checkpoint_elapsed_seconds": 3600, "checkpoint_minutes": 60,
        "inp_path": "x", "trajectory_metadata_path": "y",
    }
    for actuator_id in graph.actuator_ids:
        row[f"setting:{actuator_id}"] = 0.5
    frame = design_targeted_d3_v60(pd.DataFrame([row]), graph, basis)
    assert len(frame) == 25


def test_v60_targeted_d3_rejects_unaligned_checkpoint_elapsed_seconds():
    graph = _graph()
    basis = build_control_basis_v60(graph)
    row = {
        "checkpoint_id": "unaligned", "event_id": "e0", "rainfall_group": "r0",
        "scientific_split": "development", "development_fold": "train",
        "checkpoint_elapsed_seconds": 3960, "checkpoint_minutes": 66,
        "inp_path": "x", "trajectory_metadata_path": "y",
    }
    for actuator_id in graph.actuator_ids:
        row[f"setting:{actuator_id}"] = 0.5
    with pytest.raises(ValueError, match="align with model_step_seconds"):
        design_targeted_d3_v60(pd.DataFrame([row]), graph, basis)


def test_v60_state_conditioning_exact_zero_causality_and_disjoint_surrogates():
    torch.manual_seed(3)
    graph = _graph()
    value, hydraulic, prepared = _models(graph)
    initial, rainfall, reference, candidate, elapsed = _inputs()
    zero = value(initial, rainfall, reference, reference[:, None], prepared, elapsed)
    assert torch.equal(zero.delta_tfv_m3, torch.zeros_like(zero.delta_tfv_m3))
    hydraulic_zero = hydraulic(initial, rainfall, reference, reference[:, None], prepared)
    assert torch.equal(
        hydraulic_zero.delta_states_physical,
        torch.zeros_like(hydraulic_zero.delta_states_physical),
    )
    first = value(initial, rainfall, reference, candidate, prepared, elapsed)
    changed = initial.clone()
    changed[:, 0, 0] = 1.8
    second = value(changed, rainfall, reference, candidate, prepared, elapsed)
    assert not torch.allclose(
        first.joint_context_before_scatter, second.joint_context_before_scatter
    )
    future = candidate.clone()
    future[:, 1, 40:, 2] = 0.9
    third = value(initial, rainfall, reference, future, prepared, elapsed)
    assert torch.allclose(
        first.delta_tfv_prefix_m3[:, :, :40],
        third.delta_tfv_prefix_m3[:, :, :40],
        atol=1e-5,
        rtol=1e-5,
    )
    DualStep2SurrogateV60(value, hydraulic).assert_disjoint_parameters()


def test_v60_critical_hydraulic_weighting_and_source_scales():
    graph = _graph()
    _, _, prepared = _models(graph)
    states = torch.zeros(1, 2, 3, 14, 6)
    states[..., 0] = 0.1
    baseline = hydraulic_critical_weights_v60(states, prepared)
    critical = states.clone()
    critical[..., 0, 0] = 2.5
    critical[..., 0, 3] = 900.0
    weighted = hydraulic_critical_weights_v60(critical, prepared)
    assert weighted[..., 0].mean() > baseline[..., 0].mean()
    predicted = torch.tensor([[3.0, 1.0, 2.0]], requires_grad=True)
    truth = torch.tensor([[30.0, 10.0, 20.0]])
    loss = listwise_loss_v60(predicted, truth, 100.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(predicted.grad).all()
    scales = TargetScalesV60(np.ones(6), np.ones(109), 10.0, 100.0, 1.0)
    assert scales.tfv_scale("D2") == 10.0 and scales.tfv_scale("D3") == 100.0


def test_v60_event_balance_is_explicit_not_accidental_group_balance():
    entries = {
        "D3::r0::e0::c0": SimpleNamespace(rainfall_group="r0", event_id="e0"),
        "D3::r0::e0::c1": SimpleNamespace(rainfall_group="r0", event_id="e0"),
        "D3::r1::e1::c0": SimpleNamespace(rainfall_group="r1", event_id="e1"),
    }
    cache = SimpleNamespace(entry=lambda name: entries[name])
    summary = event_balance_summary_v60(cache, list(entries))
    assert summary["events"] == 2
    assert summary["groups_per_event"] == {"min": 1, "median": 1.5, "max": 2}
    assert summary["optimizer_step_unit"] == "event"


def test_v60_mpc_value_is_differentiable_in_control_coefficients():
    torch.manual_seed(4)
    graph = _graph()
    basis = build_control_basis_v60(graph)
    value, _, prepared = _models(graph)
    initial, rainfall, reference, _, elapsed = _inputs(2)
    coefficient = torch.zeros(
        1, 2, basis.temporal_basis_count, basis.group_count, requires_grad=True
    )
    with torch.no_grad():
        coefficient[0, 1, 0, 0] = 0.5
    candidate = basis.decode(reference[:, None].expand(1, 2, -1, -1), coefficient)
    value(initial, rainfall, reference, candidate, prepared, elapsed).delta_tfv_m3.sum().backward()
    assert coefficient.grad is not None
    assert torch.isfinite(coefficient.grad).all()
    assert torch.count_nonzero(coefficient.grad) > 0
