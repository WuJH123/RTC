from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rtc.baselines import write_no_control_inp
from rtc.calibration import fit_sitewise_safety_calibration
from rtc.closed_loop import CausalObservation
from rtc.context_features import build_node_context
from rtc.controller import ControllerConfig, TorchMPCController
from rtc.data_design import design_multi_actuator_rollouts
from rtc.dataset_compile import compile_branch_tensors
from rtc.forecast import PersistenceDecayForecast
from rtc.graph import GraphSchema, build_graph_schema
from rtc.inp import discover_actuators
from rtc.inp_runtime import build_runtime_inp, section_has_payload
from rtc.metrics import compile_event_metrics
from rtc.models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from rtc.mpc import ContinuousSafetyMPC, StateLayout
from rtc.pipeline import PipelineLedger, evidence_from_files
from rtc.runtime import choose_first_move
from rtc.splits import assign_rainfall_group_splits, verify_disjoint_rainfall_splits
from rtc.tfv_mpc import ContinuousTFVFirstMPC
from rtc.units import flow_rate_to_m3s, length_to_m, rainfall_rate_to_mmhr


SYNTHETIC_INP = """
[TITLE]
Scientific pipeline test
[OPTIONS]
FLOW_UNITS CMS
THREADS 2
[JUNCTIONS]
P1 10 2 0 0 0
P2 9 2 0 0 0
N3 8 2 0 0 0
N4 7 2 0 0 0
[PUMPS]
PU1 P1 P2 CURVE1 ON
[ORIFICES]
OR1 P2 N3 SIDE 0.4 0.6 NO 0
[WEIRS]
WE1 N3 N4 TRANSVERSE 0.7 1.2 NO 0 0 NO
[OUTLETS]
OU1 N4 P1 0 TABULAR CURVE2 NO
[XSECTIONS]
OR1 CIRCULAR 1.2 0 0 0 1
WE1 RECT_OPEN 0.5 2.0 0 0 1
[CURVES]
CURVE1 PUMP2 0 2.0
CURVE1 2 1.5
CURVE1 4 0.8
CURVE2 RATING 0 0
CURVE2 1 1
[CONTROLS]
RULE R1
IF NODE P1 DEPTH > 1
THEN PUMP PU1 SETTING = 0.7
"""


def _inp(tmp_path: Path) -> Path:
    path = tmp_path / "network.inp"
    path.write_text(SYNTHETIC_INP, encoding="utf-8")
    return path


def test_units_are_explicitly_normalized_to_si() -> None:
    assert np.isclose(float(flow_rate_to_m3s(1.0, "CFS")), 0.028316846592)
    assert np.isclose(float(length_to_m(1.0, "US")), 0.3048)
    assert np.isclose(float(rainfall_rate_to_mmhr(1.0, "US")), 25.4)


def test_graph_compiler_keeps_all_actuators_and_hydraulic_identity(tmp_path: Path) -> None:
    graph = build_graph_schema(_inp(tmp_path))
    assert graph.system_units == "SI"
    assert graph.actuator_ids == ("PU1", "OR1", "WE1", "OU1")
    assert graph.actuator_physics.shape == (4, 19)
    assert np.allclose(graph.actuator_physics[:, 4], 0.0)
    assert np.allclose(graph.actuator_physics[:, 5], 1.0)
    pump = graph.actuator_ids.index("PU1")
    assert np.isclose(graph.actuator_physics[pump, 6], 2.0)
    assert np.isclose(graph.actuator_physics[pump, 7], 4.0)
    assert graph.actuator_physics[pump, 8] == 3
    orifice = graph.actuator_ids.index("OR1")
    assert np.isclose(graph.actuator_physics[orifice, 9], 0.4)
    assert np.isclose(graph.actuator_physics[orifice, 10], 0.6)
    assert np.isclose(graph.actuator_physics[orifice, 12], 1.2)
    edges = set(map(tuple, graph.edge_index.T.tolist()))
    assert (0, 1) in edges and (1, 0) in edges
    assert graph.static_node_feature_names[:2] == ("invert_elevation_m", "max_depth_m")


def test_no_control_and_internal_rtc_are_distinct(tmp_path: Path) -> None:
    source = _inp(tmp_path)
    no_control = tmp_path / "no_control.inp"
    internal = tmp_path / "internal.inp"
    write_no_control_inp(source, no_control, swmm_threads=1)
    build_runtime_inp(source, internal, native_controls=True, swmm_threads=1)
    assert not section_has_payload(no_control, "CONTROLS")
    assert section_has_payload(internal, "CONTROLS")
    assert "PU1 P1 P2 CURVE1 ON" in no_control.read_text(encoding="utf-8")
    assert "SETTING = 1" not in no_control.read_text(encoding="utf-8")
    assert "THREADS             1" in no_control.read_text(encoding="utf-8")


def test_rainfall_group_split_has_zero_cross_role_leakage() -> None:
    frame = pd.DataFrame({
        "rainfall_group": np.repeat([f"g{i:02d}" for i in range(20)], 3),
        "row": np.arange(60),
    })
    assigned = assign_rainfall_group_splits(frame, seed=7)
    verify_disjoint_rainfall_splits(assigned)
    assert set(assigned["scientific_split"]) == {"development", "calibration", "safety_audit", "final"}
    assert assigned.groupby("rainfall_group")["scientific_split"].nunique().max() == 1


def test_sitewise_calibration_uses_truth_minus_prediction_residuals() -> None:
    pred_v = np.zeros((6, 2))
    true_v = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6]], dtype=float)
    cal = fit_sitewise_safety_calibration(
        priority_nodes=("A", "B"),
        predicted_flood_deterioration_m3=pred_v,
        true_flood_deterioration_m3=true_v,
        predicted_depth_deterioration_m=np.zeros((6, 2)),
        true_depth_deterioration_m=true_v / 100.0,
        rainfall_groups=np.array(["c1", "c1", "c2", "c2", "c3", "c3"]),
        coverage=0.8,
    )
    assert cal.flood_error_ucb_m3[0] >= 3.0
    assert cal.flood_error_ucb_m3[1] >= 5.0


def test_legacy_hard_gate_runtime_still_falls_back_when_explicitly_requested() -> None:
    decision = choose_first_move(
        optimized_sequence=np.array([[0.9, 0.1], [0.8, 0.2]]),
        surrogate_admissible=False,
        fallback_first_move=np.array([0.4, 0.6]),
        current_settings=np.array([0.4, 0.6]),
    )
    assert decision.source == "FALLBACK"


def test_tfv_first_priority_deterioration_is_not_a_hard_veto() -> None:
    torch.manual_seed(2)
    world = DifferentiableHydraulicWorldModel(
        state_dim=4, rainfall_dim=1, node_static_dim=2, actuator_physics_dim=2, hidden_dim=8
    )
    mpc = ContinuousTFVFirstMPC(
        world,
        depth_index=0,
        flood_rate_index=2,
        priority_indices=torch.tensor([0]),
        dt_seconds=300,
        flood_error_ucb_m3=1000.0,
        depth_error_ucb_m=1.0,
        forecast_quantile=0.9,
        tfv_cvar_alpha=0.9,
    )
    result = mpc.optimize(
        initial_state=torch.zeros(1, 2, 4),
        rainfall_scenarios=torch.zeros(1, 2, 2, 1),
        current_settings=torch.tensor([0.5]),
        fallback_settings=torch.full((1, 2, 1), 0.5),
        previous_actuator_flow=torch.zeros(1, 1),
        actuator_upstream=torch.tensor([0]),
        actuator_downstream=torch.tensor([1]),
        actuator_physics=torch.zeros(1, 1, 2),
        static_node_features=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        iterations=2,
        learning_rate=0.01,
    )
    assert result.candidate_valid
    assert result.priority_positive_flood_deterioration_m3 > 0.0


def test_node_local_context_does_not_broadcast_all_actuators() -> None:
    context = build_node_context(
        rainfall_mmhr=np.array([[1.0], [0.0], [2.0]], dtype=np.float32),
        actuator_setting=np.array([0.2, 0.8], dtype=np.float32),
        actuator_flow_m3s=np.array([1.0, -0.5], dtype=np.float32),
        actuator_upstream=np.array([0, 1]),
        actuator_downstream=np.array([1, 2]),
        node_count=3,
    )
    assert context.shape == (3, 5)
    assert np.isclose(context[0, 1], 0.2)
    assert np.isclose(context[1, 2], 0.2)
    assert np.isclose(context[1, 1], 0.8)
    assert np.isclose(context[1, 3], -0.5)


def test_sampled_global_peak_is_synchronous_network_sum() -> None:
    frame = pd.DataFrame({
        "elapsed_seconds": [0, 0, 300, 300],
        "node_id": ["A", "B", "A", "B"],
        "depth": [1.0, 1.0, 1.0, 1.0],
        "flooding": [1.0, 2.0, 4.0, 0.5],
    })
    metrics = compile_event_metrics(
        frame, priority_nodes=("A",), flow_units="CMS", system_units="SI", post_action_only=False
    )
    assert np.isclose(metrics.global_peak_flood_rate_m3s, 4.5)


def test_compact_branch_preserves_interval_alignment_and_exact_volume(tmp_path: Path) -> None:
    compact = tmp_path / "b.compact.npz"
    np.savez_compressed(
        compact,
        elapsed_seconds=np.array([600, 900, 1200]),
        node_ids=np.array(["N1", "N2"]),
        state_si=np.zeros((3, 2, 6), dtype=np.float32),
        state_channels=np.array(["depth_m", "head_m", "flooding_m3s", "volume_m3", "total_inflow_m3s", "total_outflow_m3s"]),
        rainfall_mmhr=np.zeros((3, 2, 1), dtype=np.float32),
        actuator_ids=np.array(["A1"]),
        target_setting=np.full((3, 1), 0.5, dtype=np.float32),
        current_setting=np.full((3, 1), 0.5, dtype=np.float32),
        actuator_flow_m3s=np.zeros((3, 1), dtype=np.float32),
        candidate_setting=np.array([0.7], dtype=np.float32),
    )
    stats = pd.DataFrame({"node_id": ["N1", "N2"], "delta_flooding_volume_m3": [12.0, 3.0]})
    stats_path = tmp_path / "b.node_statistics.csv.gz"
    stats.to_csv(stats_path, index=False, compression="gzip")
    meta = tmp_path / "b.json"
    meta.write_text(json.dumps({
        "compact_file": compact.name,
        "node_statistics_file": stats_path.name,
        "candidate_action_sha256": "x",
    }), encoding="utf-8")
    branch = compile_branch_tensors(meta)
    assert branch.settings.shape == (2, 1)
    assert np.allclose(branch.settings[:, 0], 0.7)
    assert branch.rainfall.shape == (2, 2, 1)
    assert branch.target_states.shape == (2, 2, 6)
    assert np.allclose(branch.exact_node_flood_volume_m3, [12.0, 3.0])


def test_d3_design_never_freezes_a_fixed_active_subset(tmp_path: Path) -> None:
    catalog = discover_actuators(_inp(tmp_path))
    row = {"checkpoint_id": "e1:t30", "rainfall_group": "g1"}
    row.update({f"setting:{aid}": 0.5 for aid in catalog.ids})
    manifest = design_multi_actuator_rollouts(
        pd.DataFrame([row]), catalog, horizon_steps=3, sequences_per_checkpoint=2, seed=3
    )
    assert manifest["all_actuators_eligible"].all()
    assert not manifest["fixed_active_subset"].any()


def test_pipeline_is_fail_closed_before_prerequisites(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    ledger = PipelineLedger()
    with pytest.raises(ValueError, match="prerequisite"):
        ledger.record(evidence_from_files("d0_d1_hydraulic_coverage", [evidence], passed=True))
    ledger.record(evidence_from_files("phase0_inp_timescale", [evidence], passed=True))
    ledger.record(evidence_from_files("d0_d1_hydraulic_coverage", [evidence], passed=True))


def test_step1_context_is_a_real_node_local_model_input() -> None:
    model = SparseStateEstimator(
        observed_dim=2, static_dim=3, state_dim=4, hidden_dim=8, graph_layers=1, context_dim=5
    )
    obs = torch.zeros(1, 3, 2, 2)
    mask = torch.ones_like(obs)
    static = torch.zeros(2, 3)
    edges = torch.tensor([[0, 1], [1, 0]])
    context = torch.randn(1, 3, 2, 5)
    out = model(obs, mask, static, edges, context)
    assert out.shape == (1, 2, 4)
    with pytest.raises(ValueError, match="context_history"):
        model(obs, mask, static, edges)


def test_controller_warmup_returns_causal_hold_fallback() -> None:
    graph = GraphSchema(
        node_ids=("N1", "N2"), edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        static_node_features=np.zeros((2, 6), dtype=np.float32),
        static_node_feature_names=("invert_elevation_m", "max_depth_m", "is_junction", "is_outfall", "is_storage", "is_divider"),
        actuator_ids=("A1",), actuator_upstream=np.array([0]), actuator_downstream=np.array([1]),
        actuator_physics=np.array([[1, 0, 0, 0, 0, 1]], dtype=np.float32),
        actuator_physics_feature_names=("is_pump", "is_orifice", "is_weir", "is_outlet", "min_setting", "max_setting"),
        system_units="SI",
    )
    step1 = SparseStateEstimator(observed_dim=2, static_dim=6, state_dim=4, hidden_dim=8, graph_layers=1, context_dim=5)
    world = DifferentiableHydraulicWorldModel(state_dim=4, rainfall_dim=1, node_static_dim=6, actuator_physics_dim=6, hidden_dim=8)
    mpc = ContinuousSafetyMPC(
        world, layout=StateLayout(depth_index=0, flood_rate_index=2), priority_indices=torch.tensor([0]),
        dt_seconds=300, per_site_flood_budget_m3=10.0, per_site_depth_budget_m=0.1,
    )
    controller = TorchMPCController(
        step1=step1, mpc=mpc, graph=graph, sensor_nodes=("N1",),
        forecast=PersistenceDecayForecast(scenario_multipliers=(1.0,)),
        config=ControllerConfig(history_steps=2, horizon_steps=2, optimizer_iterations=1), device="cpu",
    )
    obs = CausalObservation(
        elapsed_seconds=300, current_time=pd.Timestamp("2026-01-01T00:05:00").to_pydatetime(),
        sensor_ids=("N1",), sensor_depth_m=np.array([0.2]), sensor_head_m=np.array([10.2]),
        actuator_ids=("A1",), actuator_target_setting=np.array([0.4]), actuator_current_setting=np.array([0.4]),
        actuator_flow_m3s=np.array([0.1]), rainfall_node_ids=("N1", "N2"), observed_rainfall_mmhr=np.array([5.0, 0.0]),
    )
    action = controller(obs)
    assert action.source == "FALLBACK_HISTORY_WARMUP"
    assert np.isclose(action.settings["A1"], 0.4)
