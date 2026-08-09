from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rtc.baselines import write_passive_no_rtc_inp
from rtc.calibration import fit_sitewise_safety_calibration
from rtc.closed_loop import CausalObservation
from rtc.controller import ControllerConfig, TorchMPCController
from rtc.data_design import design_multi_actuator_rollouts
from rtc.forecast import PersistenceDecayForecast
from rtc.graph import GraphSchema, build_graph_schema
from rtc.inp import discover_actuators
from rtc.models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from rtc.mpc import ContinuousSafetyMPC, StateLayout
from rtc.pipeline import PipelineLedger, evidence_from_files
from rtc.runtime import choose_first_move
from rtc.splits import assign_rainfall_group_splits, verify_disjoint_rainfall_splits
from rtc.units import flow_rate_to_m3s, length_to_m, rainfall_rate_to_mmhr


SYNTHETIC_INP = """
[TITLE]
Scientific pipeline test
[OPTIONS]
FLOW_UNITS CMS
[JUNCTIONS]
P1 10 2 0 0 0
P2 9 2 0 0 0
N3 8 2 0 0 0
N4 7 2 0 0 0
[PUMPS]
PU1 P1 P2 CURVE1 ON
[ORIFICES]
OR1 P2 N3 SIDE 0 0.6 NO 0
[WEIRS]
WE1 N3 N4 TRANSVERSE 0 1.0 NO 0 0 NO
[OUTLETS]
OU1 N4 P1 0 TABULAR CURVE2 NO
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


def test_graph_compiler_keeps_all_actuators_continuous_and_bidirectional(tmp_path: Path) -> None:
    graph = build_graph_schema(_inp(tmp_path))
    assert graph.system_units == "SI"
    assert graph.actuator_ids == ("PU1", "OR1", "WE1", "OU1")
    assert graph.actuator_physics.shape == (4, 6)
    assert np.allclose(graph.actuator_physics[:, 4], 0.0)
    assert np.allclose(graph.actuator_physics[:, 5], 1.0)
    edges = set(map(tuple, graph.edge_index.T.tolist()))
    assert (0, 1) in edges and (1, 0) in edges
    assert graph.static_node_feature_names[:2] == ("invert_elevation_m", "max_depth_m")


def test_rainfall_group_split_has_zero_cross_role_leakage() -> None:
    frame = pd.DataFrame(
        {
            "rainfall_group": np.repeat([f"g{i:02d}" for i in range(20)], 3),
            "row": np.arange(60),
        }
    )
    assigned = assign_rainfall_group_splits(frame, seed=7)
    verify_disjoint_rainfall_splits(assigned)
    assert set(assigned["scientific_split"]) == {
        "development",
        "calibration",
        "safety_audit",
        "final",
    }
    assert assigned.groupby("rainfall_group")["scientific_split"].nunique().max() == 1


def test_sitewise_calibration_uses_truth_minus_prediction_residuals() -> None:
    pred_v = np.zeros((6, 2))
    true_v = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6]], dtype=float)
    pred_h = np.zeros((6, 2))
    true_h = true_v / 100.0
    cal = fit_sitewise_safety_calibration(
        priority_nodes=("A", "B"),
        predicted_flood_deterioration_m3=pred_v,
        true_flood_deterioration_m3=true_v,
        predicted_depth_deterioration_m=pred_h,
        true_depth_deterioration_m=true_h,
        rainfall_groups=np.array(["c1", "c1", "c2", "c2", "c3", "c3"]),
        coverage=0.8,
    )
    assert cal.calibration_sample_count == 6
    assert cal.flood_error_ucb_m3[0] >= 3.0
    assert cal.flood_error_ucb_m3[1] >= 5.0
    assert cal.contract == "SITEWISE_SPLIT_CONFORMAL_SAFETY_V1"


def test_inadmissible_mpc_first_move_fails_closed_to_fallback() -> None:
    decision = choose_first_move(
        optimized_sequence=np.array([[0.9, 0.1], [0.8, 0.2]]),
        surrogate_admissible=False,
        fallback_first_move=np.array([0.4, 0.6]),
        current_settings=np.array([0.4, 0.6]),
    )
    assert decision.source == "FALLBACK"
    assert np.allclose(decision.requested, [0.4, 0.6])


def test_passive_no_rtc_is_not_all_open(tmp_path: Path) -> None:
    source = _inp(tmp_path)
    passive = tmp_path / "passive.inp"
    write_passive_no_rtc_inp(source, passive)
    text = passive.read_text(encoding="utf-8")
    assert "disabled for PASSIVE_NO_RTC" in text
    assert "THEN PUMP PU1 SETTING = 0.7" not in text
    assert "PU1 P1 P2 CURVE1 ON" in text
    assert "SETTING = 1" not in text


def test_d3_design_never_freezes_a_fixed_active_subset(tmp_path: Path) -> None:
    catalog = discover_actuators(_inp(tmp_path))
    row = {"checkpoint_id": "e1:t30", "rainfall_group": "g1"}
    row.update({f"setting:{aid}": 0.5 for aid in catalog.ids})
    manifest = design_multi_actuator_rollouts(
        pd.DataFrame([row]),
        catalog,
        horizon_steps=3,
        sequences_per_checkpoint=2,
        seed=3,
    )
    assert manifest["all_actuators_eligible"].all()
    assert not manifest["fixed_active_subset"].any()
    for payload in manifest["settings_sequence_json"]:
        sequence = json.loads(payload)
        assert all(set(step) == set(catalog.ids) for step in sequence)


def test_pipeline_is_fail_closed_before_prerequisites(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    ledger = PipelineLedger()
    with pytest.raises(ValueError, match="prerequisite"):
        ledger.record(evidence_from_files("d0_d1_hydraulic_coverage", [evidence], passed=True))
    ledger.record(evidence_from_files("phase0_inp_timescale", [evidence], passed=True))
    ledger.record(evidence_from_files("d0_d1_hydraulic_coverage", [evidence], passed=True))
    assert ledger.passed("d0_d1_hydraulic_coverage")


def test_step1_context_is_a_real_model_input() -> None:
    model = SparseStateEstimator(
        observed_dim=2,
        static_dim=3,
        state_dim=4,
        hidden_dim=8,
        graph_layers=1,
        context_dim=5,
    )
    obs = torch.zeros(1, 3, 2, 2)
    mask = torch.ones_like(obs)
    static = torch.zeros(2, 3)
    edges = torch.tensor([[0, 1], [1, 0]])
    context = torch.randn(1, 3, 5)
    out = model(obs, mask, static, edges, context)
    assert out.shape == (1, 2, 4)
    with pytest.raises(ValueError, match="context_history"):
        model(obs, mask, static, edges)


def test_controller_warmup_returns_causal_hold_fallback() -> None:
    graph = GraphSchema(
        node_ids=("N1", "N2"),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        static_node_features=np.zeros((2, 6), dtype=np.float32),
        static_node_feature_names=(
            "invert_elevation_m", "max_depth_m", "is_junction", "is_outfall", "is_storage", "is_divider"
        ),
        actuator_ids=("A1",),
        actuator_upstream=np.array([0]),
        actuator_downstream=np.array([1]),
        actuator_physics=np.array([[1, 0, 0, 0, 0, 1]], dtype=np.float32),
        actuator_physics_feature_names=("is_pump", "is_orifice", "is_weir", "is_outlet", "min_setting", "max_setting"),
        system_units="SI",
    )
    step1 = SparseStateEstimator(
        observed_dim=2,
        static_dim=6,
        state_dim=4,
        hidden_dim=8,
        graph_layers=1,
        context_dim=5,
    )
    world = DifferentiableHydraulicWorldModel(
        state_dim=4,
        rainfall_dim=1,
        node_static_dim=6,
        actuator_physics_dim=6,
        hidden_dim=8,
    )
    mpc = ContinuousSafetyMPC(
        world,
        layout=StateLayout(depth_index=0, flood_rate_index=2),
        priority_indices=torch.tensor([0]),
        dt_seconds=300,
        per_site_flood_budget_m3=10.0,
        per_site_depth_budget_m=0.1,
    )
    controller = TorchMPCController(
        step1=step1,
        mpc=mpc,
        graph=graph,
        sensor_nodes=("N1",),
        forecast=PersistenceDecayForecast(scenario_multipliers=(1.0,)),
        config=ControllerConfig(history_steps=2, horizon_steps=2, optimizer_iterations=1),
        device="cpu",
    )
    obs = CausalObservation(
        elapsed_seconds=300,
        current_time=pd.Timestamp("2026-01-01T00:05:00").to_pydatetime(),
        sensor_ids=("N1",),
        sensor_depth_m=np.array([0.2]),
        sensor_head_m=np.array([10.2]),
        actuator_ids=("A1",),
        actuator_target_setting=np.array([0.4]),
        actuator_current_setting=np.array([0.4]),
        actuator_flow_m3s=np.array([0.1]),
        rainfall_node_ids=("N1", "N2"),
        observed_rainfall_mmhr=np.array([5.0, 0.0]),
    )
    action = controller(obs)
    assert action.source == "FALLBACK_HISTORY_WARMUP"
    assert np.isclose(action.settings["A1"], 0.4)
