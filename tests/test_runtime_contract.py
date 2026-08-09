from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rtc.closed_loop import CausalObservation
from rtc.controller import ControllerConfig, TorchMPCController
from rtc.graph import GraphSchema
from rtc.models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from rtc.mpc import ContinuousSafetyMPC, StateLayout, _expand_control_blocks
from rtc.swmm_stats import write_node_statistics


def _controller() -> TorchMPCController:
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
        observed_dim=2, static_dim=6, state_dim=4, hidden_dim=8, graph_layers=1, context_dim=5
    )
    world = DifferentiableHydraulicWorldModel(
        state_dim=4, rainfall_dim=1, node_static_dim=6, actuator_physics_dim=6, hidden_dim=8
    )
    mpc = ContinuousSafetyMPC(
        world,
        layout=StateLayout(depth_index=0, flood_rate_index=2),
        priority_indices=torch.tensor([0]),
        dt_seconds=300,
        per_site_flood_budget_m3=10.0,
        per_site_depth_budget_m=0.1,
    )
    return TorchMPCController(
        step1=step1,
        mpc=mpc,
        graph=graph,
        sensor_nodes=("N1",),
        config=ControllerConfig(history_steps=3, horizon_steps=4, control_block_steps=2, optimizer_iterations=1),
        device="cpu",
    )


def _obs(t: int) -> CausalObservation:
    return CausalObservation(
        elapsed_seconds=t,
        current_time=datetime(2026, 1, 1, 0, t // 60),
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


def test_observe_cadence_is_separate_from_decide_and_no_duplicate_frame() -> None:
    controller = _controller()
    controller.observe(_obs(300))
    controller.observe(_obs(300))
    assert len(controller.observed_history) == 1
    controller.observe(_obs(600))
    assert len(controller.observed_history) == 2
    action = controller.decide(_obs(600), observation_already_recorded=True)
    assert action.source == "FALLBACK_HISTORY_WARMUP"
    assert len(controller.observed_history) == 2
    controller.observe(_obs(900))
    assert len(controller.observed_history) == 3
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.observe(_obs(600))


def test_control_blocks_repeat_without_changing_model_horizon() -> None:
    blocks = torch.tensor([[[0.1], [0.8]]])
    expanded = _expand_control_blocks(blocks, horizon=4, block_steps=2)
    assert expanded.shape == (1, 4, 1)
    assert torch.allclose(expanded[:, :, 0], torch.tensor([[0.1, 0.1, 0.8, 0.8]]))


def test_cumulative_node_statistics_are_converted_to_si(tmp_path: Path) -> None:
    path = tmp_path / "stats.csv.gz"
    write_node_statistics(
        path,
        start_statistics={"N1": {"flooding_volume": 10.0}},
        end_statistics={
            "N1": {
                "flooding_volume": 20.0,
                "peak_flooding_rate": 2.0,
                "max_depth": 3.0,
                "flooding_duration": 1.0,
                "surcharge_duration": 0.0,
            }
        },
        system_units="US",
        flow_units="CFS",
    )
    frame = pd.read_csv(path)
    assert np.isclose(frame.loc[0, "delta_flooding_volume_m3"], 10.0 * 0.028316846592)
    assert np.isclose(frame.loc[0, "peak_flooding_rate_m3s"], 2.0 * 0.028316846592)
    assert np.isclose(frame.loc[0, "max_depth_m"], 3.0 * 0.3048)
