from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rtc.code_contract import rtc_source_tree_sha256
from rtc.control_leverage import build_control_leverage_report
from rtc.efficient_probe_design import (
    design_budgeted_independent_probes,
    summarise_budgeted_probe_design,
)
from rtc.inp import Actuator, ActuatorCatalog
from rtc.models import DifferentiableHydraulicWorldModel, Rollout
from rtc.production_cli import _load_step2
from rtc.robust_tfv_mpc import ContinuousTFVFirstMPC


def test_current_step2_training_contract_loads_in_production(tmp_path: Path) -> None:
    model = DifferentiableHydraulicWorldModel(
        state_dim=4,
        rainfall_dim=1,
        node_static_dim=2,
        actuator_physics_dim=2,
        hidden_dim=8,
        actuator_count=1,
        actuator_embedding_dim=4,
    )
    path = tmp_path / "step2.pt"
    torch.save(
        {
            "checkpoint_contract": "RTC_TORCH_CHECKPOINT_V2_CODE_BOUND",
            "rtc_source_tree_sha256": rtc_source_tree_sha256(),
            "state_dict": model.state_dict(),
            "model_config": {
                "state_dim": 4,
                "rainfall_dim": 1,
                "node_static_dim": 2,
                "actuator_physics_dim": 2,
                "hidden_dim": 8,
                "actuator_count": 1,
                "actuator_embedding_dim": 4,
                "model_step_seconds": 300,
                "horizon_steps": 2,
                "swmm_engine_version": "5.2.4",
                "time_contract": "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2",
                "training_contract_sha256": "train",
            },
            "training_manifest_sha256": "manifest",
            "scientific_split": "development",
        },
        path,
    )
    loaded = _load_step2(path, torch.device("cpu"))
    assert loaded.runtime_metadata["time_contract"] == "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2"
    assert loaded.runtime_metadata["swmm_engine_version"] == "5.2.4"
    assert loaded.runtime_metadata["model_step_seconds"] == 300
    assert loaded.runtime_metadata["horizon_steps"] == 2


class _ToyWorld(nn.Module):
    def __init__(self, *, responsive: bool) -> None:
        super().__init__()
        self.responsive = responsive

    def rollout(
        self,
        initial_state,
        rainfall,
        settings,
        previous_actuator_flow,
        actuator_upstream,
        actuator_downstream,
        actuator_physics,
        static_node_features,
        edge_index,
    ) -> Rollout:
        batch, horizon, _ = settings.shape
        states = initial_state[:, None].expand(-1, horizon, -1, -1).clone()
        if self.responsive:
            flood = 1.0 - settings[..., 0]
        else:
            flood = 1.0 + 0.0 * settings[..., 0]
        states[..., 0, 2] = flood
        flows = settings.clone()
        responsiveness = torch.ones_like(settings)
        return Rollout(states=states, actuator_flows=flows, responsiveness=responsiveness)


class _PriorityTradeoffWorld(nn.Module):
    """Higher setting lowers total flood but increases flooding at priority node 1."""

    def rollout(
        self,
        initial_state,
        rainfall,
        settings,
        previous_actuator_flow,
        actuator_upstream,
        actuator_downstream,
        actuator_physics,
        static_node_features,
        edge_index,
    ) -> Rollout:
        _, horizon, _ = settings.shape
        states = initial_state[:, None].expand(-1, horizon, -1, -1).clone()
        setting = settings[..., 0]
        states[..., 0, 2] = 1.0 - setting
        states[..., 1, 2] = 0.1 + 0.2 * setting
        flows = settings.clone()
        return Rollout(
            states=states,
            actuator_flows=flows,
            responsiveness=torch.ones_like(settings),
        )


def _mpc_inputs(horizon: int = 3):
    return {
        "initial_state": torch.tensor([[[0.0, 0.0, 0.5, 0.0]]]),
        "rainfall_scenarios": torch.zeros(1, horizon, 1, 1),
        "current_settings": torch.tensor([0.5]),
        "fallback_settings": torch.full((1, horizon, 1), 0.5),
        "previous_actuator_flow": torch.zeros(1, 1),
        "actuator_upstream": torch.tensor([0]),
        "actuator_downstream": torch.tensor([0]),
        "actuator_physics": torch.zeros(1, 1, 2),
        "static_node_features": torch.zeros(1, 2),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "iterations": 20,
        "learning_rate": 0.1,
    }


def test_robust_mpc_executes_only_when_predicted_tfv_beats_hold() -> None:
    responsive = ContinuousTFVFirstMPC(
        _ToyWorld(responsive=True),
        depth_index=0,
        flood_rate_index=2,
        priority_indices=None,
        dt_seconds=300,
    )
    good = responsive.optimize(**_mpc_inputs())
    assert good.candidate_valid is True
    assert float(good.settings[0, 0]) > 0.5

    flat = ContinuousTFVFirstMPC(
        _ToyWorld(responsive=False),
        depth_index=0,
        flood_rate_index=2,
        priority_indices=None,
        dt_seconds=300,
    )
    no_leverage = flat.optimize(**_mpc_inputs())
    assert no_leverage.candidate_valid is False


def test_priority_deterioration_is_soft_when_total_tfv_improves() -> None:
    horizon = 3
    mpc = ContinuousTFVFirstMPC(
        _PriorityTradeoffWorld(),
        depth_index=0,
        flood_rate_index=2,
        priority_indices=torch.tensor([1]),
        dt_seconds=300,
        tfv_near_opt_relative=0.0,
        tfv_near_opt_absolute_m3=0.0,
    )
    result = mpc.optimize(
        initial_state=torch.tensor(
            [[[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.1, 0.0]]]
        ),
        rainfall_scenarios=torch.zeros(1, horizon, 2, 1),
        current_settings=torch.tensor([0.5]),
        fallback_settings=torch.full((1, horizon, 1), 0.5),
        previous_actuator_flow=torch.zeros(1, 1),
        actuator_upstream=torch.tensor([0]),
        actuator_downstream=torch.tensor([1]),
        actuator_physics=torch.zeros(1, 1, 2),
        static_node_features=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        iterations=30,
        learning_rate=0.1,
    )
    assert result.candidate_valid is True
    assert float(result.settings[0, 0]) > 0.5
    assert result.priority_positive_flood_deterioration_m3 > 0.0


def _catalog(count: int) -> ActuatorCatalog:
    return ActuatorCatalog(
        tuple(
            Actuator(f"A{i}", "pump", "N1", "N2")
            for i in range(count)
        )
    )


def test_budgeted_d2_rotates_coverage_without_reducing_candidate_action_space() -> None:
    catalog = _catalog(10)
    rows = []
    for checkpoint in range(3):
        row = {
            "checkpoint_id": f"C{checkpoint}",
            "scientific_split": "development",
        }
        row.update({f"setting:A{i}": 0.5 for i in range(10)})
        rows.append(row)
    manifest = design_budgeted_independent_probes(
        pd.DataFrame(rows),
        catalog,
        actuators_per_checkpoint=4,
        seed=7,
    )
    summary = summarise_budgeted_probe_design(manifest)
    assert summary["covered_actuators"] == 10
    assert summary["online_action_space_reduced"] is False
    assert manifest.groupby("checkpoint_id")["actuator_id"].nunique().max() == 4
    for raw in manifest["candidate_settings_json"]:
        assert set(json.loads(raw)) == set(catalog.ids)


def _metadata(root: Path, name: str, tfv: float) -> Path:
    stats = root / f"{name}.stats.csv"
    stats.write_text(
        f"node_id,delta_flooding_volume_m3\nN1,{tfv}\n",
        encoding="utf-8",
    )
    meta = root / f"{name}.json"
    meta.write_text(json.dumps({"node_statistics_file": stats.name}), encoding="utf-8")
    return meta


def test_control_leverage_report_detects_exact_swmm_improvement(tmp_path: Path) -> None:
    center = _metadata(tmp_path, "center", 100.0)
    better = _metadata(tmp_path, "better", 80.0)
    worse = _metadata(tmp_path, "worse", 110.0)
    manifest = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "rainfall_group": "R1",
                "checkpoint_id": "C1",
                "actuator_id": "A1",
                "base_action_sha256": "base",
                "candidate_action_sha256": action,
            }
            for action in ("base", "better", "worse")
        ]
    )
    runs = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "rainfall_group": "R1",
                "checkpoint_id": "C1",
                "candidate_action_sha256": action,
                "metadata_path": str(path),
            }
            for action, path in (("base", center), ("better", better), ("worse", worse))
        ]
    )
    detail, report = build_control_leverage_report(
        d2_manifest=manifest,
        d2_run_summary=runs,
        meaningful_absolute_m3=1.0,
        meaningful_relative=0.01,
    )
    assert len(detail) == 1
    assert bool(detail.iloc[0]["meaningful_improvement"])
    assert np.isclose(float(detail.iloc[0]["best_improvement_m3"]), 20.0)
    assert report["interpretation"] == "PROMISING_CONTROL_LEVERAGE"
    assert report["hard_gate"] is False
