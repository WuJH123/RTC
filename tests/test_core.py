from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.contracts import TimeScaleConfig
from rtc.data_design import design_independent_actuator_probes
from rtc.inp import discover_actuators, discover_nodes
from rtc.models import DifferentiableHydraulicWorldModel
from rtc.objectives import assess_priority_safety, total_flood_volume


SYNTHETIC_INP = """
[TITLE]
Synthetic RTC test
[JUNCTIONS]
P1 0 2 0 0 0
P2 0 2 0 0 0
N3 0 2 0 0 0
N4 0 2 0 0 0
[PUMPS]
PU1 P1 P2 CURVE1 ON
[ORIFICES]
OR1 P2 N3 SIDE 0 0.6 NO 0
[WEIRS]
WE1 N3 N4 TRANSVERSE 0 1.0 NO 0 0 NO
[OUTLETS]
OU1 N4 P1 0 TABULAR CURVE2 NO
"""


def _write_inp(tmp_path: Path) -> Path:
    path = tmp_path / "network.inp"
    path.write_text(SYNTHETIC_INP, encoding="utf-8")
    return path


def test_inp_discovers_all_continuous_actuators(tmp_path: Path) -> None:
    path = _write_inp(tmp_path)
    catalog = discover_actuators(path)
    assert catalog.ids == ("PU1", "OR1", "WE1", "OU1")
    assert {a.kind for a in catalog.actuators} == {"pump", "orifice", "weir", "outlet"}
    assert all(a.continuous for a in catalog.actuators)
    assert all((a.min_setting, a.max_setting) == (0.0, 1.0) for a in catalog.actuators)
    assert set(discover_nodes(path)) == {"P1", "P2", "N3", "N4"}


def test_probe_design_uses_same_checkpoint_and_one_actuator(tmp_path: Path) -> None:
    catalog = discover_actuators(_write_inp(tmp_path))
    frame = pd.DataFrame(
        [
            {
                "checkpoint_id": "eventA:t30",
                "rainfall_group": "A",
                **{f"setting:{aid}": 0.5 for aid in catalog.ids},
            }
        ]
    )
    manifest = design_independent_actuator_probes(frame, catalog, epsilon=0.2)
    assert manifest["actuator_id"].nunique() == 4
    assert manifest["checkpoint_id"].nunique() == 1
    assert manifest["same_checkpoint_required"].all()
    assert manifest["all_other_actuators_fixed"].all()
    assert len(manifest) == 12  # three settings per actuator
    assert set(np.round(manifest["requested_setting"], 6)) == {0.3, 0.5, 0.7}


def test_priority_safety_is_relative_to_fallback() -> None:
    # [scenario, time, node]
    fallback_flood = np.zeros((3, 2, 3), dtype=float)
    candidate_flood = fallback_flood.copy()
    candidate_flood[:, :, 0] = 0.01
    fallback_depth = np.ones((3, 2, 3), dtype=float)
    candidate_depth = fallback_depth.copy()
    candidate_depth[:, :, 0] += 0.02
    result = assess_priority_safety(
        candidate_flood_rate=candidate_flood,
        fallback_flood_rate=fallback_flood,
        candidate_depth=candidate_depth,
        fallback_depth=fallback_depth,
        priority_indices=np.array([0]),
        dt_seconds=600,
        priority_flood_budget_m3=20.0,
        priority_depth_budget_m=0.05,
    )
    assert result.admissible
    assert np.isclose(result.priority_flood_deterioration_ucb_m3, 12.0)
    assert np.isclose(total_flood_volume(candidate_flood, 600), 12.0).all()


def test_world_model_is_differentiable_wrt_continuous_settings() -> None:
    torch.manual_seed(7)
    b, h, n, a = 2, 3, 4, 2
    state_dim, rain_dim, static_dim, physics_dim = 3, 1, 2, 2
    model = DifferentiableHydraulicWorldModel(
        state_dim=state_dim,
        rainfall_dim=rain_dim,
        node_static_dim=static_dim,
        actuator_physics_dim=physics_dim,
        hidden_dim=16,
    )
    initial = torch.randn(b, n, state_dim)
    rainfall = torch.randn(b, h, n, rain_dim)
    settings = torch.full((b, h, a), 0.5, requires_grad=True)
    previous_flow = torch.zeros(b, a)
    upstream = torch.tensor([0, 2])
    downstream = torch.tensor([1, 3])
    physics = torch.randn(b, a, physics_dim)
    static = torch.randn(n, static_dim)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    rollout = model.rollout(
        initial,
        rainfall,
        settings,
        previous_flow,
        upstream,
        downstream,
        physics,
        static,
        edges,
    )
    assert rollout.states.shape == (b, h, n, state_dim)
    assert rollout.actuator_flows.shape == (b, h, a)
    loss = rollout.states.square().mean() + rollout.actuator_flows.square().mean()
    loss.backward()
    assert settings.grad is not None
    assert torch.isfinite(settings.grad).all()
    assert settings.grad.abs().sum() > 0


def test_time_scale_is_configurable_not_hardcoded() -> None:
    cfg = TimeScaleConfig(
        model_step_minutes=5,
        control_update_minutes=10,
        prediction_horizon_minutes=90,
        control_block_minutes=(10, 10, 20, 20, 30),
    )
    cfg.validate()
