from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rtc.direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    stress_adaptive_hydraulic_target_v23,
)


def _graph() -> SimpleNamespace:
    actuator_ids = tuple(f"A{i:03d}" for i in range(109))
    names = (
        "min_setting",
        "max_setting",
        "is_pump",
        "is_orifice",
        "is_weir",
        "is_outlet",
    )
    physics = np.zeros((109, len(names)), dtype=np.float32)
    physics[:, 1] = 1.0
    for i in range(109):
        physics[i, 2 + (i % 4)] = 1.0
    return SimpleNamespace(
        actuator_ids=actuator_ids,
        actuator_physics_feature_names=names,
        actuator_physics=physics,
        actuator_upstream=np.zeros(109, dtype=np.int64),
        actuator_downstream=np.ones(109, dtype=np.int64),
        node_ids=("UP", "DOWN"),
        static_node_feature_names=("max_depth_m", "storage_capacity_m3"),
        static_node_features=np.asarray(
            [[2.0, 1000.0], [2.0, 1000.0]], dtype=np.float32
        ),
    )


def _target(state: torch.Tensor, *, mask: np.ndarray | None = None):
    return stress_adaptive_hydraulic_target_v23(
        current_state=state,
        rainfall_scenarios=torch.ones((3, 72, 2, 1), dtype=torch.float32),
        active_target=torch.full((109,), 0.5, dtype=torch.float32),
        graph=_graph(),
        first_radius=np.full(109, 0.15, dtype=np.float32),
        max_changed_facilities=8,
        max_delta_per_update=0.5,
        supervisory_mask=mask,
    )


def test_v23_low_stress_preserves_legacy_side_of_smooth_gate() -> None:
    state = torch.tensor(
        [[[0.5, 0.0, 0.0, 200.0], [0.4, 0.0, 0.0, 180.0]]],
        dtype=torch.float32,
    )
    target, diagnostics = _target(state)
    assert diagnostics.contract == V23_HYDRAULIC_CANDIDATE_CONTRACT
    assert diagnostics.strong_storm_blend == 0.0
    assert diagnostics.network_stress_q75 < 0.65
    assert int(torch.count_nonzero(torch.abs(target - 0.5) > 1.0e-7)) <= 8
    assert float(torch.max(torch.abs(target - 0.5))) <= 0.150001


def test_v23_high_stress_activates_absolute_fill_release_target() -> None:
    state = torch.tensor(
        [[[1.9, 0.0, 0.0, 950.0], [1.4, 0.0, 0.0, 700.0]]],
        dtype=torch.float32,
    )
    target, diagnostics = _target(state)
    assert diagnostics.strong_storm_blend > 0.80
    assert diagnostics.network_stress_q75 > 0.85
    assert diagnostics.maximum_release_fraction > 0.0
    assert diagnostics.mean_abs_strong_delta > 0.0
    assert int(torch.count_nonzero(torch.abs(target - 0.5) > 1.0e-7)) <= 8
    assert float(torch.max(torch.abs(target - 0.5))) <= 0.150001
    assert bool(torch.all((target >= 0.0) & (target <= 1.0)))


def test_v23_never_changes_passive_channels() -> None:
    state = torch.tensor(
        [[[1.9, 0.0, 0.0, 950.0], [0.5, 0.0, 0.0, 250.0]]],
        dtype=torch.float32,
    )
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    target, _diagnostics = _target(state, mask=mask)
    active = torch.full((109,), 0.5, dtype=torch.float32)
    assert torch.equal(target[82:], active[82:])
    assert int(torch.count_nonzero(torch.abs(target[:82] - active[:82]) > 1.0e-7)) <= 8


def test_v23_runtime_and_benchmark_are_explicitly_development_only() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "src" / "rtc" / "direct_tfv_operational_v23_runtime.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "scripts" / "run_policy_direct_tfv_operational_v23_development.py").read_text(
        encoding="utf-8"
    )
    benchmark = (root / "scripts" / "run_project7_operational_benchmark5_v23_development.py").read_text(
        encoding="utf-8"
    )
    assert "V23_PORTFOLIO_CONTRACT" in runtime
    assert '"candidate_generator_matches_v21_training": False' in runtime
    assert '"boundary_is_distribution_matched_to_v23_candidates": False' in runtime
    assert '"ready_for_policy_lock": False' in runtime
    assert "direct_tfv_operational_v23_runtime" in runner
    assert "run_policy_direct_tfv_operational_v23_development.py" in benchmark
