from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rtc.code_contract import rtc_source_tree_sha256
from rtc.models import DifferentiableHydraulicWorldModel, HydraulicTransition
from rtc.step2_shards import sha256_file
from rtc.step2_stability import derive_train_only_delta_scales
from rtc.step2_train_v3 import _stage
from rtc.step2_training_cache import build_step2_training_cache, load_step2_training_cache


def _transition(*, bounded: bool = True) -> HydraulicTransition:
    model = HydraulicTransition(
        state_dim=2,
        rainfall_dim=1,
        static_dim=1,
        hidden_dim=4,
        graph_layers=1,
        bounded_state_residual=bounded,
        delta_state_scale=torch.tensor([0.25, 0.5]),
    )
    model.set_normalization(
        state_mean=torch.zeros(2),
        state_std=torch.ones(2),
        rain_mean=torch.zeros(1),
        rain_std=torch.ones(1),
        static_mean=torch.zeros(1),
        static_std=torch.ones(1),
        injection_std=torch.ones(1),
    )
    with torch.no_grad():
        model.residual[-1].weight.zero_()
        model.residual[-1].bias.fill_(100.0)
    return model


def test_bounded_state_residual_limits_one_step_increment() -> None:
    model = _transition()
    state = torch.zeros(1, 2, 2)
    rainfall = torch.zeros(1, 2, 1)
    static = torch.zeros(1, 2, 1)
    injection = torch.zeros(1, 2, 1)
    edge = torch.tensor([[0, 1], [1, 0]])
    inverse = torch.ones(2)
    result = model.forward_prepared(
        state,
        rainfall,
        static,
        injection,
        edge,
        inverse,
    )
    increment = (result - state).abs().amax(dim=(0, 1))
    assert torch.all(increment <= model.delta_state_scale + 1e-6)


def test_bounded_flow_residual_limits_increment_and_setting_gradient() -> None:
    model = DifferentiableHydraulicWorldModel(
        state_dim=2,
        rainfall_dim=1,
        node_static_dim=1,
        actuator_physics_dim=1,
        hidden_dim=4,
        actuator_count=1,
        direct_action_context=True,
        bounded_state_residual=True,
        bounded_flow_residual=True,
        delta_state_scale=torch.tensor([0.25, 0.5]),
        delta_flow_scale=torch.tensor([0.2]),
    )
    with torch.no_grad():
        model.actuator.flow_delta.bias.fill_(100.0)
    setting = torch.full((1, 3, 1), 0.4, requires_grad=True)
    rollout = model.rollout(
        torch.zeros(1, 2, 2),
        torch.zeros(1, 3, 2, 1),
        setting,
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.tensor([1]),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 2, 1),
        torch.tensor([[0, 1], [1, 0]]),
    )
    flow_delta = rollout.actuator_flows[:, 0] - torch.zeros(1, 1)
    flow_delta = torch.cat(
        (flow_delta[:, None], rollout.actuator_flows[:, 1:] - rollout.actuator_flows[:, :-1]),
        dim=1,
    )
    assert torch.all(flow_delta.abs() <= 0.2 + 1e-6)
    rollout.states.sum().backward()
    assert setting.grad is not None
    assert torch.isfinite(setting.grad).all()
    assert torch.any(setting.grad.abs() > 0)


def test_stability_scales_survive_save_load() -> None:
    kwargs = dict(
        state_dim=2,
        rainfall_dim=1,
        node_static_dim=1,
        actuator_physics_dim=1,
        hidden_dim=4,
        actuator_count=1,
        direct_action_context=True,
        bounded_state_residual=True,
        bounded_flow_residual=True,
        delta_state_scale=torch.tensor([0.25, 0.5]),
        delta_flow_scale=torch.tensor([0.2]),
    )
    source = DifferentiableHydraulicWorldModel(**kwargs)
    restored = DifferentiableHydraulicWorldModel(**kwargs)
    restored.load_state_dict(source.state_dict())
    torch.testing.assert_close(
        restored.transition.delta_state_scale,
        torch.tensor([0.25, 0.5]),
    )
    torch.testing.assert_close(restored.actuator.delta_flow_scale, torch.tensor([0.2]))


def test_seventy_two_step_bounded_rollout_is_finite() -> None:
    model = DifferentiableHydraulicWorldModel(
        state_dim=2,
        rainfall_dim=1,
        node_static_dim=1,
        actuator_physics_dim=1,
        hidden_dim=4,
        actuator_count=1,
        direct_action_context=True,
        bounded_state_residual=True,
        bounded_flow_residual=True,
        delta_state_scale=torch.tensor([0.25, 0.5]),
        delta_flow_scale=torch.tensor([0.2]),
    )
    rollout = model.rollout(
        torch.zeros(1, 2, 2),
        torch.zeros(1, 72, 2, 1),
        torch.full((1, 72, 1), 0.5),
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.tensor([1]),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 2, 1),
        torch.tensor([[0, 1], [1, 0]]),
    )
    assert torch.isfinite(rollout.states).all()
    assert torch.isfinite(rollout.actuator_flows).all()
    assert float(rollout.states.abs().max()) < 100.0
    flow_delta = torch.cat(
        (
            rollout.actuator_flows[:, :1],
            rollout.actuator_flows[:, 1:] - rollout.actuator_flows[:, :-1],
        ),
        dim=1,
    )
    assert float(flow_delta.abs().max()) <= 0.2 + 1e-6


def test_train_only_delta_scales_are_deterministic_and_split_bound(tmp_path: Path) -> None:
    shard = tmp_path / "train.npz"
    initial = np.zeros((1, 2, 6), dtype=np.float32)
    target_states = np.zeros((1, 3, 2, 6), dtype=np.float32)
    target_states[:, 0, :, 0] = 1.0
    target_states[:, 1, :, 0] = 3.0
    target_states[:, 2, :, 0] = 6.0
    target_flows = np.asarray([[[1.0, 2.0], [3.0, 6.0], [6.0, 10.0]]], dtype=np.float32)
    np.savez(
        shard,
        initial_state=initial,
        target_states=target_states,
        previous_actuator_flow=np.zeros((1, 2), dtype=np.float32),
        target_actuator_flows=target_flows,
        scientific_split=np.asarray(["development"]),
        development_fold=np.asarray(["train"]),
    )
    manifest = {
        "shards": [{"path": str(shard)}],
        "row_count": 1,
        "contract": "STEP2_SHARDED_DATASET_V6_COUNTERFACTUAL_GROUP_PRESERVING",
    }
    state_scale, flow_scale, details = derive_train_only_delta_scales(
        manifest,
        state_std=np.asarray([0.1] * 6, dtype=np.float32),
        flow_std=np.asarray([0.5], dtype=np.float32),
    )
    # State deltas are [1, 2, 3], so RMS=sqrt(14/3), then multiplied by six.
    np.testing.assert_allclose(state_scale[0], 6.0 * np.sqrt(14.0 / 3.0), rtol=1e-5)
    assert state_scale[1] >= 0.001
    assert flow_scale.shape == (2,)
    assert details["scientific_split"] == "development"
    assert details["development_fold"] == "train"


def test_stability_curriculum_has_explicit_h72_trajectory_and_exact_stages() -> None:
    rows = [_stage(epoch, 24, 72) for epoch in range(24)]
    assert [row[0] for row in rows] == [
        "flow_h1",
        "flow_h1",
        "joint_h1",
        "joint_h1",
        "h2",
        "h2",
        "h6",
        "h6",
        "h12",
        "h12",
        "h24",
        "h24",
        "h24",
        "h36",
        "h36",
        "h48",
        "h48",
        "h60",
        "h60",
        "h60",
        "h72_trajectory",
        "h72_trajectory",
        "h72_exact",
        "h72_exact",
    ]
    assert rows[20][3:] == (False, False)
    assert rows[22][2:] == (False, True, True)


def test_rebuildable_cache_is_mmap_and_numerically_equivalent(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    np.savez(
        source,
        initial_state=np.zeros((1, 2, 6), dtype=np.float32),
        rainfall=np.zeros((1, 1, 2, 1), dtype=np.float32),
        settings=np.zeros((1, 1, 1), dtype=np.float32),
        previous_actuator_flow=np.zeros((1, 1), dtype=np.float32),
        target_states=np.zeros((1, 1, 2, 6), dtype=np.float32),
        target_actuator_flows=np.zeros((1, 1, 1), dtype=np.float32),
        elapsed_seconds=np.asarray([[0.0, 300.0]], dtype=np.float32),
        exact_node_flood_volume_m3=np.zeros((1, 2), dtype=np.float32),
        node_ids=np.asarray(["N1", "N2"]),
        actuator_ids=np.asarray(["A1"]),
        action_or_sequence_sha256=np.asarray(["a"]),
        event_id=np.asarray(["e"]),
        rainfall_group=np.asarray(["r"]),
        scientific_split=np.asarray(["development"]),
        development_fold=np.asarray(["train"]),
        data_role=np.asarray(["D2_BASE"]),
        checkpoint_id=np.asarray(["c"]),
        base_action_sha256=np.asarray(["a"]),
        source_kind=np.asarray(["D2"]),
        model_step_seconds=np.asarray(300),
        horizon_steps=np.asarray(1),
        swmm_engine_version=np.asarray("5.2.4"),
    )
    manifest = tmp_path / "manifest.json"
    payload = {
        "contract": "STEP2_SHARDED_DATASET_V6_COUNTERFACTUAL_GROUP_PRESERVING",
        "row_count": 1,
        "shard_size": 1,
        "model_step_seconds": 300,
        "horizon_steps": 1,
        "swmm_engine_version": "5.2.4",
        "source_run_index_sha256": "index",
        "source_simulation_identity_set_sha256": "",
        "simulation_identity_bound": False,
        "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        "counterfactual_groups_preserved": True,
        "counterfactual_group_columns": ["source_kind", "rainfall_group", "event_id", "checkpoint_id"],
        "shards": [
            {
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
                "rows": 1,
                "start_row": 0,
                "model_step_seconds": 300,
                "horizon_steps": 1,
                "swmm_engine_version": "5.2.4",
                "simulation_identity_set_sha256": "",
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    cache_manifest = build_step2_training_cache(manifest, tmp_path / "cache")
    cached = load_step2_training_cache(cache_manifest)
    array = cached["shards"][0]["arrays"]["target_states"]
    assert isinstance(array, np.memmap)
    with np.load(source, allow_pickle=False) as raw:
        np.testing.assert_array_equal(array, raw["target_states"])
