from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from rtc.data_index import build_d2_run_index
from rtc.dataset_compile import compile_branches_to_npz
from rtc.dataset_compile import BranchTensors
from rtc.models import DifferentiableHydraulicWorldModel, GraphMessageBlock, _inverse_degree
from rtc.step2_counterfactual import (
    CounterfactualLossWeights,
    counterfactual_action_loss,
    counterfactual_groups,
    reference_index,
    rotated_reference_pairs,
    same_prefix_diagnostic,
)
from rtc.step2_shards import _group_preserving_chunks


class FakeDataset(dict):
    @property
    def files(self):
        return list(self.keys())


def test_counterfactual_group_and_reference_use_provenance_not_outcomes():
    ds = FakeDataset(
        initial_state=np.zeros((4, 2, 6), dtype=np.float32),
        event_id=np.asarray(["e", "e", "e", "e"]),
        rainfall_group=np.asarray(["r", "r", "r", "r"]),
        checkpoint_id=np.asarray(["c", "c", "c", "c"]),
        source_kind=np.asarray(["D2", "D2", "D2", "D2"]),
        action_or_sequence_sha256=np.asarray(["z", "base", "a", "b"]),
        base_action_sha256=np.asarray(["base", "base", "base", "base"]),
        data_role=np.asarray(["", "", "", ""]),
    )
    groups = counterfactual_groups(ds)
    indices = next(iter(groups.values()))
    assert reference_index(ds, indices) == 1
    pairs = rotated_reference_pairs(ds, indices, epoch=0, budget=2)
    assert len(pairs) == 2
    assert all(left == 1 and right != 1 for left, right in pairs)


def test_counterfactual_loss_penalizes_action_collapse():
    initial = torch.zeros(2, 2, 6)
    target_state = torch.zeros(2, 1, 2, 6)
    target_state[0, :, :, 2] = 1.0
    target_state[1, :, :, 2] = 2.0
    target_flow = torch.zeros(2, 1, 1)
    good_state = target_state.clone()
    collapsed_state = target_state.clone()
    collapsed_state[1] = collapsed_state[0]
    good_flow = target_flow.clone()

    kwargs = dict(
        initial_state=initial,
        rollout_flows=good_flow,
        target_states=target_state,
        target_flows=target_flow,
        exact_node_flood_volume_m3=None,
        dt_seconds=torch.full((2, 1), 300.0),
        state_std=torch.ones(6),
        flow_std=torch.ones(1),
        full_horizon=False,
        weights=CounterfactualLossWeights(),
        flow_only=False,
    )
    good = counterfactual_action_loss(rollout_states=good_state, **kwargs)
    collapsed = counterfactual_action_loss(rollout_states=collapsed_state, **kwargs)
    assert float(good.delta_tfv) < float(collapsed.delta_tfv)
    assert float(good.ranking) < float(collapsed.ranking)
    assert float(good.sensitivity_ratio) > float(collapsed.sensitivity_ratio)


def test_counterfactual_loss_vectorizes_multiple_pairs():
    initial = torch.zeros(4, 2, 6)
    target_state = torch.zeros(4, 2, 2, 6)
    target_state[1, :, 0, 2] = 1.0
    target_state[3, :, 1, 2] = 2.0
    target_flow = torch.zeros(4, 2, 2)
    target_flow[1, :, 0] = 0.5
    target_flow[3, :, 1] = -0.25
    predicted = target_state.clone()
    flow_pred = target_flow.clone()
    same_prefix_diagnostic(
        initial,
        torch.zeros(4, 2, 2, 1),
        torch.zeros(4, 2),
    )
    metrics = counterfactual_action_loss(
        initial_state=initial,
        rollout_states=predicted,
        rollout_flows=flow_pred,
        target_states=target_state,
        target_flows=target_flow,
        exact_node_flood_volume_m3=None,
        dt_seconds=torch.full((4, 2), 300.0),
        state_std=torch.ones(6),
        flow_std=torch.ones(1),
        full_horizon=False,
        weights=CounterfactualLossWeights(),
    )
    assert metrics.true_delta_tfv_m3.shape == (2,)
    assert metrics.predicted_delta_tfv_m3.shape == (2,)
    assert torch.isfinite(metrics.total)
    assert float(metrics.sign_correct) == 1.0


def test_direct_setting_context_keeps_setting_gradient():
    setting = torch.tensor([[0.2, 0.8]], requires_grad=True)
    up = torch.tensor([0, 1])
    down = torch.tensor([1, 2])
    context = DifferentiableHydraulicWorldModel._setting_context(
        setting, up, down, node_count=3, dtype=torch.float32
    )
    assert context.shape == (1, 3, 2)
    context.sum().backward()
    assert setting.grad is not None
    assert torch.all(setting.grad != 0)


def test_cached_inverse_degree_matches_uncached_message_passing():
    torch.manual_seed(0)
    block = GraphMessageBlock(4)
    x = torch.randn(2, 3, 4)
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    inverse = _inverse_degree(edge, 3, dtype=x.dtype)
    uncached = block(x, edge)
    cached = block(x, edge, inverse)
    torch.testing.assert_close(cached, uncached)


def test_group_preserving_shards_never_split_checkpoint_group():
    frame = pd.DataFrame(
        {
            "source_kind": ["D2"] * 9,
            "rainfall_group": ["r"] * 9,
            "event_id": ["e"] * 9,
            "checkpoint_id": ["c0"] * 3 + ["c1"] * 3 + ["c2"] * 3,
            "candidate_action_sha256": [f"a{i}" for i in range(9)],
            "metadata_path": [f"p{i}" for i in range(9)],
        }
    )
    ordered, chunks, keys = _group_preserving_chunks(frame, shard_size=5)
    assert keys
    assert len(ordered) == 9
    for checkpoint in frame["checkpoint_id"].unique():
        containing = [
            chunk for chunk in chunks if checkpoint in set(chunk["checkpoint_id"])
        ]
        assert len(containing) == 1


def test_d2_index_preserves_base_action_sha_for_pair_reference():
    manifest = pd.DataFrame(
        {
            "candidate_action_sha256": ["base", "cand"],
            "base_action_sha256": ["base", "base"],
            "checkpoint_id": ["c", "c"],
            "event_id": ["e", "e"],
            "rainfall_group": ["r", "r"],
            "scientific_split": ["development", "development"],
            "development_fold": ["train", "train"],
        }
    )
    runs = pd.DataFrame(
        {
            "candidate_action_sha256": ["base", "cand"],
            "checkpoint_id": ["c", "c"],
            "event_id": ["e", "e"],
            "metadata_path": ["a.json", "b.json"],
        }
    )
    result = build_d2_run_index(manifest, runs)
    assert result["base_action_sha256"].tolist() == ["base", "base"]


def test_step2_shard_provenance_is_unicode_and_pickle_free(tmp_path, monkeypatch):
    branch = BranchTensors(
        initial_state=np.zeros((2, 6), dtype=np.float32),
        rainfall=np.zeros((1, 2, 1), dtype=np.float32),
        settings=np.zeros((1, 1), dtype=np.float32),
        previous_actuator_flow=np.zeros((1,), dtype=np.float32),
        target_states=np.zeros((1, 2, 6), dtype=np.float32),
        target_actuator_flows=np.zeros((1, 1), dtype=np.float32),
        elapsed_seconds=np.asarray([0, 300], dtype=np.int64),
        node_ids=("N1", "N2"),
        actuator_ids=("A1",),
        action_or_sequence_sha256="action",
        swmm_engine_version="5.2.4",
        exact_node_flood_volume_m3=np.zeros((2,), dtype=np.float32),
    )
    monkeypatch.setattr("rtc.dataset_compile.compile_branch_tensors", lambda _: branch)
    provenance = pd.DataFrame(
        {
            "metadata_path": ["branch.json"],
            "event_id": ["event"],
            "rainfall_group": ["rain"],
            "scientific_split": ["development"],
            "development_fold": ["train"],
            "data_role": ["counterfactual"],
            "checkpoint_id": ["cp0"],
            "base_action_sha256": ["base"],
            "source_kind": ["D2"],
        }
    )
    out = tmp_path / "step2.npz"
    compile_branches_to_npz(["branch.json"], out, provenance=provenance)
    with np.load(out, allow_pickle=False) as ds:
        assert ds["event_id"].dtype.kind == "U"
        assert ds["source_kind"].dtype.kind == "U"
        assert counterfactual_groups(ds) == {"D2::rain::event::cp0": [0]}
