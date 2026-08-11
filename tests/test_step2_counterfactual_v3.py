from __future__ import annotations

import numpy as np
import torch

from rtc.models import DifferentiableHydraulicWorldModel, GraphMessageBlock, _inverse_degree
from rtc.step2_counterfactual import (
    CounterfactualLossWeights,
    counterfactual_action_loss,
    counterfactual_groups,
    reference_index,
    rotated_reference_pairs,
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
    import pandas as pd

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
