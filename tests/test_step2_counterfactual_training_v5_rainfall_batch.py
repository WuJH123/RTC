from __future__ import annotations

import pytest
import torch

from rtc.step2_counterfactual_training_v5 import _direct_rainfall_batch


def test_direct_rainfall_batch_preserves_node_and_feature_axes() -> None:
    rainfall = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    pair = {"rainfall": rainfall}

    batched = _direct_rainfall_batch(pair, branches=2, device=torch.device("cpu"))

    assert batched.shape == (2, 6, 2)
    assert torch.equal(batched[0], rainfall)
    assert torch.equal(batched[1], rainfall)


def test_direct_rainfall_batch_rejects_rank_drift() -> None:
    pair = {"rainfall": torch.ones(6, dtype=torch.float32)}

    with pytest.raises(ValueError, match="rainfall must be "):
        _direct_rainfall_batch(pair, branches=2, device=torch.device("cpu"))


def test_direct_rainfall_batch_requires_positive_branch_count() -> None:
    pair = {"rainfall": torch.ones((6, 1), dtype=torch.float32)}

    with pytest.raises(ValueError, match="positive branch count"):
        _direct_rainfall_batch(pair, branches=0, device=torch.device("cpu"))
