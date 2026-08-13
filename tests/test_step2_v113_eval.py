from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.step2_hydraulic_eval_v113 import _channel_metric, build_oracle_support_override_v113
from rtc.step2_hydraulic_objective_v113 import derive_v113_scales


def test_v113_channel_metric_accepts_torch_inputs_without_device_leak():
    result = _channel_metric(
        torch.tensor([1.0, -1.0]),
        torch.tensor([2.0, -2.0]),
        torch.tensor([True, False]),
    )
    assert result["count"] == 2
    assert result["active_count"] == 1
    assert result["active_sign_accuracy"] == 1.0


def test_v113_oracle_support_override_is_per_changed_actuator_and_storage_domain():
    batch = SimpleNamespace(
        true_reference_states=torch.zeros(1, 4, 2, 6),
        true_candidate_states=torch.zeros(1, 1, 4, 2, 6),
        candidate_settings=torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]]),
        reference_settings=torch.zeros(1, 4, 2),
    )
    batch.true_candidate_states[:, 0, :, 0, 0] = 2.0
    batch.true_candidate_states[:, 0, :, 1, 3] = 2.0
    scales = derive_v113_scales(
        torch.ones(2, 5), torch.ones(2), torch.ones(2, 5), torch.ones(2), device="cpu"
    )
    override = build_oracle_support_override_v113(batch, scales, torch.tensor([False, True]), torch.tensor([0, 1]))
    assert override.shape == (1, 1, 2, 2, 2, 5)
    assert bool(override[:, :, 0, 0, 0, 0].item())
    assert not bool(override[:, :, 1, 0, 0, 0].item())
    assert bool(override[:, :, 0, 0, 1, 2].item())
    assert not bool(override[:, :, 1, 0, 1, 2].item())
