from __future__ import annotations

import numpy as np
import pytest
import torch

from rtc.step2_mpc_v110 import RuntimeNormalizationV110, _integrated_positive_flood_m3


def test_runtime_normalization_round_trip_payload_and_tensor_scaling() -> None:
    norm = RuntimeNormalizationV110(
        state_mean=np.asarray([1.0, 2.0], dtype=np.float32),
        state_std=np.asarray([2.0, 4.0], dtype=np.float32),
        rainfall_mean=np.asarray([10.0], dtype=np.float32),
        rainfall_std=np.asarray([5.0], dtype=np.float32),
        flow_mean=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
        flow_std=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    loaded = RuntimeNormalizationV110.from_payload(norm.as_payload())
    state = torch.tensor([[[3.0, 6.0]]])
    rain = torch.tensor([[[[15.0]]]])
    flow = torch.tensor([[3.0, 6.0, 9.0]])
    assert torch.allclose(loaded.state(state), torch.tensor([[[1.0, 1.0]]]))
    assert torch.allclose(loaded.rainfall(rain), torch.ones_like(rain))
    assert torch.allclose(loaded.flow(flow), torch.ones_like(flow))


def test_runtime_normalization_rejects_zero_std() -> None:
    with pytest.raises(ValueError, match="std must be positive"):
        RuntimeNormalizationV110(
            state_mean=np.zeros(2, dtype=np.float32),
            state_std=np.asarray([1.0, 0.0], dtype=np.float32),
            rainfall_mean=np.zeros(1, dtype=np.float32),
            rainfall_std=np.ones(1, dtype=np.float32),
            flow_mean=np.zeros(1, dtype=np.float32),
            flow_std=np.ones(1, dtype=np.float32),
        ).validate()


def test_positive_flood_deterioration_uses_irregular_physical_time() -> None:
    # At t=0 the counterfactual deterioration is structurally zero. A 1 m3/s
    # deterioration at 5 and 10 min integrates to 150 + 300 = 450 m3.
    delta = torch.tensor([[[[1.0], [1.0]]]])
    result = _integrated_positive_flood_m3(delta, torch.tensor([5.0, 10.0]))
    assert result.shape == (1, 1, 1)
    assert float(result.item()) == pytest.approx(450.0)


def test_positive_flood_deterioration_does_not_penalize_improvement() -> None:
    delta = torch.tensor([[[[-1.0], [-2.0]]]])
    result = _integrated_positive_flood_m3(delta, torch.tensor([5.0, 10.0]))
    assert torch.equal(result, torch.zeros_like(result))
