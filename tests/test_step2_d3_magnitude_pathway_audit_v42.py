from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rtc.step2_d3_magnitude_pathway_audit_v42 import (
    action_descriptors_v42,
    causal_memory_trace_v42,
    magnitude_stratum_v42,
    rank_correlation_v42,
)


def test_action_descriptors_preserve_absolute_energy_and_duration():
    delta = np.zeros((4, 3), dtype=np.float32)
    delta[1, 0] = 0.5
    delta[1, 1] = -0.25
    delta[3, 1] = 0.75
    result = action_descriptors_v42(delta, step_minutes=5.0)
    assert result["action_l1"] == pytest.approx(1.5)
    assert result["action_l2"] == pytest.approx(math.sqrt(0.5**2 + 0.25**2 + 0.75**2))
    assert result["action_linf"] == pytest.approx(0.75)
    assert result["action_energy_l1"] == pytest.approx(1.5)
    assert result["squared_action_energy"] == pytest.approx(0.5**2 + 0.25**2 + 0.75**2)
    assert result["active_actuator_count_per_block"] == [0, 2, 0, 1]
    assert result["max_active_actuator_count"] == 2
    assert result["changed_control_blocks"] == 2
    assert result["cumulative_changed_actuator_count"] == 2
    assert result["action_duration_minutes"] == pytest.approx(15.0)


def test_causal_memory_trace_is_causal_and_finite():
    value = torch.zeros(1, 5, 1, 1)
    value[0, 0, 0, 0] = 1.0
    trace = causal_memory_trace_v42(value, rho=0.65)
    assert torch.isfinite(trace).all()
    assert trace[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert trace[0, 1, 0, 0].item() == pytest.approx(0.65)
    assert torch.allclose(trace[:, :1], causal_memory_trace_v42(value[:, :1], rho=0.65))
    future = value.clone()
    future[0, 4, 0, 0] = 100.0
    assert torch.allclose(trace[:, :4], causal_memory_trace_v42(future, rho=0.65)[:, :4])


def test_magnitude_stratum_uses_fixed_train_boundaries():
    assert magnitude_stratum_v42(1.0, q33=2.0, q67=5.0) == "small"
    assert magnitude_stratum_v42(2.0, q33=2.0, q67=5.0) == "medium"
    assert magnitude_stratum_v42(5.0, q33=2.0, q67=5.0) == "large"


def test_rank_correlation_returns_nan_for_constant_input():
    assert math.isnan(rank_correlation_v42(np.ones(4), np.arange(4)))
    assert rank_correlation_v42(np.arange(4), np.arange(4)) == pytest.approx(1.0)
