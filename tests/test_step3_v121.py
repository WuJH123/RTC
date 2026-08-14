from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step3_policy_v121 import (
    FIRST_MOVE_GROUP_ATOL,
    FirstMoveRobustCandidatePolicyV121,
    _first_move_groups,
)


class _IdentityNormalization:
    def state(self, value):
        return value

    def rainfall(self, value):
        return value

    def flow(self, value):
        return value


class _FakeModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, state, rainfall, reference, candidate, flow, prepared):
        scenarios = state.shape[0]
        return SimpleNamespace(delta_tfv_m3=self.values[None].expand(scenarios, -1))


class _FakeBasis:
    def __init__(self, candidates: np.ndarray):
        self._candidates = torch.as_tensor(candidates, dtype=torch.float32)
        self.horizon = SimpleNamespace(horizon_steps=candidates.shape[1], control_block_steps=2)
        self.grouping = SimpleNamespace(actuator_count=candidates.shape[2])
        self.contract = SimpleNamespace(max_setting_delta_per_update=1.0)
        self.min_setting = np.zeros(candidates.shape[2], dtype=np.float32)
        self.max_setting = np.ones(candidates.shape[2], dtype=np.float32)

    def decode(self, reference, coefficients):
        return self._candidates[None].to(reference)


def _policy(candidates: np.ndarray, values: list[float]) -> FirstMoveRobustCandidatePolicyV121:
    policy = object.__new__(FirstMoveRobustCandidatePolicyV121)
    policy.base = SimpleNamespace(
        _coefficients=np.zeros((len(candidates), 1, 1), dtype=np.float32),
        min_predicted_improvement_m3=0.0,
    )
    policy.model = _FakeModel(values)
    policy.basis = _FakeBasis(candidates)
    policy.prepared = None
    policy.normalization = _IdentityNormalization()
    policy.contract = SimpleNamespace()
    policy.first_move_group_atol = FIRST_MOVE_GROUP_ATOL
    policy._families = tuple(["hold"] + [f"candidate_{i}" for i in range(1, len(candidates))])
    policy.last_result = None
    return policy


def _run(policy: FirstMoveRobustCandidatePolicyV121):
    horizon = policy.basis.horizon.horizon_steps
    return policy.optimize(
        initial_state=torch.zeros((1, 1), dtype=torch.float32),
        rainfall_scenarios=torch.zeros((1, horizon, 1, 1), dtype=torch.float32),
        fallback_settings=torch.zeros((1, horizon, 1), dtype=torch.float32),
        current_settings=torch.zeros(1, dtype=torch.float32),
        previous_requested_settings=torch.zeros(1, dtype=torch.float32),
        previous_actuator_flow=torch.zeros(1, dtype=torch.float32),
        max_delta_per_update=1.0,
    )


def test_first_move_groups_coalesce_equal_executable_actions():
    groups = _first_move_groups(
        np.asarray([[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.2, 0.0]], dtype=float)
    )
    assert groups == ((0, 1), (2, 3))


def test_tail_only_improvement_cannot_beat_hold():
    # Candidate 1 has an extremely attractive H360-style value but its executable
    # first 10-minute move is exactly HOLD. Candidate 2 actually changes the first move.
    candidates = np.asarray(
        [
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.5], [0.5], [0.5], [0.5]],
        ],
        dtype=np.float32,
    )
    result = _run(_policy(candidates, [0.0, -100.0, -50.0]))
    assert result.candidate_valid is True
    assert result.selected_candidate_index == 2
    assert result.hold_group_size == 2
    assert result.tail_only_noop_candidates == 1
    assert result.first_move_mean_abs_delta == 0.5
    assert result.robust_group_delta_tfv_m3 == -50.0


def test_only_tail_improvement_returns_exact_hold():
    candidates = np.asarray(
        [
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0], [1.0]],
        ],
        dtype=np.float32,
    )
    result = _run(_policy(candidates, [0.0, -100.0]))
    assert result.candidate_valid is False
    assert result.selected_candidate_index == 0
    assert result.reference_is_best is True
    assert result.tfv_risk_m3 == 0.0
