from __future__ import annotations

import torch
import numpy as np

from rtc.controller_v122 import planned_sequence_max_step_delta_v122
from rtc.step2_policy_v120 import _project_executable_sequences_v120
from rtc.step3_mpc_v122 import (
    RollingMPCDesignV122,
    _select_first_move_group_v122,
)


def test_v122_tail_only_candidate_cannot_beat_hold() -> None:
    design = RollingMPCDesignV122(optimizer_steps=1)
    passive = torch.zeros((design.prediction_horizon_steps, 2))
    tail_only = passive.clone()
    tail_only[design.control_block_steps * 2 :] = 0.4
    real_first = passive.clone()
    real_first[: design.control_block_steps] = 0.2
    settings = torch.stack((passive, tail_only, real_first))
    scores = torch.tensor([0.0, -100.0, -5.0])
    selected, score, _, passive_count = _select_first_move_group_v122(
        settings, scores, passive, design=design
    )
    assert selected == 2
    assert score == -5.0
    assert passive_count == 2


def test_projected_candidates_are_the_scored_sequences() -> None:
    design = RollingMPCDesignV122(optimizer_steps=1)
    raw = torch.full((2, design.prediction_horizon_steps, 2), 0.9)
    projected = _project_executable_sequences_v120(
        raw,
        current_settings=torch.zeros(2),
        previous_requested_settings=torch.zeros(2),
        min_settings=torch.zeros(2),
        max_settings=torch.ones(2),
        max_delta_per_update=0.5,
        control_block_steps=2,
    )[0]
    assert torch.allclose(projected[:, :2], torch.full((2, 2, 2), 0.5))


def test_v122_planned_path_includes_active_target_anchor() -> None:
    value = planned_sequence_max_step_delta_v122(
        np.asarray([0.0, 0.5]),
        np.asarray([[0.5, 0.5], [0.5, 0.0]], dtype=float),
    )
    assert value == 0.5
