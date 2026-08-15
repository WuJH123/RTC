from __future__ import annotations

import numpy as np
import pytest

from rtc.step2_train_v127_control import (
    V127ControlTrainingDesign,
    _candidate_permutation,
    informative_pair_threshold_v127,
)


def test_control_training_design_covers_every_teacher_phase() -> None:
    design = V127ControlTrainingDesign()
    design.validate()
    assert design.hydraulic_epochs >= design.teacher_stride
    assert design.rollout_horizons == (12, 24)
    assert design.objective_candidate_chunk == 3


def test_control_training_design_rejects_incomplete_teacher_phase_budget() -> None:
    with pytest.raises(ValueError, match="cover every teacher phase"):
        V127ControlTrainingDesign(hydraulic_epochs=3, teacher_stride=4).validate()


def test_candidate_permutation_has_full_unique_coverage_and_changes_by_epoch() -> None:
    p1 = _candidate_permutation(24, group_name="D3::rain::event::cp", epoch=1, seed=42)
    p2 = _candidate_permutation(24, group_name="D3::rain::event::cp", epoch=2, seed=42)
    assert sorted(p1.tolist()) == list(range(1, 25))
    assert sorted(p2.tolist()) == list(range(1, 25))
    assert len(np.unique(p1)) == 24
    assert len(np.unique(p2)) == 24
    assert not np.array_equal(p1, p2)


def test_informative_pair_threshold_is_fixed_engineering_relative_rule() -> None:
    design = V127ControlTrainingDesign()
    assert informative_pair_threshold_v127(500.0, design) == pytest.approx(1.0)
    assert informative_pair_threshold_v127(250_000.0, design) == pytest.approx(250.0)
