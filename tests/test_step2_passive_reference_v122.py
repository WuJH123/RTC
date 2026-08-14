from __future__ import annotations

import numpy as np
import pytest

from rtc.step2_passive_reference_v122 import (
    assert_passive_command_sequence_v122,
    passive_target_latch_v122,
)


def test_passive_latch_repeats_target_not_current() -> None:
    target = np.array([0.2, 0.8])
    current = np.array([0.9, 0.1])
    result = passive_target_latch_v122(target, 3)
    assert result.shape == (3, 2)
    assert np.array_equal(result, np.repeat(target[None], 3, axis=0))
    assert not np.array_equal(result[0], current)


def test_passive_latch_rejects_invalid_target() -> None:
    with pytest.raises(ValueError):
        passive_target_latch_v122(np.array([0.2, 1.1]), 2)


def test_hold_sequence_requires_target_latch() -> None:
    assert_passive_command_sequence_v122(
        np.repeat(np.array([[0.2, 0.8]]), 4, axis=0), np.array([0.2, 0.8])
    )
    with pytest.raises(ValueError):
        assert_passive_command_sequence_v122(
            np.repeat(np.array([[0.2, 0.8]]), 3, axis=0), np.array([0.2, 0.7])
        )
