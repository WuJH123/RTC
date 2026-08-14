from __future__ import annotations

import numpy as np
import pytest

from rtc.step2_passive_reference_v122 import assert_passive_command_sequence_v122


def test_passive_audit_rejects_command_that_follows_realised_current() -> None:
    target = np.array([0.2, 0.8])
    current = np.array([0.9, 0.1])
    with pytest.raises(ValueError):
        assert_passive_command_sequence_v122(
            np.repeat(current[None], 3, axis=0), target
        )
