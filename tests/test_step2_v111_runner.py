from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_step2_v111 import _select_micro  # noqa: E402


class _Entry:
    def __init__(self, actuator):
        self.reference_index = 0
        self.indices = (0, 1)
        settings = np.zeros((2, 4, 109), dtype=np.float32)
        settings[1, :, actuator] = 1.0
        self.arrays = {"settings": settings}
        self.event_id = "e"


class _Cache:
    def __init__(self):
        self.entries = {"D2::a": _Entry(1), "D2::b": _Entry(60), "D2::c": _Entry(100)}

    def entry(self, name):
        return self.entries[name]


def test_v111_micro_selector_handles_cache_without_actuator_type_array():
    selected = _select_micro(_Cache(), list(_Cache().entries))
    assert selected
    assert len(selected) <= 12

