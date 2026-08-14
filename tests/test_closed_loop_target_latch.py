from __future__ import annotations

import numpy as np

from rtc.closed_loop import _reassert_target_latch


class _Link:
    def __init__(self, target: float) -> None:
        self.target_setting = target


class _Links(dict[str, _Link]):
    pass


def test_target_latch_reasserts_internal_pump_reset() -> None:
    links = _Links({"P1": _Link(1.0), "O1": _Link(0.25)})
    discrepancy = _reassert_target_latch(
        links, {"P1": 0.0, "O1": 0.25}
    )
    assert discrepancy == 1.0
    assert links["P1"].target_setting == 0.0
    assert links["O1"].target_setting == 0.25


def test_target_latch_reassertion_is_noop_without_held_command() -> None:
    links = _Links({"P1": _Link(float("nan"))})
    assert _reassert_target_latch(links, None) == 0.0
    assert np.isnan(links["P1"].target_setting)
