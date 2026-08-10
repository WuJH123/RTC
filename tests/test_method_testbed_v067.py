from __future__ import annotations

import math

from rtc.method_testbed_v067 import (
    CHICAGO_R,
    _modify_curves_to_variable_depth,
    _modify_orifices,
    _repair_known_noop_rules,
    chicago_5min_hyetograph,
    wuhan_idf_average_mm_min,
)


def test_chicago_blocks_integrate_to_wuhan_idf_depth() -> None:
    for rp in (5, 10, 20, 50, 100):
        for duration in (60, 120, 180, 240, 300, 360):
            values = chicago_5min_hyetograph(rp, duration)
            depth = sum(intensity * 5.0 / 60.0 for _, intensity in values)
            expected = wuhan_idf_average_mm_min(rp, duration) * duration
            assert math.isclose(depth, expected, rel_tol=1e-10, abs_tol=1e-10)
            assert len(values) == duration // 5
            assert max(value for _, value in values) > 0
            assert CHICAGO_R == 0.39


def test_network_semantic_repairs_are_narrow() -> None:
    lines = [
        "[PUMPS]",
        "P1 N1 N2 PC1 OFF 0 0",
        "",
        "[ORIFICES]",
        "RTC_IN_01 N1 ST SIDE 0 0.65 NO 0",
        "O2 N1 N2 BOTTOM 0 0.65 NO 0",
        "",
        "[CURVES]",
        "PC1 PUMP2 0 0",
        "PC1 1 2",
        "",
        "[CONTROLS]",
        "RULE VP0600010.4_ON",
        "IF NODE N1 HEAD >= 1",
        "THEN ORIFICE VP0600010.4 SETTING = 1",
        "RULE VP0600010.4_Off",
        "IF NODE N1 HEAD < 1",
        "THEN ORIFICE VP0600010.4 SETTING = 1",
        "",
        "[END]",
    ]
    assert _modify_curves_to_variable_depth(lines, {"PC1"}) == 1
    assert any("PC1    PUMP4" in row for row in lines)
    changed, flaps = _modify_orifices(lines, travel_minutes=10)
    assert changed == 2
    assert flaps == 1
    rtc = next(row for row in lines if row.strip().startswith("RTC_IN_01"))
    ordinary = next(row for row in lines if row.strip().startswith("O2"))
    assert " YES " in f" {rtc} "
    assert " NO " in f" {ordinary} "
    assert rtc.split()[-1] == "0.166667"
    assert ordinary.split()[-1] == "0.166667"
    assert _repair_known_noop_rules(lines) == 1
    assert "THEN ORIFICE VP0600010.4 SETTING = 0" in lines
