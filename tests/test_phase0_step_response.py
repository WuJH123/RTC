from __future__ import annotations

import numpy as np

from rtc.phase0_timescale import _step_response_times


def test_step_response_reports_amplitude_timing_not_area_mass() -> None:
    t = np.arange(0.0, 601.0, 60.0)
    effect = np.array([0.0, 0.05, 0.2, 0.55, 0.92, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    peak, t10, t50, t90, peak_time, censored, endpoint_ratio = _step_response_times(t, effect)
    assert peak == 1.0
    assert t10 == 120.0
    assert t50 == 180.0
    assert t90 == 240.0
    assert peak_time == 300.0
    assert censored is False
    assert endpoint_ratio == 1.0


def test_peak_near_horizon_is_flagged_for_longer_pilot() -> None:
    t = np.arange(0.0, 601.0, 60.0)
    effect = np.linspace(0.0, 1.0, len(t))
    _, _, _, _, peak_time, censored, _ = _step_response_times(t, effect)
    assert peak_time == 600.0
    assert censored is True
