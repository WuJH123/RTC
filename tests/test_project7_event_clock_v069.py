from __future__ import annotations

from pathlib import Path

import pytest

from rtc.event_clock import inspect_prepared_event_clock


def test_prepared_event_clock_measures_120_min_to_positive_rain(tmp_path: Path) -> None:
    inp = tmp_path / "event.inp"
    inp.write_text(
        """[OPTIONS]
FLOW_UNITS           CMS
START_DATE           01/01/2020
START_TIME           00:00:00
REPORT_START_DATE    01/01/2020
REPORT_START_TIME    00:00:00
END_DATE             01/01/2020
END_TIME             08:00:00

[RAINGAGES]
RG1 INTENSITY 0:05 1.0 TIMESERIES TS1

[TIMESERIES]
TS1 01/01/2020 01:55:00 0
TS1 01/01/2020 02:00:00 3.0
TS1 01/01/2020 02:05:00 0

[END]
""",
        encoding="utf-8",
    )
    evidence = inspect_prepared_event_clock(inp)
    assert evidence["effective_warmup_minutes"] == pytest.approx(120.0)
    assert evidence["rainfall_interval_minutes"] == pytest.approx(5.0)
