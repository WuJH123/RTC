from __future__ import annotations

from rtc.direct_tfv_v12_lineage_audit import V12_REFRESH_RECOMMENDATION


def test_v12_refresh_recommendation_is_stable() -> None:
    assert V12_REFRESH_RECOMMENDATION == (
        "RECALIBRATE_CURRENT_MAIN_WITH_EXISTING_V12_CALIBRATION_GROUPS"
    )
