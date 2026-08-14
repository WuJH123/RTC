from __future__ import annotations

import numpy as np

from rtc.step3_calibration_v125 import (
    anchor_override_audit_v125,
    calibrate_anchor_override_margin_v125,
)


def test_anchor_override_margin_is_one_sided_upper_residual() -> None:
    # truth advantage = [-50,-10,+20,+40], prediction = [-70,-30,-10,+10]
    # residual = [20,20,30,30], so q95(higher)=30.
    cal = calibrate_anchor_override_margin_v125(
        truth_candidate_tfv_m3=[50, 90, 120, 140],
        truth_anchor_tfv_m3=[100, 100, 100, 100],
        predicted_candidate_delta_tfv_m3=[-70, -30, -10, 10],
        predicted_anchor_delta_tfv_m3=[0, 0, 0, 0],
        rainfall_groups=["r1", "r1", "r2", "r2"],
        row_ids=["a", "b", "c", "d"],
        quantile=0.95,
    )
    assert cal.margin_m3 == 30.0
    assert cal.sample_count == 4


def test_anchor_override_audit_counts_false_benefit() -> None:
    report = anchor_override_audit_v125(
        truth_advantage_m3=[-50.0, 20.0, -10.0],
        predicted_advantage_m3=[-80.0, -60.0, -5.0],
        margin_m3=30.0,
    )
    # First and second are admitted; second is a false benefit.
    assert report["admitted_count"] == 2
    assert report["false_benefit_count"] == 1
    assert np.isclose(report["beneficial_override_precision"], 0.5)
