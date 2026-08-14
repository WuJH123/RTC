from __future__ import annotations

import numpy as np

from rtc.step3_calibration_v125 import (
    anchor_override_audit_v125,
    calibrate_anchor_override_margin_v125,
    pfv_deterioration_audit_v125,
)


def test_anchor_override_margins_are_one_sided_upper_residuals() -> None:
    # TFV residuals [20,20,30,30] => q95(higher)=30.
    # PFV residuals [5,10,15,20] => q95(higher)=20.
    cal = calibrate_anchor_override_margin_v125(
        truth_tfv_advantage_m3=[-50, -10, 20, 40],
        predicted_tfv_advantage_m3=[-70, -30, -10, 10],
        truth_pfv_advantage_m3=[5, 10, 20, 30],
        predicted_pfv_advantage_m3=[0, 0, 5, 10],
        rainfall_groups=["r1", "r1", "r2", "r2"],
        row_ids=["a", "b", "c", "d"],
        quantile=0.95,
    )
    assert cal.tfv_margin_m3 == 30.0
    assert cal.pfv_error_margin_m3 == 20.0
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


def test_pfv_audit_reports_false_soft_safety() -> None:
    report = pfv_deterioration_audit_v125(
        truth_advantage_m3=[10.0, 80.0, -20.0],
        predicted_advantage_m3=[0.0, 20.0, -30.0],
        error_margin_m3=20.0,
        soft_margin_m3=50.0,
    )
    assert report["predicted_soft_safe_count"] == 3
    assert report["truth_soft_safe_count"] == 2
    assert report["false_soft_safe_count"] == 1
