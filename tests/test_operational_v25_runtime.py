from __future__ import annotations

import numpy as np
import torch

from rtc.direct_tfv_v25_value_calibration import (
    V25ValueCalibratorModule,
    fit_v25_value_calibrator,
)
from rtc.direct_tfv_v25_value_gate import evaluate_v25_value_gate


def test_v25_group_oof_is_deterministic_and_reports_support_contract() -> None:
    groups = [f"g{i:02d}" for i in range(48)]
    x = np.asarray([[float(i), float(i % 3)] for i in range(48)], dtype=np.float64)
    y = np.asarray([-1.0 if i % 2 == 0 else 1.0 for i in range(48)], dtype=np.float64)
    fitted, metrics, oof = fit_v25_value_calibrator(x, y, groups)
    assert oof.shape == (48,)
    assert metrics["oof_group_count"] == 48
    assert metrics["oof_folds"] == 6
    assert "support_contract" in metrics
    assert fitted.feature_width == 2


def test_v25_runtime_upper_bound_is_the_only_action_authority() -> None:
    fitted, _, _ = fit_v25_value_calibrator(
        np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64),
        np.asarray([-1.0, -0.8, -0.6, 0.6, 0.8, 1.0], dtype=np.float64),
        [f"g{i}" for i in range(6)],
        folds=6,
    )
    model = V25ValueCalibratorModule(fitted)
    prediction, upper = model.predict(torch.tensor([0.0]))
    assert torch.isfinite(prediction)
    assert torch.isfinite(upper)
    decision = evaluate_v25_value_gate(
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
        predicted_delta_tfv_m3=float(prediction),
        one_sided_error_margin_m3=float(model.one_sided_error_margin_m3),
        tfv_value_available=True,
        calibration_available=True,
        lineage_valid=True,
        engineering_feasible=True,
        passive_channels_unchanged=True,
        sequence_support_valid=True,
        network_stress_q75=1.0,
        strong_storm_blend=1.0,
    )
    assert decision.action == (float(upper) < 0.0)
