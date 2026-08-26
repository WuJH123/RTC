from __future__ import annotations

from pathlib import Path

import numpy as np

from rtc.direct_tfv_v25_value_calibration import FittedV25ValueCalibrator
from rtc.direct_tfv_v25r2_value_calibration import (
    V25R2_ESTIMAND,
    V25R2_TRUTH_FIELD,
    V25R2_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT,
    fitted_payload_v25r2,
)


def test_v25r2_payload_uses_exact_policy_return_not_h120() -> None:
    fitted = FittedV25ValueCalibrator(
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        weight=np.zeros(3),
        intercept=0.0,
        one_sided_error_margin_m3=1.0,
        ridge=1.0,
        error_quantile=0.9,
    )
    payload = fitted_payload_v25r2(
        fitted,
        lineage={"truth_field": V25R2_TRUTH_FIELD, "estimand": V25R2_ESTIMAND},
        oof_metrics={"train_oof_supported": False},
    )
    assert payload["contract"] == V25R2_VALUE_CALIBRATOR_CHECKPOINT_CONTRACT
    assert payload["truth_field"] == "true_policy_return_delta_tfv_m3"
    assert "H120" not in payload["estimand"]
    assert payload["development_only"] is True
    assert payload["formal_evidence"] is False


def test_v25r2_trainer_does_not_supervise_on_h120() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "train_project7_v25r2_exact_return_calibrator_current.py").read_text(encoding="utf-8")
    assert 'V25R2_TRUTH_FIELD' in source
    assert 'truth.append(exact_delta)' in source
    assert 'additional_swmm_runs": 0' in source
    assert 'true_policy_return_delta_tfv_h120_m3' not in source


def test_v25r2_runtime_keeps_h120_diagnostic_only() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "src" / "rtc" / "direct_tfv_operational_v25r2_runtime.py").read_text(encoding="utf-8")
    assert 'exact_policy_return_target_used_for_admission' in runtime
    assert 'h120_value_is_diagnostic_only' in runtime
    assert 'v24_hydraulic_stress_escape_enabled' in runtime
    assert 'V25R2 exact-return calibrator does not pass the Development OOF gate' in runtime
