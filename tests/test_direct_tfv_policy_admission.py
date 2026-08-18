from __future__ import annotations

import pytest

from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
from rtc.direct_tfv_policy_admission import (
    DIRECT_TFV_POLICY_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_PANEL_CONTRACT,
    DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
    derive_policy_matched_admission,
)


def _base() -> dict:
    return {
        "contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
        "development_only": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "density_floor_changed_facilities": 20,
        "d3_rainfall_group_residual_conformal_upper_m3": 20_000.0,
        "d3_dense_rainfall_group_residual_conformal_upper_m3": 25_000.0,
        "optimizer_replay_residual_max_m3": 156_384.2207,
        "optimizer_replay_dense_residual_max_m3": 156_384.2207,
    }


def _records() -> list[dict]:
    rows = []
    for index in range(10):
        predicted = -100_000.0 - 1_000.0 * index
        residual = 5_000.0 + 1_000.0 * index
        rows.append(
            {
                "rainfall_group": f"fresh-{index}",
                "plan_sha256": f"{index:064x}",
                "predicted_delta_tfv_m3": predicted,
                "true_delta_tfv_m3": predicted + residual,
                "active_facility_count": 23,
            }
        )
    return rows


def test_policy_matched_margin_does_not_use_obsolete_pre_v6_max_as_floor() -> None:
    groups = [f"fresh-{index}" for index in range(10)]
    result = derive_policy_matched_admission(
        base_admission=_base(),
        panel_contract=DIRECT_TFV_POLICY_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
        panel_records=_records(),
        expected_rainfall_groups=groups,
        coverage=0.90,
    )
    assert result["contract"] == DIRECT_TFV_POLICY_ADMISSION_CONTRACT
    assert result["legacy_optimizer_replay_residual_max_m3"] == pytest.approx(156_384.2207)
    assert result["legacy_optimizer_replay_controls_current_margin"] is False
    assert result["policy_calibration_rainfall_group_count"] == 10
    assert result["global_margin_m3"] < 156_384.2207
    assert result["dense_margin_m3"] < 156_384.2207
    assert result["global_margin_m3"] >= 20_000.0
    assert result["dense_margin_m3"] >= 25_000.0


def test_policy_matched_admission_requires_every_fresh_rainfall_group() -> None:
    groups = [f"fresh-{index}" for index in range(10)]
    with pytest.raises(ValueError, match="cover every fresh rainfall group"):
        derive_policy_matched_admission(
            base_admission=_base(),
            panel_contract=DIRECT_TFV_POLICY_PANEL_CONTRACT,
            panel_step3_contract=DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
            panel_records=_records()[:-1],
            expected_rainfall_groups=groups,
            coverage=0.90,
        )


def test_policy_matched_admission_rejects_wrong_optimizer_contract() -> None:
    groups = [f"fresh-{index}" for index in range(10)]
    with pytest.raises(ValueError, match="current V6 raw optimizer"):
        derive_policy_matched_admission(
            base_admission=_base(),
            panel_contract=DIRECT_TFV_POLICY_PANEL_CONTRACT,
            panel_step3_contract="PROJECT7_OLD_OPTIMIZER",
            panel_records=_records(),
            expected_rainfall_groups=groups,
            coverage=0.90,
        )
