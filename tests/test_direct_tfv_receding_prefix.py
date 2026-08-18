from __future__ import annotations

import torch

from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_receding_prefix import (
    DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
    derive_receding_prefix_admission,
    executable_prefix_sequence,
)


def _base_policy() -> dict:
    return {
        "contract": DIRECT_TFV_POLICY_ADMISSION_CONTRACT,
        "development_only": True,
        "density_floor_changed_facilities": 20,
        "global_margin_m3": 102_726.0,
        "dense_margin_m3": 102_726.0,
        "d3_rainfall_group_residual_conformal_upper_m3": 25_000.0,
        "d3_dense_rainfall_group_residual_conformal_upper_m3": 17_000.0,
    }


def test_executable_prefix_keeps_only_first_control_block() -> None:
    target = torch.full((109,), 0.5)
    candidate = target[None].repeat(72, 1)
    candidate[:2, 0] = 0.8
    candidate[2:, 1] = 0.9
    prefix = executable_prefix_sequence(candidate, target, control_block_steps=2)
    assert torch.allclose(prefix[:2, 0], torch.tensor([0.8, 0.8]))
    assert torch.allclose(prefix[2:, 0], torch.full((70,), 0.5))
    assert torch.allclose(prefix[:, 1], torch.full((72,), 0.5))


def test_prefix_margin_is_not_forced_to_v2_full_plan_margin() -> None:
    records = []
    groups = [f"G{i:02d}" for i in range(10)]
    for i, group in enumerate(groups):
        predicted = -50_000.0 - 1000.0 * i
        # Residuals stay below the generic D3 25k floor.
        truth = predicted + 5_000.0 + 500.0 * i
        records.append(
            {
                "rainfall_group": group,
                "plan_sha256": f"sha{i:02d}",
                "predicted_prefix_delta_tfv_m3": predicted,
                "true_prefix_delta_tfv_m3": truth,
                "active_facility_count": 23,
            }
        )
    payload = derive_receding_prefix_admission(
        base_policy_admission=_base_policy(),
        panel_contract=DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=groups,
        coverage=0.90,
    )
    assert payload["contract"] == DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CONTRACT
    assert payload["execution_estimand"] == DIRECT_TFV_RECEDING_PREFIX_SEMANTICS
    assert payload["global_margin_m3"] == 25_000.0
    assert payload["dense_margin_m3"] == 25_000.0
    assert payload["global_margin_m3"] < payload["v2_full_plan_global_margin_m3"]
    assert payload["full_plan_policy_margin_controls_execution"] is False


def test_prefix_calibration_requires_all_independent_rainfall_groups() -> None:
    groups = [f"G{i:02d}" for i in range(10)]
    records = [
        {
            "rainfall_group": group,
            "plan_sha256": f"sha{i:02d}",
            "predicted_prefix_delta_tfv_m3": -10_000.0,
            "true_prefix_delta_tfv_m3": -9_000.0,
            "active_facility_count": 23,
        }
        for i, group in enumerate(groups[:-1])
    ]
    try:
        derive_receding_prefix_admission(
            base_policy_admission=_base_policy(),
            panel_contract=DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
            panel_step3_contract=DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
            panel_records=records,
            expected_rainfall_groups=groups,
            coverage=0.90,
        )
    except ValueError as exc:
        assert "cover every fresh rainfall group" in str(exc)
    else:
        raise AssertionError("receding-prefix calibration accepted incomplete rainfall-group coverage")
