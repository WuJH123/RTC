from __future__ import annotations

from rtc.direct_tfv_operational_v23_runtime import OPERATIONAL_V23_RUNTIME_CONTRACT
from rtc.direct_tfv_operational_v24_runtime import OPERATIONAL_V24_RUNTIME_CONTRACT
from rtc.direct_tfv_v25_value_gate import (
    V25_VALUE_GATE_CONTRACT,
    evaluate_v25_value_gate,
    validate_v25_lineage,
)


def _decision(**overrides):
    values = {
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "predicted_delta_tfv_m3": -10.0,
        "one_sided_error_margin_m3": 2.0,
        "tfv_value_available": True,
        "calibration_available": True,
        "lineage_valid": True,
        "engineering_feasible": True,
        "passive_channels_unchanged": True,
        "sequence_support_valid": True,
        "network_stress_q75": 0.95,
        "strong_storm_blend": 1.0,
    }
    values.update(overrides)
    return evaluate_v25_value_gate(**values)


def test_high_stress_with_positive_ucb_holds():
    result = _decision(predicted_delta_tfv_m3=1.0, one_sided_error_margin_m3=2.0)
    assert result.action is False
    assert result.tfv_value_upper_bound_m3 == 3.0


def test_high_stress_with_zero_ucb_holds():
    result = _decision(predicted_delta_tfv_m3=-2.0, one_sided_error_margin_m3=2.0)
    assert result.action is False
    assert result.reason == "TFV_UCB_NONNEGATIVE"


def test_low_stress_with_negative_ucb_can_act():
    result = _decision(
        predicted_delta_tfv_m3=-10.0,
        one_sided_error_margin_m3=2.0,
        network_stress_q75=0.1,
        strong_storm_blend=0.0,
    )
    assert result.action is True
    assert result.reason == "TFV_UCB_NEGATIVE"


def test_hydraulic_candidate_does_not_override_value_admission():
    result = _decision(
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
        predicted_delta_tfv_m3=-1.0,
        one_sided_error_margin_m3=1.0,
    )
    assert result.action is False
    assert result.reason == "TFV_UCB_NONNEGATIVE"


def test_negative_point_prediction_with_nonnegative_ucb_holds():
    result = _decision(predicted_delta_tfv_m3=-0.5, one_sided_error_margin_m3=1.0)
    assert result.action is False


def test_negative_conservative_ucb_acts():
    result = _decision(predicted_delta_tfv_m3=-2.0, one_sided_error_margin_m3=0.5)
    assert result.action is True


def test_missing_or_wrong_lineage_fails_closed():
    expected = {
        "base_step2_sha256": "step2-sha",
        "v15_rank_sha256": "rank-sha",
        "candidate_generator_contract": "generator-v25",
        "supervisory_mask_sha256": "mask-sha",
        "truth_bank_sha256": "truth-sha",
    }
    actual = dict(expected)
    assert validate_v25_lineage(actual, expected) is True
    for key in expected:
        wrong = dict(actual)
        wrong[key] = "wrong"
        assert validate_v25_lineage(wrong, expected) is False
        result = _decision(lineage_valid=validate_v25_lineage(wrong, expected))
        assert result.action is False
        assert result.reason == "LINEAGE_INVALID"


def test_passive_channel_modification_fails_closed():
    result = _decision(passive_channels_unchanged=False)
    assert result.action is False
    assert result.reason == "PASSIVE_CHANNEL_VIOLATION"


def test_sequence_support_violation_fails_closed():
    result = _decision(sequence_support_valid=False)
    assert result.action is False
    assert result.reason == "SEQUENCE_SUPPORT_VIOLATION"


def test_historical_v23_v24_contracts_remain_unchanged():
    assert V25_VALUE_GATE_CONTRACT == "PROJECT7_STEP3_V25_CONSERVATIVE_TFV_VALUE_GATE_V1"
    assert (
        OPERATIONAL_V23_RUNTIME_CONTRACT
        == "PROJECT7_OPERATIONAL_DEVELOPMENT_V23_STRONG_STORM_HYDRAULIC_CANDIDATE_V1"
    )
    assert (
        OPERATIONAL_V24_RUNTIME_CONTRACT
        == "PROJECT7_OPERATIONAL_DEVELOPMENT_V24_RBC_INFORMED_STRESS_ESCAPE_V1"
    )
