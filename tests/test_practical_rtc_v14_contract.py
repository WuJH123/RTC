from __future__ import annotations

from pathlib import Path

from rtc.actuator_release_semantics import release_fraction_to_setting
from rtc.baselines import (
    BASELINES,
    COMPETITIVE_BASELINE_IDS,
    DIAGNOSTIC_FIXED_BASELINE_IDS,
    FORMAL_FIXED_BASELINE_IDS,
)
from rtc.direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
from rtc.direct_tfv_policy_return_portfolio import (
    DEFAULT_LEARNED_SHRINK_SCALES,
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
)
from rtc.step3_tfv_value_mpc_v12 import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT


def test_six_strategy_panel_is_retained_but_extremes_are_not_competitive() -> None:
    assert FORMAL_FIXED_BASELINE_IDS == (
        "no_control",
        "internal_rtc",
        "auto_rbc",
        "efd",
        "all_open",
        "all_closed",
    )
    assert COMPETITIVE_BASELINE_IDS == (
        "no_control",
        "internal_rtc",
        "auto_rbc",
        "efd",
    )
    assert DIAGNOSTIC_FIXED_BASELINE_IDS == ("all_open", "all_closed", "hold")
    assert BASELINES["all_open"].formal_comparator is False
    assert BASELINES["all_closed"].formal_comparator is False
    assert BASELINES["auto_rbc"].formal_comparator is True
    assert BASELINES["efd"].formal_comparator is True


def test_practical_portfolio_has_only_half_and_full_learned_scales() -> None:
    assert DEFAULT_LEARNED_SHRINK_SCALES == (0.50, 1.00)
    assert "H10_PROBE" in DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT
    assert "H10_POLICY_RETURN_PORTFOLIO" in DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT
    assert DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING == "H10_CANDIDATE_THEN_H350_HOLD_ACTION_TOKEN_V1"


def test_practical_paper_contract_explicitly_removes_lbfgsb_from_online_path() -> None:
    text = Path("PROJECT7_PRACTICAL_RTC_V14.md").read_text(encoding="utf-8")
    assert "does **not** solve the historical 12 x 109 = 1308-dimensional" in text
    assert "H10 candidate target -> H350 current HOLD target" in text
    assert "Priority8 PFV" in text
    assert "READY_FOR_POLICY_LOCK=false" in text


def test_weir_release_intent_is_not_generic_setting_opening() -> None:
    assert release_fraction_to_setting("pump", 0.8) == 0.8
    assert release_fraction_to_setting("orifice", 0.8) == 0.8
    assert release_fraction_to_setting("outlet", 0.8) == 0.8
    assert abs(release_fraction_to_setting("weir", 0.8) - 0.2) < 1.0e-12
