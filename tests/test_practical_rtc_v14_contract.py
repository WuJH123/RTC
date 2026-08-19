from __future__ import annotations

from rtc.actuator_release_semantics import release_fraction_to_setting
from rtc.baselines import (
    BASELINES,
    COMPETITIVE_BASELINE_IDS,
    DIAGNOSTIC_FIXED_BASELINE_IDS,
    FORMAL_FIXED_BASELINE_IDS,
)
from rtc.direct_tfv_policy_return_portfolio import DEFAULT_V12_SHRINK_SCALES


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
    assert DEFAULT_V12_SHRINK_SCALES == (0.50, 1.00)


def test_weir_release_intent_is_not_generic_setting_opening() -> None:
    assert release_fraction_to_setting("pump", 0.8) == 0.8
    assert release_fraction_to_setting("orifice", 0.8) == 0.8
    assert release_fraction_to_setting("outlet", 0.8) == 0.8
    assert abs(release_fraction_to_setting("weir", 0.8) - 0.2) < 1.0e-12
