import numpy as np
import pytest

from rtc.step2_v113_audit import (
    atlas_binary_metrics_v113,
    deterministic_event_split_v113,
    signed_distribution_v113,
)


def test_nested_event_split_is_deterministic_and_disjoint():
    events = ["event_0", "event_1", "event_2", "event_3", "event_4"]
    first = deterministic_event_split_v113(events, devfit_count=3)
    second = deterministic_event_split_v113(events, devfit_count=3)
    assert first == second
    assert len(first["devfit_events"]) == 3
    assert len(first["devcheck_events"]) == 2
    assert not set(first["devfit_events"]) & set(first["devcheck_events"])
    assert first["selection_rule"] == "sha256(event_id) ascending, first devfit_count"


def test_signed_distribution_reports_negative_and_active_signed_effect():
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 3.0])
    active = np.asarray([True, True, False, True, True])
    result = signed_distribution_v113(values, active)
    assert result["signed_mean"] == pytest.approx(0.2)
    assert result["positive_active_fraction"] == pytest.approx(0.5)
    assert result["negative_active_fraction"] == pytest.approx(0.5)
    assert result["active_fraction"] == pytest.approx(0.8)
    assert result["p50_abs"] == pytest.approx(1.0)
    assert result["max_abs"] == pytest.approx(3.0)


def test_empty_signed_distribution_is_json_safe():
    result = signed_distribution_v113(np.empty(0))
    assert result["count"] == 0
    assert result["active_fraction"] is None


def test_atlas_binary_metrics_fail_closed_for_single_class_and_measure_mass_recall():
    support = np.asarray([0.05, 0.2, 0.8, 0.9])
    truth_abs = np.asarray([0.0, 0.5, 2.0, 1.0])
    active = np.asarray([False, True, True, True])
    result = atlas_binary_metrics_v113(support, truth_abs, active, threshold=0.5)
    assert result["support_threshold"] == pytest.approx(0.5)
    assert result["active_recall"] == pytest.approx(2.0 / 3.0)
    assert result["effect_mass_recall"] == pytest.approx(3.0 / 3.5)

    no_active = atlas_binary_metrics_v113(support, truth_abs, np.zeros(4, dtype=bool), threshold=0.5)
    assert np.isnan(no_active["auroc"])
    assert np.isnan(no_active["auprc"])
