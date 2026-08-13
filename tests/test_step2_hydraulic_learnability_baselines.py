"""Focused correctness tests for the Train-only local-effect baseline audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _baseline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_step2_hydraulic_learnability_baselines.py"
    spec = importlib.util.spec_from_file_location("step2_hydraulic_learnability_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_active_metrics_use_per_actuator_flow_scales() -> None:
    """A global median must not classify a high-scale low-normalized flow as active."""
    module = _baseline_module()
    metrics = module.event_balanced_effect_metrics(
        {
            "event": (
                np.asarray([[-0.30, 15.0]], dtype=np.float64),
                np.asarray([[0.30, 15.0]], dtype=np.float64),
                np.asarray([[1.0, 100.0]], dtype=np.float64),
            )
        },
        active_fraction=0.25,
    )
    assert metrics["active_fraction"] == 0.5
    assert metrics["active_sign_accuracy"] == 0.0


def test_baseline_metrics_are_event_balanced_not_row_balanced() -> None:
    """Each event has equal scientific weight even with unequal endpoint row counts."""
    module = _baseline_module()
    metrics = module.event_balanced_effect_metrics(
        {
            "small_event": (
                np.asarray([1.0]),
                np.asarray([1.0]),
                np.asarray([1.0]),
            ),
            "large_event": (
                np.zeros(100, dtype=np.float64),
                np.ones(100, dtype=np.float64),
                np.ones(100, dtype=np.float64),
            ),
        },
        active_fraction=0.25,
    )
    assert metrics["skill_vs_zero"] == 0.5


def test_local_features_keep_identity_and_causal_prefix_distinct() -> None:
    """Two actuators with equal amplitudes remain distinguishable and future action is absent."""
    module = _baseline_module()
    features_a = module.local_feature_row(
        actuator_index=0,
        actuator_count=2,
        current_delta=0.1,
        prefix_delta=0.1,
        up_state=np.arange(6, dtype=np.float64),
        down_state=np.arange(6, 12, dtype=np.float64),
        previous_flow=2.0,
        rainfall_up=3.0,
        rainfall_down=4.0,
        rainfall_mean=5.0,
        retained_minutes=5.0,
        physics=np.asarray([6.0, 7.0]),
    )
    features_b = module.local_feature_row(
        actuator_index=1,
        actuator_count=2,
        current_delta=0.1,
        prefix_delta=0.1,
        up_state=np.arange(6, dtype=np.float64),
        down_state=np.arange(6, 12, dtype=np.float64),
        previous_flow=2.0,
        rainfall_up=3.0,
        rainfall_down=4.0,
        rainfall_mean=5.0,
        retained_minutes=5.0,
        physics=np.asarray([6.0, 7.0]),
    )
    assert features_a.shape == features_b.shape
    assert not np.array_equal(features_a[:2], features_b[:2])
    assert np.array_equal(features_a[2:], features_b[2:])
