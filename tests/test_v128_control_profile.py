from __future__ import annotations

import pytest
import torch

from rtc.v128_control_profile import (
    V128_RANKING_REFERENCE_FRACTION,
    build_v128_control_training_design,
    configure_v128_cuda_matmul_precision,
    v128_training_pair_threshold_m3,
)


def test_v128_keeps_large_event_action_order_labels_at_absolute_floor() -> None:
    design = build_v128_control_training_design()
    assert design.informative_pair_absolute_m3 == pytest.approx(1.0)
    assert design.informative_pair_reference_fraction == pytest.approx(0.0)
    assert V128_RANKING_REFERENCE_FRACTION == pytest.approx(0.0)
    assert v128_training_pair_threshold_m3(500.0) == pytest.approx(1.0)
    assert v128_training_pair_threshold_m3(250_000.0) == pytest.approx(1.0)


def test_v128_profile_refuses_ad_hoc_ranking_fraction_override() -> None:
    with pytest.raises(ValueError, match="fixes informative_pair_reference_fraction"):
        build_v128_control_training_design(informative_pair_reference_fraction=1.0e-3)


def test_v128_matmul_precision_is_explicit_and_restorable() -> None:
    original = torch.get_float32_matmul_precision()
    try:
        report = configure_v128_cuda_matmul_precision("high")
        assert report["float32_matmul_precision"] == "high"
        assert report["amp_enabled"] is False
        assert report["training_design"]["informative_pair_reference_fraction"] == 0.0
    finally:
        torch.set_float32_matmul_precision(original)


def test_v128_matmul_precision_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="highest/high/medium"):
        configure_v128_cuda_matmul_precision("fastest")
