from __future__ import annotations

import numpy as np

from rtc.direct_tfv_v28_residual_value import V28_RESIDUAL_FEATURE_NAMES
from rtc.direct_tfv_v28r1_supported_residual import (
    V28R1_SUPPORTED_FEATURE_INDICES,
    V28R1_SUPPORTED_FEATURE_NAMES,
    V28R1_ZERO_WEIGHT_FEATURE_NAMES,
    fit_v28r1_supported_residual,
)


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(42)
    rows = 24
    x = rng.normal(size=(rows, len(V28_RESIDUAL_FEATURE_NAMES)))
    q27 = rng.normal(scale=100.0, size=rows)
    # Truth depends only on deployment-parity supported features.
    safe = x[:, V28R1_SUPPORTED_FEATURE_INDICES]
    truth = q27 + 8.0 * safe[:, 0] - 4.0 * safe[:, 1] + 2.0 * safe[:, 2]
    groups = [f"g{index // 4}" for index in range(rows)]
    units = [f"u{index // 2}" for index in range(rows)]
    return x, q27, truth, groups, units


def test_supported_feature_contract_excludes_raw_q95_geometry() -> None:
    assert "q27_supported_score_m3" in V28R1_SUPPORTED_FEATURE_NAMES
    assert "supported_first_move_l1" in V28R1_SUPPORTED_FEATURE_NAMES
    assert "q95_contraction_scale" in V28R1_ZERO_WEIGHT_FEATURE_NAMES
    assert "raw_first_move_l1" in V28R1_ZERO_WEIGHT_FEATURE_NAMES
    assert "raw_to_supported_first_move_l1" in V28R1_ZERO_WEIGHT_FEATURE_NAMES
    assert "raw_to_supported_h120_l1" in V28R1_ZERO_WEIGHT_FEATURE_NAMES
    assert "raw_to_supported_total_variation_l1" in V28R1_ZERO_WEIGHT_FEATURE_NAMES


def test_v28r1_fit_assigns_exact_zero_weight_to_mismatched_features() -> None:
    x, q27, truth, groups, units = _toy()
    model, report = fit_v28r1_supported_residual(
        train_features=x,
        train_q27_scores_m3=q27,
        train_truth_m3=truth,
        train_groups=groups,
        train_units=units,
        q27_checkpoint_sha256="a" * 64,
        cv_folds=3,
        ridge_grid=(0.0, 1.0),
        shrinkage_grid=(0.0, 0.5, 1.0),
    )
    safe = set(V28R1_SUPPORTED_FEATURE_INDICES)
    for index, name in enumerate(V28_RESIDUAL_FEATURE_NAMES):
        if index not in safe:
            assert model.weight[index] == 0.0, name
    assert report["validation_used_for_model_selection"] is False
    assert report["test_used_for_model_selection"] is False

    # Deployment-only changes in raw/q95 geometry cannot affect a supported-manifold prediction.
    shifted = x.copy()
    for index in range(shifted.shape[1]):
        if index not in safe:
            shifted[:, index] += 1.0e6
    np.testing.assert_allclose(model.predict_many(x), model.predict_many(shifted), rtol=0.0, atol=0.0)


def test_alpha_zero_is_exact_q27_fallback_candidate() -> None:
    x, q27, truth, groups, units = _toy()
    model, report = fit_v28r1_supported_residual(
        train_features=x,
        train_q27_scores_m3=q27,
        train_truth_m3=q27,  # Q27 is already exact; correction should be zero.
        train_groups=groups,
        train_units=units,
        q27_checkpoint_sha256="b" * 64,
        cv_folds=3,
        ridge_grid=(0.0, 1.0),
        shrinkage_grid=(0.0, 1.0),
    )
    assert report["selected_residual_shrinkage"] == 0.0
    np.testing.assert_allclose(model.predict_many(x), 0.0, rtol=0.0, atol=0.0)
