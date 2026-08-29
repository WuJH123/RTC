from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.direct_tfv_v28_residual_value import V28_RESIDUAL_FEATURE_NAMES
from rtc.direct_tfv_v29_auto_rbc_shadow import (
    V29_AUTO_RBC_UTILITY_SOURCE,
    build_auto_rbc_hydraulic_utility_proposal,
)
from rtc.direct_tfv_v29_regime_value import (
    V29_FEATURE_NAMES,
    build_v29_regime_features,
    fit_v29_regime_value,
    group_balanced_row_weights,
)


def _v28_vector(**overrides: float) -> np.ndarray:
    values = np.zeros(len(V28_RESIDUAL_FEATURE_NAMES), dtype=np.float64)
    for name, value in overrides.items():
        values[V28_RESIDUAL_FEATURE_NAMES.index(name)] = float(value)
    return values


def test_v29_regime_features_are_continuous_and_exclude_event_identity() -> None:
    base = _v28_vector(
        q27_supported_score_m3=-1000.0,
        supported_first_move_l1=2.0,
        changed_facility_count=5.0,
        network_stress_q75=0.4,
        rain_level=10.0,
        strong_storm_blend=0.0,
        candidate_family_hydraulic=1.0,
    )
    feature = build_v29_regime_features(base)
    assert feature.shape == (len(V29_FEATURE_NAMES),)
    joined = "|".join(V29_FEATURE_NAMES).lower()
    assert "event_id" not in joined
    assert "return_period" not in joined
    assert "duration" not in joined
    assert feature[V29_FEATURE_NAMES.index("network_stress_squared")] == 0.16
    assert feature[V29_FEATURE_NAMES.index("q27_x_stress")] == -400.0


def test_group_balanced_weights_give_each_group_equal_total_weight() -> None:
    groups = ["a", "a", "a", "b", "c", "c"]
    weights = group_balanced_row_weights(groups)
    labels = np.asarray(groups, dtype=object)
    totals = [float(weights[labels == group].sum()) for group in sorted(set(groups))]
    assert np.allclose(totals, totals[0])


def test_v29_alpha_zero_is_exact_q27_fallback() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(12, len(V29_FEATURE_NAMES)))
    q27 = rng.normal(size=12) * 1000.0
    truth = q27 + rng.normal(size=12) * 200.0
    groups = [f"g{i // 2}" for i in range(12)]
    units = [f"u{i // 2}" for i in range(12)]
    model, report = fit_v29_regime_value(
        train_features=features,
        train_q27_scores_m3=q27,
        train_truth_m3=truth,
        train_groups=groups,
        train_units=units,
        q27_checkpoint_sha256="a" * 64,
        cv_folds=3,
        ridge_grid=(0.1,),
        shrinkage_grid=(0.0,),
    )
    assert report["selected_shrinkage"] == 0.0
    assert np.allclose(model.weight, 0.0)
    assert model.intercept == 0.0
    assert np.allclose(model.predict_residual_many(features), 0.0)


def _dummy_graph() -> SimpleNamespace:
    node_count = 110
    actuator_count = 109
    static = np.ones((node_count, 1), dtype=np.float64)
    physics_names = (
        "is_pump",
        "is_orifice",
        "is_weir",
        "is_outlet",
        "min_setting",
        "max_setting",
    )
    physics = np.zeros((actuator_count, len(physics_names)), dtype=np.float64)
    physics[:, 0] = 1.0
    physics[:, 4] = 0.0
    physics[:, 5] = 1.0
    return SimpleNamespace(
        node_ids=tuple(f"n{i}" for i in range(node_count)),
        actuator_ids=tuple(f"a{i}" for i in range(actuator_count)),
        actuator_upstream=np.arange(actuator_count, dtype=np.int64),
        actuator_downstream=(np.arange(actuator_count, dtype=np.int64) + 1) % node_count,
        static_node_feature_names=("max_depth_m",),
        static_node_features=static,
        actuator_physics_feature_names=physics_names,
        actuator_physics=physics,
    )


def test_v29_auto_rbc_utility_shadow_respects_frozen_engineering_envelope() -> None:
    graph = _dummy_graph()
    state = torch.zeros((110, 4), dtype=torch.float32)
    state[:, 0] = torch.linspace(0.1, 1.0, 110)
    active = torch.full((109,), 0.2, dtype=torch.float32)
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    radius = np.full(109, 0.5, dtype=np.float64)
    proposal = build_auto_rbc_hydraulic_utility_proposal(
        graph=graph,
        current_state=state,
        active_target=active,
        supervisory_mask=mask,
        first_radius=radius,
        max_changed_facilities=8,
        max_delta_per_update=0.5,
    )
    target = proposal.target.detach().cpu().numpy()
    changed = np.abs(target - active.numpy()) > 1.0e-7
    assert proposal.source == V29_AUTO_RBC_UTILITY_SOURCE
    assert int(changed.sum()) <= 8
    assert np.allclose(target[~mask], active.numpy()[~mask])
    assert np.max(np.abs(target - active.numpy())) <= 0.5000001
    assert np.all((target >= 0.0) & (target <= 1.0))
    assert 0.0 <= proposal.retained_utility_fraction <= 1.0
