from __future__ import annotations

import numpy as np
import pytest

from rtc.direct_tfv_operational_v28_runtime import (
    V28_OPERATIONAL_RUNTIME_CONTRACT,
    V28Candidate,
    build_v28_lineage,
    post_q95_deduplicate,
    select_v28_candidate,
    validate_v28_executable_candidate,
)
from rtc.direct_tfv_v28_residual_value import (
    V28_RESIDUAL_FEATURE_NAMES,
    V28ResidualValueModel,
    build_v28_residual_features,
    q28_score_m3,
)


def _candidate(source: str, target: list[float], q27: float) -> V28Candidate:
    return V28Candidate(
        source=source,
        target=np.asarray(target, dtype=np.float64),
        supported_target=np.asarray(target, dtype=np.float64),
        supported_sequence=np.asarray([target, target], dtype=np.float64),
        q27_score_m3=q27,
        residual_m3=0.0,
        q28_score_m3=q27,
        q95_scale=0.5,
        q95_max_ratio=1.0,
        q95_binding=True,
        raw_first_move_l1=1.0,
        supported_first_move_l1=0.5,
        raw_to_supported_first_move_l1=0.5,
        raw_to_supported_h120_l1=1.0,
        raw_to_supported_tv_l1=0.5,
        changed_facility_count=2,
        network_stress_q75=0.8,
        rain_level=0.5,
        strong_storm_blend=0.6,
        candidate_selected=False,
        contributing_sources=(source,),
    )


def test_raw_action_never_executes_and_q95_is_mandatory() -> None:
    raw = np.asarray([0.0, 0.4, 0.0], dtype=np.float64)
    supported = np.asarray([0.0, 0.2, 0.0], dtype=np.float64)
    candidate = _candidate("TYPE_AWARE_HYDRAULIC_PRESSURE", supported.tolist(), -2.0)
    candidate = candidate.__class__(**{**candidate.__dict__, "target": raw, "supported_target": supported})
    chosen = select_v28_candidate([candidate])
    assert chosen is candidate
    assert np.array_equal(chosen.target, raw)
    assert np.array_equal(chosen.supported_target, supported)
    assert not np.array_equal(chosen.executed_target, raw)
    assert np.array_equal(chosen.executed_target, supported)


def test_post_q95_dedup_keeps_canonical_source_and_contributors() -> None:
    first = _candidate("STEP2_H10_PROBE_SCALE_0.50", [0.0, 0.2, 0.0], -1.0)
    second = _candidate("TYPE_AWARE_HYDRAULIC_PRESSURE", [0.0, 0.2, 0.0], -2.0)
    unique, duplicate_count = post_q95_deduplicate([first, second])
    assert len(unique) == 1
    assert duplicate_count == 1
    assert unique[0].contributing_sources == (
        "STEP2_H10_PROBE_SCALE_0.50",
        "TYPE_AWARE_HYDRAULIC_PRESSURE",
    )


def test_hold_score_is_exactly_zero_and_q28_is_additive() -> None:
    assert q28_score_m3(0.0, 0.0) == 0.0
    assert q28_score_m3(-12.5, 2.5) == pytest.approx(-10.0)


def test_zero_residual_reproduces_v27_selection() -> None:
    candidates = [_candidate("a", [0.0, 0.1, 0.0], 2.0), _candidate("b", [0.0, 0.2, 0.0], -1.0)]
    chosen = select_v28_candidate(candidates, residuals=[0.0, 0.0])
    assert chosen.source == "b"


def test_value_residual_can_change_selection_without_source_priority() -> None:
    candidates = [_candidate("AUTO_RBC_SHADOW_TOPK", [0.0, 0.1, 0.0], -1.0), _candidate("hydraulic", [0.0, 0.2, 0.0], -0.5)]
    chosen = select_v28_candidate(candidates, residuals=[3.0, 0.0])
    assert chosen.source == "hydraulic"


def test_passive_channels_are_never_executable() -> None:
    active = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
    supported = np.asarray([0.4, 0.5, 0.5], dtype=np.float64)
    mask = np.asarray([True, False, True])
    assert validate_v28_executable_candidate(supported, active, mask)
    with pytest.raises(ValueError, match="passive"):
        validate_v28_executable_candidate(np.asarray([0.4, 0.4, 0.5]), active, mask)


def test_residual_feature_contract_has_no_event_id_and_is_structured() -> None:
    assert len(V28_RESIDUAL_FEATURE_NAMES) >= 12
    assert not any("event" in name.lower() for name in V28_RESIDUAL_FEATURE_NAMES)
    vector = build_v28_residual_features(
        q27_score_m3=-3.0,
        q95_scale=0.75,
        q95_max_ratio=1.0,
        q95_binding=True,
        raw_first_move_l1=2.0,
        supported_first_move_l1=1.0,
        raw_to_supported_first_move_l1=1.0,
        raw_to_supported_h120_l1=4.0,
        raw_to_supported_tv_l1=2.0,
        changed_facility_count=3,
        network_stress_q75=0.9,
        rain_level=0.6,
        strong_storm_blend=0.8,
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
    )
    assert vector.shape == (len(V28_RESIDUAL_FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_lineage_firewall_is_development_only() -> None:
    lineage = build_v28_lineage(
        q27_checkpoint_sha256="a" * 64,
        residual_checkpoint_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        asset_manifest_sha256="d" * 64,
    )
    assert lineage["contract"] == V28_OPERATIONAL_RUNTIME_CONTRACT
    assert lineage["development_only"] is True
    assert lineage["formal_evidence"] is False
    assert lineage["ready_for_policy_lock"] is False
    assert lineage["raw_action_executable"] is False


def test_residual_model_requires_frozen_q27_lineage() -> None:
    model = V28ResidualValueModel(
        feature_mean=np.zeros(len(V28_RESIDUAL_FEATURE_NAMES)),
        feature_scale=np.ones(len(V28_RESIDUAL_FEATURE_NAMES)),
        weight=np.zeros(len(V28_RESIDUAL_FEATURE_NAMES)),
        intercept=0.0,
        q27_checkpoint_sha256="a" * 64,
        ridge=0.01,
    )
    assert model.predict_m3(np.zeros(len(V28_RESIDUAL_FEATURE_NAMES))) == 0.0
