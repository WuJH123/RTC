from __future__ import annotations

import pytest

from rtc.step2_tfv_value import DirectFacilityTFVValueModel, DirectTFVValueDesign
from rtc.step2_tfv_value_training_historical_v6 import (
    HISTORICAL_RETRAIN_CONTRACT,
    HISTORICAL_UPDATE_POLICY,
    _set_trainable_historical,
)


def _model() -> DirectFacilityTFVValueModel:
    return DirectFacilityTFVValueModel(
        state_dim=4,
        rainfall_dim=1,
        actuator_physics_dim=6,
        target_scale_m3=1000.0,
        design=DirectTFVValueDesign(hidden_dim=32, actuator_embedding_dim=8),
    )


def _trainable_names(model: DirectFacilityTFVValueModel) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_historical_retrain_has_distinct_development_contract() -> None:
    assert HISTORICAL_RETRAIN_CONTRACT == "PROJECT7_DIRECT_TFV_HISTORICAL_PRESERVE_MAIN_RETRAIN_V6"
    assert HISTORICAL_UPDATE_POLICY == "MAIN_ALL_THEN_INTERACTION_ONLY"


def test_main_stage_keeps_full_v5_capacity() -> None:
    model = _model()
    _set_trainable_historical(model, stage="main")
    assert _trainable_names(model) == {name for name, _ in model.named_parameters()}


def test_joint_stage_freezes_main_backbone_and_updates_only_interaction() -> None:
    model = _model()
    _set_trainable_historical(model, stage="joint")
    names = _trainable_names(model)
    assert names
    assert all(name.startswith("interaction_head.") for name in names)
    assert not any(name.startswith("facility_encoder.") for name in names)
    assert not any(name.startswith("facility_head.") for name in names)
    assert not any(name.startswith("global_state_encoder.") for name in names)
    assert not any(name.startswith("rainfall_encoder.") for name in names)


def test_control_stage_preserves_same_main_backbone() -> None:
    model = _model()
    _set_trainable_historical(model, stage="control")
    names = _trainable_names(model)
    assert names
    assert all(name.startswith("interaction_head.") for name in names)


def test_unknown_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown historical Direct-TFV stage"):
        _set_trainable_historical(_model(), stage="formal")
