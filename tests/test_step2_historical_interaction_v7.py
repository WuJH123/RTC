from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from rtc.step2_tfv_value import DirectTFVValueDesign
from rtc.step2_tfv_value_historical_v7 import (
    HISTORICAL_INTERACTION_VALUE_CONTRACT,
    HistoricalInteractionTFVValueModelV7,
)
from rtc.step2_tfv_value_training_historical_v7 import (
    HISTORICAL_INTERACTION_TRAINING_CONTRACT,
    HISTORICAL_INTERACTION_UPDATE_POLICY,
    _set_trainable_historical_v7,
)


def _model() -> HistoricalInteractionTFVValueModelV7:
    torch.manual_seed(42)
    return HistoricalInteractionTFVValueModelV7(
        state_dim=4,
        rainfall_dim=1,
        actuator_physics_dim=6,
        target_scale_m3=1000.0,
        design=DirectTFVValueDesign(hidden_dim=32, actuator_embedding_dim=8),
    )


def _inputs() -> dict[str, torch.Tensor]:
    batch = 2
    nodes = 5
    horizon = 72
    actuators = 109
    reference = torch.full((batch, horizon, actuators), 0.5)
    return {
        "current_state": torch.randn(batch, nodes, 4),
        "rainfall": torch.rand(batch, horizon, nodes, 1),
        "reference_settings": reference,
        "candidate_settings": reference.clone(),
        "previous_actuator_flow": torch.randn(batch, actuators),
        "actuator_upstream": torch.arange(actuators) % nodes,
        "actuator_downstream": (torch.arange(actuators) + 1) % nodes,
        "actuator_physics": torch.randn(actuators, 6),
    }


def _trainable_names(model: HistoricalInteractionTFVValueModelV7) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_v7_wrapper_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/run_step2_tfv_value_historical_interaction_v7.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_v7_contract_is_development_distinct() -> None:
    assert HISTORICAL_INTERACTION_VALUE_CONTRACT.endswith("INTERACTION_V7")
    assert HISTORICAL_INTERACTION_TRAINING_CONTRACT.endswith("TRAINING_V7")
    assert HISTORICAL_INTERACTION_UPDATE_POLICY == "MAIN_ALL_THEN_HISTORICAL_INTERACTION_ONLY"


def test_v7_hold_is_exact_zero() -> None:
    model = _model().eval()
    kwargs = _inputs()
    with torch.no_grad():
        out = model(**kwargs)
    assert torch.equal(out.total_delta_tfv_m3, torch.zeros_like(out.total_delta_tfv_m3))
    assert torch.equal(out.interaction_residual_m3, torch.zeros_like(out.interaction_residual_m3))


def test_v7_single_facility_keeps_interaction_exact_zero() -> None:
    model = _model().eval()
    kwargs = _inputs()
    candidate = kwargs["candidate_settings"].clone()
    candidate[:, :2, 0] = 0.8
    kwargs["candidate_settings"] = candidate
    with torch.no_grad():
        out = model(**kwargs)
    assert torch.equal(out.interaction_residual_m3, torch.zeros_like(out.interaction_residual_m3))


def test_v7_candidate_reference_swap_is_antisymmetric() -> None:
    model = _model().eval()
    final = model.historical_interaction_head[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.weight.fill_(0.05)
        final.bias.zero_()
    kwargs = _inputs()
    candidate = kwargs["candidate_settings"].clone()
    candidate[:, :4, 0] = 0.8
    candidate[:, :4, 3] = 0.2
    kwargs["candidate_settings"] = candidate
    with torch.no_grad():
        forward = model(**kwargs)
        reverse_kwargs = dict(kwargs)
        reverse_kwargs["reference_settings"] = candidate
        reverse_kwargs["candidate_settings"] = kwargs["reference_settings"]
        reverse = model(**reverse_kwargs)
    assert torch.allclose(
        forward.total_delta_tfv_m3,
        -reverse.total_delta_tfv_m3,
        atol=2.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        forward.interaction_residual_m3,
        -reverse.interaction_residual_m3,
        atol=2.0e-5,
        rtol=1.0e-5,
    )


def test_v7_joint_and_control_update_only_new_interaction_head() -> None:
    model = _model()
    for stage in ("joint", "control"):
        _set_trainable_historical_v7(model, stage=stage)
        names = _trainable_names(model)
        assert names
        assert all(name.startswith("historical_interaction_head.") for name in names)
        assert not any(name.startswith("facility_encoder.") for name in names)
        assert not any(name.startswith("facility_head.") for name in names)
        assert not any(name.startswith("interaction_head.") for name in names)


def test_v7_main_stage_retains_full_capacity() -> None:
    model = _model()
    _set_trainable_historical_v7(model, stage="main")
    assert _trainable_names(model) == {name for name, _ in model.named_parameters()}


def test_v7_unknown_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown historical V7"):
        _set_trainable_historical_v7(_model(), stage="formal")
