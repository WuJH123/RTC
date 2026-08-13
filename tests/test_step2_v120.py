from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.production_v120_router import is_v120_bundle
from rtc.step2_policy_v120 import RuntimeNormalizationV120, _upper_tail_cvar_per_candidate
from rtc.step2_v120_contract import Step2V120Contract, V120_BUNDLE_CONTRACT, V120_CONTRACT


def test_v120_contract_is_tfv_only_and_first_move() -> None:
    contract = Step2V120Contract()
    contract.validate()
    assert contract.primary_objective == "whole_system_cumulative_TFV_m3"
    assert contract.step2_target == "direct_signed_authoritative_delta_TFV_m3"
    assert contract.hydraulic_model_required_online is False
    assert contract.nodewise_action_effect_required is False
    assert contract.sum_d2_effects_for_joint_action is False
    assert contract.execute_first_move_only is True


def test_v120_refuses_hydraulic_or_additive_reintroduction() -> None:
    with pytest.raises(ValueError):
        Step2V120Contract(hydraulic_model_required_online=True).validate()
    with pytest.raises(ValueError):
        Step2V120Contract(sum_d2_effects_for_joint_action=True).validate()
    with pytest.raises(ValueError):
        Step2V120Contract(execute_first_move_only=False).validate()


def test_runtime_normalization_round_trip_and_scaling() -> None:
    norm = RuntimeNormalizationV120(
        state_mean=np.asarray([1.0, 2.0], np.float32),
        state_std=np.asarray([2.0, 4.0], np.float32),
        rainfall_mean=np.asarray([10.0], np.float32),
        rainfall_std=np.asarray([5.0], np.float32),
        flow_mean=np.asarray([2.0], np.float32),
        flow_std=np.asarray([2.0], np.float32),
    )
    loaded = RuntimeNormalizationV120.from_payload(norm.as_payload())
    assert torch.allclose(
        loaded.state(torch.tensor([[[3.0, 6.0]]])),
        torch.tensor([[[1.0, 1.0]]]),
    )
    assert torch.allclose(loaded.rainfall(torch.tensor([[[[15.0]]]])), torch.ones(1, 1, 1, 1))
    assert torch.allclose(loaded.flow(torch.tensor([[4.0]])), torch.ones(1, 1))


def test_cvar_uses_worst_rainfall_tail() -> None:
    values = torch.tensor([
        [-10.0, 5.0],
        [-5.0, 2.0],
        [1.0, 7.0],
    ])
    # With three scenarios and alpha=.9, one worst scenario remains.
    risk = _upper_tail_cvar_per_candidate(values, 0.90)
    assert torch.equal(risk, torch.tensor([1.0, 7.0]))


def test_router_identifies_only_v120_bundle(tmp_path: Path) -> None:
    good = tmp_path / "good.pt"
    old = tmp_path / "old.pt"
    torch.save({"bundle_contract": V120_BUNDLE_CONTRACT, "step2_contract": V120_CONTRACT}, good)
    torch.save({"contract": "PROJECT7_STEP2_V110"}, old)
    assert is_v120_bundle(str(good)) is True
    assert is_v120_bundle(str(old)) is False
    assert is_v120_bundle(None) is False
