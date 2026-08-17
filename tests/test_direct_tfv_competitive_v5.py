from __future__ import annotations

import pytest
import torch

from rtc.direct_tfv_replay_guard import (
    DIRECT_TFV_REPLAY_TARGET_LATCH_CONTRACT,
    reassert_authoritative_target_latch,
    require_identical_prefix,
)
from rtc.step2_tfv_support import DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT
from rtc.step2_tfv_value_training_v4 import _joint_density_balanced_regression
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4


class _Link:
    def __init__(self, target: float) -> None:
        self.target_setting = target


def _bare_mpc(support: dict, *, quantile: str = "q95") -> DirectTFVRecedingMPCV4:
    mpc = object.__new__(DirectTFVRecedingMPCV4)
    mpc.action_support = dict(support)
    mpc.design = DirectTFVMPCDesignV4(active_support_quantile=quantile)
    return mpc


def test_v4_active_density_falls_back_to_q90_for_legacy_support() -> None:
    mpc = _bare_mpc({"joint_changed_facility_count_q90": 22.0})
    assert mpc.active_support_quantile_effective() == "q90"
    assert mpc.active_support_ceiling() == 22
    assert mpc._active_count(80) == 22


def test_v4_uses_q95_but_never_exceeds_observed_joint_max() -> None:
    mpc = _bare_mpc(
        {
            "joint_density_extension_contract": DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT,
            "joint_changed_facility_count_q90": 22.0,
            "joint_changed_facility_count_q95": 31.2,
            "joint_changed_facility_count_q99": 45.0,
            "joint_changed_facility_count_max": 34,
        }
    )
    assert mpc.active_support_quantile_effective() == "q95"
    assert mpc.active_support_ceiling() == 32
    assert mpc._active_count(80) == 32

    q99 = _bare_mpc(mpc.action_support, quantile="q99")
    assert q99.active_support_ceiling() == 34
    assert q99._active_count(80) == 34


def test_joint_density_regression_balances_changed_count_strata() -> None:
    prediction = torch.tensor([0.0, 0.0, 0.0, 0.0])
    truth = torch.tensor([1.0, 1.0, 1.0, 10.0])
    counts = torch.tensor([2, 2, 2, 20])
    scale = torch.tensor(1.0)
    balanced = _joint_density_balanced_regression(
        prediction,
        truth,
        scale_m3=scale,
        changed_facility_counts=counts,
    )
    raw = torch.nn.functional.smooth_l1_loss(prediction, truth)
    assert float(balanced) > float(raw)


def test_replay_reasserts_same_authoritative_target_latch_and_fails_closed() -> None:
    links = {"P1": _Link(0.46), "O1": _Link(0.72)}
    discrepancy = reassert_authoritative_target_latch(links, {"P1": 0.50, "O1": 0.70})
    assert discrepancy == pytest.approx(0.04)
    assert links["P1"].target_setting == pytest.approx(0.50)
    assert links["O1"].target_setting == pytest.approx(0.70)

    evidence = require_identical_prefix(
        candidate_state=[1.0, 2.0],
        hold_state=[1.0, 2.0],
        candidate_target=[0.5, 0.7],
        hold_target=[0.5, 0.7],
        candidate_current=[0.48, 0.69],
        hold_current=[0.48, 0.69],
        candidate_statistics=[10.0],
        hold_statistics=[10.0],
        hold_reference_target=[0.5, 0.7],
    )
    assert evidence["contract"] == DIRECT_TFV_REPLAY_TARGET_LATCH_CONTRACT

    with pytest.raises(RuntimeError, match="COUNTERFACTUAL_REPLAY_P0"):
        require_identical_prefix(
            candidate_state=[1.0], hold_state=[1.0],
            candidate_target=[0.5], hold_target=[0.5],
            candidate_current=[0.5], hold_current=[0.5],
            candidate_statistics=[1.0], hold_statistics=[1.0],
            hold_reference_target=[0.46],
        )
