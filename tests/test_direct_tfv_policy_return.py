from __future__ import annotations

import torch
import pytest

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    derive_policy_return_admission,
    encode_policy_return_action_token,
    policy_return_margin_m3,
    validate_policy_return_record,
)


def _sha(seed: str) -> str:
    return (seed * 64)[:64]


def _row(group: str, *, pred: float, truth: float, changed: int = 4) -> dict:
    hold = 1000.0
    return {
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "data_role": "policy_return_calibration",
        "rainfall_group": group,
        "event_id": f"event_{group}",
        "first_move_changed_facility_count": changed,
        "predicted_policy_return_delta_tfv_m3": pred,
        "true_policy_return_delta_tfv_m3": truth,
        "candidate_branch_tfv_m3": hold + truth,
        "hold_branch_tfv_m3": hold,
        "same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "future_realized_rainfall_used_online": False,
        "continuation_policy_sha256": "a" * 64,
        "prefix_sha256": _sha("b"),
        "candidate_first_target_sha256": _sha("c"),
        "hold_first_target_sha256": _sha("d"),
    }


def test_h10_action_token_changes_only_first_control_block() -> None:
    active = torch.full((2, 109), 0.5)
    target = active.clone()
    target[:, :3] = 0.8
    reference, candidate = encode_policy_return_action_token(
        active,
        target,
        horizon_steps=72,
        first_action_steps=2,
    )
    assert tuple(reference.shape) == (2, 72, 109)
    assert tuple(candidate.shape) == (2, 72, 109)
    assert torch.equal(reference[:, 0], active)
    assert torch.equal(candidate[:, 0], target)
    assert torch.equal(candidate[:, 1], target)
    assert torch.equal(candidate[:, 2:], reference[:, 2:])
    assert not torch.equal(candidate[:, 0], reference[:, 0])


def test_policy_return_record_requires_h10_action_encoding() -> None:
    row = _row("g0", pred=-10.0, truth=-5.0)
    row["action_encoding_contract"] = "PERSISTENT_H360"
    with pytest.raises(ValueError, match="wrong H10 action encoding"):
        validate_policy_return_record(row)


def test_policy_return_record_requires_same_continuation_policy() -> None:
    row = _row("g0", pred=-10.0, truth=-5.0)
    row["same_continuation_policy_verified"] = False
    with pytest.raises(ValueError, match="same frozen continuation policy"):
        validate_policy_return_record(row)


def test_policy_return_admission_is_rainfall_group_based() -> None:
    rows = [_row(f"g{i:02d}", pred=-100.0, truth=-80.0 + i, changed=4) for i in range(24)]
    admission = derive_policy_return_admission(
        records=rows,
        expected_rainfall_groups=[f"g{i:02d}" for i in range(24)],
        policy_return_checkpoint_sha256="e" * 64,
        continuation_policy_sha256="a" * 64,
        coverage=0.90,
    )
    assert admission["contract"] == DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT
    assert admission["action_encoding_contract"] == DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
    assert admission["calibration_rainfall_group_count"] == 24
    assert admission["generic_d3_floor_controls_execution"] is False
    assert admission["open_loop_first_move_margin_controls_execution"] is False
    assert policy_return_margin_m3(admission, 4) >= 0.0


def test_policy_return_truth_must_equal_paired_branch_difference() -> None:
    row = _row("g0", pred=-10.0, truth=-5.0)
    row["candidate_branch_tfv_m3"] += 50.0
    with pytest.raises(ValueError, match="inconsistent with paired authoritative TFV"):
        validate_policy_return_record(row)
