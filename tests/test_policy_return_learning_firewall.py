from __future__ import annotations

from pathlib import Path

import pytest

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
)
from rtc.direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
)
from rtc.direct_tfv_policy_return_portfolio_admission import (
    validate_policy_return_learning_record,
)


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "scripts" / "compile_direct_tfv_policy_return_dataset_current.py"
SCORER = ROOT / "scripts" / "score_direct_tfv_policy_return_calibration_current.py"
CALIBRATOR = ROOT / "scripts" / "calibrate_direct_tfv_policy_return_portfolio_admission_current.py"


def _sha(ch: str) -> str:
    return ch * 64


def _record(*, role: str = "policy_return_train") -> dict:
    hold = 1000.0
    truth = -25.0
    return {
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "data_role": role,
        "development_diagnostic_only": False,
        "eligible_for_learning_dataset": True,
        "rainfall_group": "G001",
        "event_id": "event-G001",
        "decision_index": 3,
        "decision_elapsed_seconds": 5400,
        "query_set_id": _sha("1"),
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "supervisory_mask_sha256": _sha("2"),
        "passive_setting_channels_unchanged": True,
        "first_move_changed_facility_count": 4,
        "true_policy_return_delta_tfv_m3": truth,
        "candidate_branch_tfv_m3": hold + truth,
        "hold_branch_tfv_m3": hold,
        "shared_hold_branch": True,
        "same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "future_realized_rainfall_used_online": False,
        "continuation_policy_sha256": _sha("3"),
        "prefix_sha256": _sha("4"),
        "candidate_first_target_sha256": _sha("5"),
        "hold_first_target_sha256": _sha("6"),
        "target_write_readback_verified": True,
        "engineering_bounds_verified": True,
        "candidate_manifest_support_lineage_verified": True,
        "projected_gradient_online": False,
        "online_lbfgsb_used": False,
        "candidate_manifest_sha256": _sha("7"),
        "parent_decisions_sha256": _sha("8"),
        "source_inp_sha256": _sha("9"),
        "asset_manifest_sha256": _sha("a"),
        "graph_sha256": _sha("b"),
        "base_step2_sha256": _sha("c"),
        "sequence_support_sha256": _sha("d"),
        "supervisory_control_sha256": _sha("e"),
        "candidate_flow_routing_error_pct": 0.1,
        "hold_flow_routing_error_pct": 0.1,
    }


def test_learning_firewall_accepts_current_authoritative_truth() -> None:
    validate_policy_return_learning_record(_record())


def test_learning_firewall_rejects_development_mechanism_truth() -> None:
    row = _record(role="policy_return_development_diagnostic")
    row["development_diagnostic_only"] = True
    row["eligible_for_learning_dataset"] = False
    with pytest.raises(ValueError, match="train/validation/calibration"):
        validate_policy_return_learning_record(row)


def test_learning_firewall_rejects_unverified_truth_gate() -> None:
    row = _record()
    row["target_write_readback_verified"] = False
    with pytest.raises(ValueError, match="target_write_readback_verified"):
        validate_policy_return_learning_record(row)


def test_learning_firewall_rejects_noncanonical_provenance() -> None:
    row = _record()
    row["candidate_manifest_sha256"] = "stale"
    with pytest.raises(ValueError, match="candidate_manifest_sha256"):
        validate_policy_return_learning_record(row)


def test_learning_entrypoints_use_the_same_strict_validator() -> None:
    for path in (COMPILER, SCORER, CALIBRATOR):
        text = path.read_text(encoding="utf-8")
        assert "validate_policy_return_learning_record" in text
    assert "validate_policy_return_record(row)" not in COMPILER.read_text(encoding="utf-8")
