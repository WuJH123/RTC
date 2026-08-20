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
    validate_policy_return_portfolio_record,
)

from scripts.audit_direct_tfv_policy_return_mechanism_panel_current import _read_records


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "audit_direct_tfv_policy_return_mechanism_panel_current.py"


def _sha(ch: str) -> str:
    return ch * 64


def _record() -> dict:
    hold = 1000.0
    truth = -25.0
    return {
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "data_role": "policy_return_development_diagnostic",
        "development_diagnostic_only": True,
        "eligible_for_learning_dataset": False,
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
        "base_step2_h10_score_m3": -10.0,
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


def test_diagnostic_portfolio_record_passes_but_learning_firewall_rejects() -> None:
    row = _record()
    validate_policy_return_portfolio_record(row)
    with pytest.raises(ValueError, match="train/validation/calibration"):
        validate_policy_return_learning_record(row)


def test_mechanism_audit_reads_diagnostic_record(tmp_path: Path) -> None:
    record_path = tmp_path / "diagnostic.jsonl"
    record_path.write_text(__import__("json").dumps(_record()) + "\n", encoding="utf-8")
    rows = _read_records([str(record_path)])
    assert len(rows) == 1
    assert rows[0]["data_role"] == "policy_return_development_diagnostic"


@pytest.mark.parametrize(
    ("field", "value"),
    (("eligible_for_learning_dataset", True), ("development_diagnostic_only", False)),
)
def test_invalid_diagnostic_flags_fail_closed(field: str, value: object) -> None:
    row = _record()
    row[field] = value
    with pytest.raises(ValueError, match="Development mechanism truth"):
        validate_policy_return_portfolio_record(row)


def test_mechanism_audit_uses_portfolio_validator() -> None:
    text = AUDIT.read_text(encoding="utf-8")
    assert "validate_policy_return_portfolio_record" in text
    assert "validate_policy_return_record(row)" not in text
