from __future__ import annotations

import json

from scripts.audit_direct_tfv_closed_loop_current import _verify_no_control_baseline


def _metadata(strategy: str) -> dict:
    return {
        "strategy": strategy,
        "source_inp_sha256": "inp",
        "controller_config_sha256": "cfg",
        "swmm_engine_version": "5.2.4",
        "prepared_event_clock": {
            "effective_warmup_minutes": 120.0,
            "post_rain_tail_minutes": 360.0,
        },
    }


def test_no_control_baseline_provenance_verifies_matched_lineage(tmp_path) -> None:
    proposed = _metadata("proposed_direct_tfv_all109_receding_mpc")
    baseline = _metadata("no_control")
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    verified, loaded, notes = _verify_no_control_baseline(
        proposed=proposed, baseline_metadata=path
    )
    assert verified is True
    assert loaded == baseline
    assert notes == []


def test_baseline_provenance_rejects_wrong_strategy(tmp_path) -> None:
    proposed = _metadata("proposed_direct_tfv_all109_receding_mpc")
    baseline = _metadata("auto_rbc")
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    try:
        _verify_no_control_baseline(proposed=proposed, baseline_metadata=path)
    except ValueError as exc:
        assert "expected 'no_control'" in str(exc)
    else:
        raise AssertionError("Direct-TFV benefit audit accepted Auto-RBC as the no-control baseline")


def test_historical_audit_without_baseline_metadata_is_explicitly_unverified() -> None:
    verified, loaded, notes = _verify_no_control_baseline(
        proposed=_metadata("proposed_direct_tfv_all109_receding_mpc"), baseline_metadata=None
    )
    assert verified is False
    assert loaded is None
    assert notes == ["baseline metadata not supplied"]
