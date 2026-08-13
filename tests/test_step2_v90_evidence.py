from __future__ import annotations

import copy

import pytest

from rtc.step2_v90_contract import LEVEL_A, LEVEL_B, LEVEL_C, V90_CONTRACT
from rtc.step2_v90_evidence import validate_state_sufficiency_evidence_v90


HEAD = "1" * 40
SHA256 = "a" * 64
LEVELS = (LEVEL_A, LEVEL_B, LEVEL_C)


def _payload() -> dict:
    return {
        "contract": V90_CONTRACT,
        "development_only": True,
        "production_compatible": False,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "oracle_level_c_forbidden_online": True,
        "lineage": {
            "git_head": HEAD,
            "graph_sha256": SHA256,
            "cache_manifest_sha256": SHA256,
            "value_checkpoint_sha256": SHA256,
            "hydraulic_checkpoint_sha256": SHA256,
            "fit_d2_group_count": 112,
            "fit_d2_group_digest": SHA256,
            "seed": 42,
        },
        "preflight": {
            level: {
                "signed_state_exact_zero": True,
                "signed_flow_exact_zero": True,
                "reference_frozen": True,
            }
            for level in LEVELS
        },
        "training_history": {
            level: [{"epoch": epoch} for epoch in range(1, 5)] for level in LEVELS
        },
        "ladder": {level: {"overall": {"skill": -0.1}} for level in LEVELS},
        "decision": {
            "decision": "MARKOV_INSUFFICIENCY_SUPPORTED",
            "next_step": "audit existing history",
        },
    }


def test_v90_evidence_accepts_exact_current_lineage():
    result = validate_state_sufficiency_evidence_v90(
        _payload(), expected_git_head=HEAD
    )
    assert result["accepted"] is True
    assert result["git_head"] == HEAD
    assert result["diagnostic_d2_epochs"] == 4


def test_v90_evidence_rejects_premerge_or_stale_git_head():
    with pytest.raises(ValueError, match="stale V9 evidence"):
        validate_state_sufficiency_evidence_v90(
            _payload(), expected_git_head="2" * 40
        )


def test_v90_evidence_rejects_legacy_summary_without_current_contract():
    payload = _payload()
    payload["contract"] = "PROJECT7_STEP2_V90_STATE_SUFFICIENCY_LADDER_V1"
    payload.pop("lineage")
    with pytest.raises(ValueError, match="evidence contract"):
        validate_state_sufficiency_evidence_v90(payload, expected_git_head=HEAD)


def test_v90_evidence_rejects_noncanonical_three_epoch_diagnostic():
    payload = copy.deepcopy(_payload())
    for level in LEVELS:
        payload["training_history"][level] = [
            {"epoch": epoch} for epoch in range(1, 4)
        ]
    with pytest.raises(ValueError, match="epochs=3 != canonical 4"):
        validate_state_sufficiency_evidence_v90(payload, expected_git_head=HEAD)


def test_v90_evidence_rejects_any_validation_or_swmm_access():
    for key in ("validation_accessed", "swmm_run"):
        payload = _payload()
        payload[key] = True
        with pytest.raises(ValueError, match=key):
            validate_state_sufficiency_evidence_v90(payload, expected_git_head=HEAD)
