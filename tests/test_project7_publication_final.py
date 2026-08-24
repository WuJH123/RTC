from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtc.project7_publication_final import (
    EXPECTED_DISTRIBUTION_SHIFT_DISPOSITION,
    EXPECTED_PFV_CONTRACT,
    EXPECTED_STEP2_SHA256,
    PUBLICATION_FINAL_CONTRACT,
    validate_publication_controller_contract,
    validate_publication_policy_lock,
    validate_publication_validation,
)


REPO = Path(__file__).resolve().parents[1]


def _controller() -> dict:
    return json.loads(
        (REPO / "configs" / "project7_publication_final_controller_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_publication_controller_contract_is_frozen_and_valid() -> None:
    payload = _controller()
    assert payload["contract"] == PUBLICATION_FINAL_CONTRACT
    validate_publication_controller_contract(payload)
    assert payload["step2"]["checkpoint_sha256"] == EXPECTED_STEP2_SHA256
    assert payload["step2"]["development_followups"]["v8_allowed"] is False
    assert payload["step3"]["continuous_gradient_mpc"] is False
    assert payload["step3"]["v21_boundary_candidate_distribution_exact_match"] is False
    assert payload["step3"]["distribution_shift_disposition"] == EXPECTED_DISTRIBUTION_SHIFT_DISPOSITION
    assert payload["step3"]["distribution_shift_must_be_reported_as_component_limitation"] is True
    assert payload["research_goal"]["priority8_safety_contract"] == EXPECTED_PFV_CONTRACT


def test_publication_contract_rejects_reopening_step2_search() -> None:
    payload = _controller()
    payload["step2"]["development_followups"]["v8_allowed"] = True
    with pytest.raises(RuntimeError, match="reopened Step2 architecture search"):
        validate_publication_controller_contract(payload)


def test_publication_contract_rejects_gradient_mpc_claim() -> None:
    payload = _controller()
    payload["step3"]["continuous_gradient_mpc"] = True
    with pytest.raises(RuntimeError, match="continuous gradient MPC"):
        validate_publication_controller_contract(payload)


def test_publication_contract_rejects_hiding_boundary_distribution_shift() -> None:
    payload = _controller()
    payload["step3"]["distribution_shift_must_be_reported_as_component_limitation"] = False
    with pytest.raises(RuntimeError, match="distribution-shift limitation"):
        validate_publication_controller_contract(payload)


def test_policy_lock_requires_same_v5_and_pfv_contract() -> None:
    lock = {
        "contract": "PROJECT7_V23_POLICY_LOCK_V1",
        "locked": True,
        "formal_mode": "FIXED_POLICY_NO_RETRAIN",
        "step2_checkpoint_sha256": EXPECTED_STEP2_SHA256,
        "step2_runtime_lineage_accepted": True,
        "policy_mutation_after_lock_forbidden": True,
        "final_can_tune_policy": False,
        "pfv_safety_contract": EXPECTED_PFV_CONTRACT,
    }
    validate_publication_policy_lock(lock)
    lock["step2_checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="another Step2"):
        validate_publication_policy_lock(lock)


def test_validation_requires_pfv_engineering_and_final_firewall() -> None:
    validation = {
        "contract": "PROJECT7_V23_FORMAL_DEVELOPMENT_VALIDATION_EVIDENCE_V1",
        "pfv_safety_all_events_pass": True,
        "engineering_all_events_pass": True,
        "policy_changed_after_validation_started": False,
        "final_truth_opened": False,
    }
    validate_publication_validation(validation)
    validation["pfv_safety_all_events_pass"] = False
    with pytest.raises(RuntimeError, match="PFV safety"):
        validate_publication_validation(validation)
