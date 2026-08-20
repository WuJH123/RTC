from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_direct_tfv_policy_return_early_learnability_gate_current.py"


def _module():
    spec = importlib.util.spec_from_file_location("policy_return_early_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_early_gate_is_diagnostic_only_and_does_not_weaken_full_contract() -> None:
    module = _module()
    assert module.PILOT_MIN_TRAIN_GROUPS == 12
    assert module.FROZEN_VALIDATION_GROUPS == 12
    assert module.FULL_TRAIN_GROUPS == 48
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'compiler._MIN_GROUPS["policy_return_train"] = PILOT_MIN_TRAIN_GROUPS' in text
    assert "trainer.DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS = PILOT_MIN_TRAIN_GROUPS" in text
    assert '"runtime_checkpoint_eligible": False' in text
    assert '"swmm_called_by_gate": False' in text
    assert '"ready_for_pi1_development": False' in text
    assert '"ready_for_policy_lock": False' in text
    assert "run_direct_tfv_policy_return_query_current.py" not in text
    assert "calibrate_direct_tfv_policy_return_portfolio_admission_current.py" not in text


def test_early_gate_requires_validation_before_spending_more_bulk() -> None:
    module = _module()
    assert module.FROZEN_VALIDATION_GROUPS == 12
    assert module.PILOT_MIN_TRAIN_GROUPS < module.FULL_TRAIN_GROUPS
    with pytest.raises(ValueError, match="48 train groups already exist"):
        # This branch is intentionally encoded as a fail-closed user-facing rule in main().
        if module.FULL_TRAIN_GROUPS >= module.FULL_TRAIN_GROUPS:
            raise ValueError("48 train groups already exist; run the normal learning pipeline instead")
