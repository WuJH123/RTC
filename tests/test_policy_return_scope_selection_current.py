from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_direct_tfv_policy_return_scope_selection_current.py"


def _module():
    spec = importlib.util.spec_from_file_location("policy_return_scope_selection", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(*, passed: bool, key: list[float], hold: float = 0.25) -> dict:
    return {
        "validation_selected_epoch": 3,
        "validation_selection_key": key,
        "fine_tuning_improved_over_epoch0": True,
        "fine_tuning_improved_decision_metrics_over_epoch0": passed,
        "validation_action_starvation_detected": False,
        "validation_baseline_metrics": {},
        "validation_metrics": {
            "predicted_hold_fraction": hold,
            "oracle_hold_optimal_fraction": 0.25,
            "hold_aware_decision_accuracy": 0.5 if passed else 0.25,
        },
    }


def test_scope_selection_is_frozen_to_control_heads_and_all() -> None:
    module = _module()
    assert module.PRECOMMITTED_SCOPES == ("control-heads", "all")
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"calibration_used_for_scope_selection": False' in text
    assert '"calibration_scored_before_scope_selection": False' in text
    assert "run_direct_tfv_policy_return_learning_pipeline_current.py" in text
    assert "run_policy_direct_tfv_policy_return_development.py" not in text


def test_scope_selection_chooses_only_deployable_scope() -> None:
    module = _module()
    reports = {
        "control-heads": _report(passed=False, key=[0.6, 0.3, 100.0]),
        "all": _report(passed=True, key=[0.4, 0.2, 50.0]),
    }
    selected, diagnostics = module._select_scope(reports)
    assert selected == "all"
    assert diagnostics["control-heads"]["deployability"]["passed"] is False
    assert diagnostics["all"]["deployability"]["passed"] is True


def test_scope_selection_uses_frozen_validation_key_if_both_pass() -> None:
    module = _module()
    reports = {
        "control-heads": _report(passed=True, key=[0.4, 0.2, 100.0]),
        "all": _report(passed=True, key=[0.3, 0.25, 200.0]),
    }
    selected, _ = module._select_scope(reports)
    assert selected == "all"


def test_scope_selection_fails_closed_when_neither_scope_is_deployable() -> None:
    module = _module()
    reports = {
        "control-heads": _report(passed=False, key=[0.6, 0.3, 100.0]),
        "all": _report(passed=False, key=[0.5, 0.25, 80.0], hold=1.0),
    }
    selected, diagnostics = module._select_scope(reports)
    assert selected is None
    assert diagnostics["control-heads"]["deployability"]["passed"] is False
    assert diagnostics["all"]["deployability"]["passed"] is False


def test_scope_selection_detects_all_hold_starvation() -> None:
    module = _module()
    verdict = module._deployability(
        {
            "fine_tuning_improved_decision_metrics_over_epoch0": True,
            "validation_action_starvation_detected": False,
            "validation_metrics": {
                "predicted_hold_fraction": 1.0,
                "oracle_hold_optimal_fraction": 0.25,
                "hold_aware_decision_accuracy": 0.0,
            },
        }
    )
    assert verdict["passed"] is False
    assert verdict["validation_action_starvation_detected"] is True
