from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_direct_tfv_policy_return_learning_pipeline_current.py"


def _module():
    spec = importlib.util.spec_from_file_location("policy_return_learning_pipeline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(group: str, source: str) -> dict:
    return {
        "rainfall_group": group,
        "continuation_policy_sha256": "a" * 64,
        "supervisory_mask_sha256": "b" * 64,
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "candidate_source": source,
    }


def test_pipeline_role_audit_enforces_disjoint_groups_and_frozen_lineage() -> None:
    module = _module()
    train = [_row("train-1", "STEP2_H10_PROBE_SCALE_0.50")]
    validation = [_row("validation-1", "TYPE_AWARE_HYDRAULIC_PRESSURE")]
    calibration = [
        _row("calibration-1", "STEP2_H10_PROBE_SCALE_0.50"),
        _row("calibration-2", "TYPE_AWARE_HYDRAULIC_PRESSURE"),
    ]
    audit = module._audit_roles(train, validation, calibration)
    assert audit["role_disjoint"] is True
    assert audit["authoritative_truth_firewall_verified"] is True
    assert audit["development_diagnostic_rows_allowed"] is False
    assert audit["supervisory_mask_sha256"] == "b" * 64

    overlap_validation = [_row("train-1", "TYPE_AWARE_HYDRAULIC_PRESSURE")]
    with pytest.raises(ValueError, match="role overlap"):
        module._audit_roles(train, overlap_validation, calibration)


def test_pipeline_refuses_lineage_and_noncurrent_family_drift() -> None:
    module = _module()
    train = [_row("train-1", "STEP2_H10_PROBE_SCALE_0.50")]
    validation = [_row("validation-1", "TYPE_AWARE_HYDRAULIC_PRESSURE")]
    calibration = [_row("calibration-1", "STEP2_H10_PROBE_SCALE_0.50")]

    changed_mask = dict(validation[0])
    changed_mask["supervisory_mask_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="supervisory-control masks"):
        module._audit_roles(train, [changed_mask], calibration)

    bad_family = dict(validation[0])
    bad_family["candidate_source"] = "SUPPORT_CONSTRAINED_GRADIENT_H10"
    with pytest.raises(ValueError, match="non-current candidate family"):
        module._audit_roles(train, [bad_family], calibration)


def test_pipeline_deployability_rejects_all_hold_action_starvation() -> None:
    module = _module()
    starved = module._validation_deployability(
        {
            "fine_tuning_improved_decision_metrics_over_epoch0": True,
            "validation_action_starvation_detected": True,
            "validation_metrics": {
                "predicted_hold_fraction": 1.0,
                "oracle_hold_optimal_fraction": 0.25,
                "hold_aware_decision_accuracy": 0.0,
            },
        }
    )
    assert starved["passed"] is False
    assert starved["validation_action_starvation_detected"] is True

    deployable = module._validation_deployability(
        {
            "fine_tuning_improved_decision_metrics_over_epoch0": True,
            "validation_action_starvation_detected": False,
            "validation_metrics": {
                "predicted_hold_fraction": 0.25,
                "oracle_hold_optimal_fraction": 0.25,
                "hold_aware_decision_accuracy": 0.5,
            },
        }
    )
    assert deployable["passed"] is True


def test_pipeline_rejects_mae_only_fine_tuning() -> None:
    module = _module()
    verdict = module._validation_deployability(
        {
            "fine_tuning_improved_over_epoch0": True,
            "fine_tuning_improved_decision_metrics_over_epoch0": False,
            "validation_metrics": {
                "predicted_hold_fraction": 0.3,
                "oracle_hold_optimal_fraction": 0.3,
                "hold_aware_decision_accuracy": 0.5,
            },
        }
    )
    assert verdict["passed"] is False


def test_pipeline_is_no_swmm_and_keeps_policy_lock_closed() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "compile_direct_tfv_policy_return_dataset_current.py",
        "train_direct_tfv_policy_return_current.py",
        "score_direct_tfv_policy_return_calibration_current.py",
        "calibrate_direct_tfv_policy_return_portfolio_admission_current.py",
    ):
        assert required in text
    assert '"swmm_called_by_pipeline": False' in text
    assert '"ready_for_pi1_development": bool(deployability["passed"])' in text
    assert '"ready_for_policy_lock": False' in text
    assert 'abs(float(args.coverage) - 0.90)' in text
    assert "run_policy_direct_tfv_policy_return_development.py" not in text
