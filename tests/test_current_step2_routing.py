from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "configs" / "step2_current_contract.json"
REGISTRY = ROOT / "configs" / "project7_execution_registry.json"
LINT = ROOT / "configs" / "project7_current_lint_surface.json"
CURRENT_STEP2 = ROOT / "scripts" / "run_step2_current.py"
DIRECT_RUNNER = ROOT / "scripts" / "run_step2_tfv_value_core_current.py"
CONTEXT = ROOT / "scripts" / "build_direct_tfv_first_move_context_current.py"
FIRST_MOVE_PANEL = ROOT / "scripts" / "design_direct_tfv_first_move_calibration_current.py"
FIRST_MOVE_ADMISSION = ROOT / "scripts" / "calibrate_direct_tfv_first_move_admission_current.py"
V11_RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_first_move_development.py"
V12_RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_robust_rainfall_development.py"
POLICY_RETURN_PAIR = ROOT / "scripts" / "run_direct_tfv_policy_return_pair_current.py"
BASE_HYBRID_PARENT = ROOT / "scripts" / "run_policy_direct_tfv_base_hybrid_parent_current.py"
POLICY_RETURN_CAPTURE = ROOT / "scripts" / "capture_direct_tfv_policy_return_context_current.py"
POLICY_RETURN_DESIGN = ROOT / "scripts" / "design_direct_tfv_policy_return_portfolio_current.py"
POLICY_RETURN_QUERY = ROOT / "scripts" / "run_direct_tfv_policy_return_query_current.py"
POLICY_RETURN_TRAIN = ROOT / "scripts" / "train_direct_tfv_policy_return_current.py"
POLICY_RETURN_RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_policy_return_development.py"
CURRENT_POLICY = ROOT / "scripts" / "run_policy_current.py"
CURRENT_SEVEN = ROOT / "scripts" / "run_seven_strategies_current.py"


def _help(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    return result.stdout


def test_archival_step2_contract_remains_frozen_history_not_current_router() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V11"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert payload["objective_contract"]["online_primary_objective"] == (
        "SYSTEM_WIDE_CUMULATIVE_TFV_MINIMIZATION"
    )
    assert payload["action_contract"]["hold_online_reference_semantics"] == (
        "LATCH_PREVIOUS_SUPERVISORY_TARGET_NOT_NO_CONTROL_NOT_RESET"
    )
    assert payload["time_contract"]["cross_decision_target_semantics"] == (
        "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED"
    )
    assert payload["scientific_bottleneck"]["classification"] == (
        "OPEN_LOOP_VALUE_VS_RECEDING_CONTROL_MISMATCH"
    )
    assert payload["v12_evidence"]["accepted_h360_replay_sign_correct"] == "9/9"
    assert payload["v12_evidence"]["event_level_no_control_consistency"] == "1/3"
    assert payload["next_scientific_stage"]["policy_iteration_required"] is True


def test_execution_registry_routes_current_hybrid_policy_return_workflow() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert payload["contract"] == (
        "PROJECT7_EXECUTION_REGISTRY_PRACTICAL_RTC_H10_HYBRID_POLICY_RETURN_V2"
    )
    assert current["research_contract"] == "PROJECT7_PRACTICAL_RTC_V14"
    assert current["base_step2_training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert current["first_round_parent_policy"] == (
        "PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V2"
    )
    assert current["online_step3_contract"] == (
        "PROJECT7_PRACTICAL_RTC_H10_POLICY_RETURN_HYBRID_GRADIENT_V13"
    )
    assert current["online_candidate_portfolio_contract"].endswith(
        "V4_H10_HYBRID_GRADIENT"
    )
    assert current["projected_gradient_dimension"] == 109
    assert current["projected_gradient_action_horizon"] == "H10_ONLY"
    assert current["base_parent_runtime"] == (
        "scripts/run_policy_direct_tfv_base_hybrid_parent_current.py"
    )
    assert current["policy_return_context_capture"] == (
        "scripts/capture_direct_tfv_policy_return_context_current.py"
    )
    assert current["policy_return_portfolio_design"] == (
        "scripts/design_direct_tfv_policy_return_portfolio_current.py"
    )
    assert current["policy_return_query_truth"] == (
        "scripts/run_direct_tfv_policy_return_query_current.py"
    )
    assert current["policy_return_train"] == "scripts/train_direct_tfv_policy_return_current.py"
    assert current["policy_return_development_runtime"] == (
        "scripts/run_policy_direct_tfv_policy_return_development.py"
    )
    assert current["runtime_enabled"] is False
    assert current["validation_enabled"] is False
    assert current["final_enabled"] is False
    assert current["formal_enabled"] is False
    assert current["policy_lock_enabled"] is False
    assert any("same raw causal prefix" in rule for rule in payload["hard_rules"])
    assert any("projected-gradient proposer" in rule for rule in payload["hard_rules"])
    assert any("near-all-HOLD" in rule for rule in payload["hard_rules"])


def test_step2_base_remains_frozen_v5() -> None:
    assert "run_step2_tfv_value_core_current import main" in CURRENT_STEP2.read_text(encoding="utf-8")
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v4" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct


def test_archival_and_current_policy_return_clis_are_explicitly_separated() -> None:
    context = _help(CONTEXT)
    assert "--run-index" in context and "--event-registry" in context
    panel = _help(FIRST_MOVE_PANEL)
    assert "--context-store" in panel
    calibration = _help(FIRST_MOVE_ADMISSION)
    assert "--first-move-run-dir" in calibration
    v11 = _help(V11_RUNTIME)
    assert "--first-move-admission-calibration" in v11
    v12 = _help(V12_RUNTIME)
    assert "--first-move-admission" in v12
    pair = _help(POLICY_RETURN_PAIR)
    assert "--continuation-kind" in pair

    parent = _help(BASE_HYBRID_PARENT)
    assert "--projected-gradient-steps" in parent
    capture = _help(POLICY_RETURN_CAPTURE)
    assert "--continuation-kind" in capture
    design = _help(POLICY_RETURN_DESIGN)
    assert "--projected-gradient-steps" in design
    query = _help(POLICY_RETURN_QUERY)
    assert "--candidate-manifest" in query
    assert "--continuation-kind" in query
    assert "--projected-gradient-steps" in query
    train = _help(POLICY_RETURN_TRAIN)
    assert "--train-dataset" in train and "--validation-dataset" in train
    runtime = _help(POLICY_RETURN_RUNTIME)
    assert "--policy-return-checkpoint" in runtime
    assert "--policy-return-admission" in runtime
    assert "--projected-gradient-steps" in runtime


def test_current_lint_surface_tracks_complete_hybrid_policy_return_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_PRACTICAL_RTC_V1"
    paths = set(payload["paths"])
    required = {
        "src/rtc/direct_tfv_policy_return.py",
        "src/rtc/direct_tfv_policy_return_hybrid_portfolio.py",
        "src/rtc/direct_tfv_policy_return_portfolio_admission.py",
        "src/rtc/direct_tfv_base_probe_runtime_factory.py",
        "src/rtc/direct_tfv_policy_return_runtime_factory.py",
        "src/rtc/step3_tfv_base_probe_parent_v2.py",
        "src/rtc/step3_tfv_value_mpc_v13.py",
        "scripts/run_policy_direct_tfv_base_hybrid_parent_current.py",
        "scripts/capture_direct_tfv_policy_return_context_current.py",
        "scripts/design_direct_tfv_policy_return_portfolio_current.py",
        "scripts/run_direct_tfv_policy_return_query_current.py",
        "scripts/compile_direct_tfv_policy_return_dataset_current.py",
        "scripts/train_direct_tfv_policy_return_current.py",
        "scripts/run_policy_direct_tfv_policy_return_development.py",
        "tests/test_direct_tfv_hybrid_gradient_portfolio.py",
        "tests/test_direct_tfv_policy_return_portfolio.py",
    }
    assert required <= paths
    assert all((ROOT / path).is_file() for path in paths)


def test_production_and_untouched_evaluation_remain_fail_closed() -> None:
    for path in (CURRENT_POLICY, CURRENT_SEVEN):
        help_text = _help(path)
        assert "--promotion-status" in help_text
        result = subprocess.run(
            [sys.executable, str(path)], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode != 0
        assert "production" in result.stderr.lower()
