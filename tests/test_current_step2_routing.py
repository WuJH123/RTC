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


def test_current_contract_preserves_v11_base_and_records_v12_policy_return_stage() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V11"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert payload["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V9"
    assert payload["v12_step3_contract"] == (
        "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V10_CAUSAL_RAINFALL_SCENARIO_MEAN"
    )
    assert payload["next_step3_contract"] == (
        "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V11_POLICY_RETURN_FIRST_ACTION"
    )
    assert payload["objective_contract"]["online_primary_objective"] == (
        "SYSTEM_WIDE_CUMULATIVE_TFV_MINIMIZATION"
    )
    assert payload["objective_contract"]["pfv_role"] == "REPORT_ONLY_SECONDARY_RISK_METRIC"
    assert payload["objective_contract"]["pfv_enters_step3_objective"] is False
    assert payload["action_contract"]["hold_online_reference_semantics"] == (
        "LATCH_PREVIOUS_SUPERVISORY_TARGET_NOT_NO_CONTROL_NOT_RESET"
    )
    assert payload["action_contract"]["unchanged_facility_semantics"] == (
        "COPY_PREVIOUS_COMMANDED_TARGET_EXACTLY"
    )
    assert payload["time_contract"]["cross_decision_target_semantics"] == (
        "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED"
    )
    assert payload["first_move_data_contract"]["context_contract"].startswith("CANDIDATE_FREE")
    assert payload["first_move_data_contract"]["source_fingerprint"].startswith("BEHAVIORAL_")
    assert payload["scientific_bottleneck"]["classification"] == (
        "OPEN_LOOP_VALUE_VS_RECEDING_CONTROL_MISMATCH"
    )
    assert payload["v12_evidence"]["accepted_h360_replay_sign_correct"] == "9/9"
    assert payload["v12_evidence"]["event_level_no_control_consistency"] == "1/3"
    assert payload["next_scientific_stage"]["policy_iteration_required"] is True


def test_execution_registry_routes_policy_return_development_workflow() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert payload["contract"] == "PROJECT7_EXECUTION_REGISTRY_DIRECT_TFV_V14_POLICY_RETURN"
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V11"
    assert current["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert current["first_move_context_build"] == "scripts/build_direct_tfv_first_move_context_current.py"
    assert current["v12_development_runtime"] == (
        "scripts/run_policy_direct_tfv_robust_rainfall_development.py"
    )
    assert current["policy_return_pair_truth"] == (
        "scripts/run_direct_tfv_policy_return_pair_current.py"
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
    assert any("candidate-free first-move context" in rule for rule in payload["hard_rules"])
    assert any("same frozen continuation policy" in rule for rule in payload["hard_rules"])


def test_step2_base_remains_frozen_v5() -> None:
    assert "run_step2_tfv_value_core_current import main" in CURRENT_STEP2.read_text(encoding="utf-8")
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v4" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct


def test_candidate_free_and_policy_return_clis_are_exposed() -> None:
    context = _help(CONTEXT)
    assert "--run-index" in context and "--event-registry" in context
    panel = _help(FIRST_MOVE_PANEL)
    assert "--context-store" in panel
    assert "--fresh-cache-manifest" not in panel
    assert "--template-d3-manifest" not in panel
    calibration = _help(FIRST_MOVE_ADMISSION)
    assert "--first-move-run-dir" in calibration
    assert "--first-move-cache-manifest" not in calibration
    v11 = _help(V11_RUNTIME)
    assert "--first-move-admission-calibration" in v11
    v12 = _help(V12_RUNTIME)
    assert "--first-move-admission" in v12
    pair = _help(POLICY_RETURN_PAIR)
    assert "--continuation-kind" in pair
    assert "--policy-return-checkpoint" in pair
    train = _help(POLICY_RETURN_TRAIN)
    assert "--train-dataset" in train and "--validation-dataset" in train
    runtime = _help(POLICY_RETURN_RUNTIME)
    assert "--policy-return-checkpoint" in runtime
    assert "--policy-return-admission" in runtime


def test_current_lint_surface_tracks_complete_policy_return_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V16"
    paths = set(payload["paths"])
    required = {
        "src/rtc/direct_tfv_first_move_context.py",
        "src/rtc/controller_direct_tfv_safe.py",
        "src/rtc/direct_tfv_v12_lineage.py",
        "src/rtc/direct_tfv_policy_return.py",
        "src/rtc/direct_tfv_policy_return_runtime_factory.py",
        "src/rtc/step3_tfv_value_mpc_v11.py",
        "scripts/build_direct_tfv_first_move_context_current.py",
        "scripts/design_direct_tfv_robust_rainfall_first_move_calibration_current.py",
        "scripts/calibrate_direct_tfv_robust_rainfall_first_move_admission_current.py",
        "scripts/run_direct_tfv_policy_return_pair_current.py",
        "scripts/train_direct_tfv_policy_return_current.py",
        "scripts/run_policy_direct_tfv_policy_return_development.py",
        "tests/test_direct_tfv_policy_return.py",
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
