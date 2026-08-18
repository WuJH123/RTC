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
SEQUENCE_SUPPORT = ROOT / "scripts" / "build_direct_tfv_sequence_support_current.py"
FRESH_PREFLIGHT = ROOT / "scripts" / "validate_direct_tfv_fresh_admission_data_current.py"
BASE_ADMISSION = ROOT / "scripts" / "calibrate_direct_tfv_admission_current.py"
POLICY_PANEL = ROOT / "scripts" / "design_direct_tfv_policy_calibration_current.py"
POLICY_ADMISSION = ROOT / "scripts" / "calibrate_direct_tfv_policy_admission_current.py"
STEP3_RUNNER = ROOT / "scripts" / "run_step3_direct_tfv_solver_calibrated_current.py"
DEV_RUNTIME_V6 = ROOT / "scripts" / "run_policy_direct_tfv_development.py"
DEV_RUNTIME_V7 = ROOT / "scripts" / "run_policy_direct_tfv_policy_calibrated_development.py"
DEV_AUDIT = ROOT / "scripts" / "audit_direct_tfv_closed_loop_current.py"
COUNTERFACTUAL_PLAN = ROOT / "scripts" / "plan_direct_tfv_counterfactual_current.py"
RUNTIME_DIAGNOSTIC = ROOT / "scripts" / "diagnose_direct_tfv_runtime_failures_current.py"
BASELINE_RUNNER = ROOT / "scripts" / "run_six_baselines_development_current.py"
BASELINE_COMPARE = ROOT / "scripts" / "compare_direct_tfv_baselines_current.py"
BASELINE_AGGREGATE = ROOT / "scripts" / "aggregate_direct_tfv_baselines_current.py"
CURRENT_POLICY = ROOT / "scripts" / "run_policy_current.py"
CURRENT_SEVEN = ROOT / "scripts" / "run_seven_strategies_current.py"


def _help(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    return result.stdout


def test_current_contract_is_policy_matched_direct_tfv_v9() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V9"
    assert payload["selected_implementation_contract"] == "PROJECT7_DIRECT_109ACT_PAIRWISE_VALUE_TO_DELTA_TFV_V2"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert payload["raw_optimizer_query_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6"
    assert payload["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V7"
    assert payload["admission_contract"] == "PROJECT7_DIRECT_TFV_POLICY_MATCHED_ONE_SIDED_ADMISSION_V2"
    assert payload["policy_panel_contract"] == "PROJECT7_DIRECT_TFV_V6_RAW_OPTIMIZER_CALIBRATION_PANEL_V1"
    assert payload["sequence_support_contract"] == "PROJECT7_DIRECT_TFV_D3_HOLD_JOINT_SEQUENCE_SUPPORT_V1"
    assert payload["policy_admission_data_contract"]["minimum_policy_calibration_rainfall_groups"] == 9
    assert payload["action_contract"]["all_writable_actuators_screened_every_decision"] is True
    assert payload["action_contract"]["score_equals_execute"] is True
    assert payload["scientific_bottleneck"]["classification"] == "ADMISSION_POLICY_MISMATCH_OVERCONSERVATIVE"


def test_execution_registry_routes_policy_panel_before_v7_runtime() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert payload["contract"] == "PROJECT7_EXECUTION_REGISTRY_DIRECT_TFV_V10"
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V9"
    assert current["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert current["raw_optimizer_query_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6"
    assert current["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V7"
    assert current["policy_panel_design"] == "scripts/design_direct_tfv_policy_calibration_current.py"
    assert current["policy_admission_calibration"] == "scripts/calibrate_direct_tfv_policy_admission_current.py"
    assert current["development_authoritative_runtime"] == "scripts/run_policy_direct_tfv_policy_calibrated_development.py"
    assert current["runtime_enabled"] is False
    assert current["validation_enabled"] is False
    assert current["final_enabled"] is False
    assert current["formal_enabled"] is False
    assert current["policy_lock_enabled"] is False


def test_current_step2_wrapper_still_routes_to_frozen_v5_core_runner() -> None:
    text = CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_step2_tfv_value_core_current import main" in text
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v4" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct


def test_current_cli_surfaces_expose_policy_matched_v7() -> None:
    step2 = _help(CURRENT_STEP2)
    for argument in (
        "--profile {smoke,dev}", "--graph", "--cache-manifest", "--d4-fit-cache",
        "--d4-audit-cache", "--causal-store", "--causal-state-store", "--out-dir", "--control-epochs",
    ):
        assert argument in step2
    sequence = _help(SEQUENCE_SUPPORT)
    for argument in ("--checkpoint", "--graph", "--cache-manifest", "--out"):
        assert argument in sequence
    preflight = _help(FRESH_PREFLIGHT)
    for argument in (
        "--base-cache-manifest", "--fresh-calibration-cache-manifest",
        "--optimizer-replay-report", "--coverage", "--reserved-event-id", "--out",
    ):
        assert argument in preflight
    base_calibration = _help(BASE_ADMISSION)
    for argument in (
        "--checkpoint", "--optimizer-replay-report", "--cache-manifest", "--causal-store",
        "--causal-state-store", "--fresh-calibration-cache-manifest",
        "--fresh-calibration-causal-store", "--fresh-calibration-causal-state-store",
        "--reserved-event-id", "--coverage", "--out",
    ):
        assert argument in base_calibration
    panel = _help(POLICY_PANEL)
    for argument in (
        "--checkpoint", "--base-admission", "--sequence-support", "--fresh-cache-manifest",
        "--fresh-causal-store", "--fresh-causal-state-store", "--template-d3-manifest", "--out",
    ):
        assert argument in panel
    policy_calibration = _help(POLICY_ADMISSION)
    for argument in (
        "--base-admission", "--policy-cache-manifest", "--policy-design-manifest", "--coverage", "--out",
    ):
        assert argument in policy_calibration
    step3 = _help(STEP3_RUNNER)
    for argument in (
        "--checkpoint", "--admission-calibration", "--sequence-support",
        "--active-support-quantile", "--max-groups",
    ):
        assert argument in step3
    runtime_v6 = _help(DEV_RUNTIME_V6)
    assert "--admission-calibration" in runtime_v6
    runtime_v7 = _help(DEV_RUNTIME_V7)
    for argument in (
        "--inp", "--step1", "--step2", "--policy-admission-calibration", "--sequence-support",
        "--sensors", "--active-support-quantile",
    ):
        assert argument in runtime_v7
    audit = _help(DEV_AUDIT)
    assert "--metadata" in audit and "--baseline-node-statistics" in audit and "--baseline-metadata" in audit
    plan = _help(COUNTERFACTUAL_PLAN)
    assert "--metadata" in plan and "--max-decisions" in plan and "--out" in plan
    failure = _help(RUNTIME_DIAGNOSTIC)
    assert "--metadata" in failure and "--out" in failure
    baseline = _help(BASELINE_RUNNER)
    assert "--inp" in baseline and "--event-id" in baseline and "--out-dir" in baseline
    compare = _help(BASELINE_COMPARE)
    assert "--proposed-metadata" in compare and "--baseline-panel" in compare
    aggregate = _help(BASELINE_AGGREGATE)
    assert "--comparison" in aggregate and "--out-json" in aggregate


def test_current_lint_surface_tracks_policy_matched_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V12"
    paths = set(payload["paths"])
    required = {
        "scripts/design_direct_tfv_policy_calibration_current.py",
        "scripts/calibrate_direct_tfv_policy_admission_current.py",
        "scripts/run_policy_direct_tfv_policy_calibrated_development.py",
        "src/rtc/direct_tfv_policy_admission.py",
        "src/rtc/step3_tfv_value_mpc_v7.py",
        "tests/test_direct_tfv_policy_admission.py",
        "src/rtc/baselines.py",
        "src/rtc/rule_baselines.py",
    }
    assert required <= paths
    assert all((ROOT / path).is_file() for path in paths)


def test_runtime_and_seven_strategy_remain_fail_closed() -> None:
    for path in (CURRENT_POLICY, CURRENT_SEVEN):
        help_text = _help(path)
        assert "--promotion-status" in help_text
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode != 0
        assert "production" in result.stderr.lower()
