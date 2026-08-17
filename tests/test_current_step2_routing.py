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
ADMISSION = ROOT / "scripts" / "calibrate_direct_tfv_admission_current.py"
STEP3_RUNNER = ROOT / "scripts" / "run_step3_direct_tfv_solver_calibrated_current.py"
DEV_RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_development.py"
DEV_AUDIT = ROOT / "scripts" / "audit_direct_tfv_calibrated_closed_loop_current.py"
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


def test_current_contract_is_calibrated_direct_tfv_v6() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V6"
    assert payload["selected_implementation_contract"] == "PROJECT7_DIRECT_109ACT_PAIRWISE_VALUE_TO_DELTA_TFV_V2"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert payload["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V5"
    assert payload["admission_contract"] == "PROJECT7_DIRECT_TFV_OPTIMIZER_AWARE_ONE_SIDED_ADMISSION_V1"
    assert "upper bound < 0" in payload["step3_current"]["admission"]
    assert payload["action_contract"]["all_writable_actuators_screened_every_decision"] is True
    assert payload["scientific_bottleneck"]["classification"] == "DEVELOPMENT_NO_CONTROL_BENEFIT_INCONSISTENT"


def test_execution_registry_routes_calibration_before_step3() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V6"
    assert current["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert current["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V5"
    assert current["admission_contract"] == "PROJECT7_DIRECT_TFV_OPTIMIZER_AWARE_ONE_SIDED_ADMISSION_V1"
    assert current["admission_calibration"] == "scripts/calibrate_direct_tfv_admission_current.py"
    assert current["runtime_enabled"] is False
    assert current["validation_enabled"] is False
    assert current["final_enabled"] is False


def test_current_step2_wrapper_still_routes_to_v5_core_runner() -> None:
    text = CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_step2_tfv_value_core_current import main" in text
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v4" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct


def test_current_cli_surfaces_expose_optimizer_aware_admission() -> None:
    step2 = _help(CURRENT_STEP2)
    for argument in (
        "--profile {smoke,dev}", "--graph", "--cache-manifest", "--d4-fit-cache",
        "--d4-audit-cache", "--causal-store", "--causal-state-store", "--out-dir", "--control-epochs",
    ):
        assert argument in step2
    calibration = _help(ADMISSION)
    for argument in (
        "--checkpoint", "--optimizer-replay-report", "--cache-manifest", "--causal-store",
        "--causal-state-store", "--coverage", "--out",
    ):
        assert argument in calibration
    step3 = _help(STEP3_RUNNER)
    for argument in ("--checkpoint", "--admission-calibration", "--active-support-quantile", "--max-groups"):
        assert argument in step3
    runtime = _help(DEV_RUNTIME)
    for argument in ("--inp", "--step1", "--step2", "--admission-calibration", "--sensors", "--active-support-quantile"):
        assert argument in runtime
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


def test_current_lint_surface_tracks_calibrated_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V9"
    paths = set(payload["paths"])
    required = {
        "scripts/calibrate_direct_tfv_admission_current.py",
        "scripts/run_step3_direct_tfv_solver_calibrated_current.py",
        "scripts/run_policy_direct_tfv_development.py",
        "scripts/audit_direct_tfv_calibrated_closed_loop_current.py",
        "src/rtc/direct_tfv_admission.py",
        "src/rtc/controller_direct_tfv.py",
        "src/rtc/step3_tfv_value_mpc_v5.py",
        "tests/test_direct_tfv_calibrated_admission.py",
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
