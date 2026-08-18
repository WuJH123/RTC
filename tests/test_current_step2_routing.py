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
FIRST_MOVE_PANEL = ROOT / "scripts" / "design_direct_tfv_first_move_calibration_current.py"
FIRST_MOVE_ADMISSION = ROOT / "scripts" / "calibrate_direct_tfv_first_move_admission_current.py"
DEV_RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_first_move_development.py"
PFV_REPORT = ROOT / "scripts" / "add_pfv_to_direct_tfv_comparison_current.py"
CURRENT_POLICY = ROOT / "scripts" / "run_policy_current.py"
CURRENT_SEVEN = ROOT / "scripts" / "run_seven_strategies_current.py"


def _help(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    return result.stdout


def test_current_contract_is_refined_first_move_v11() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V11"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert payload["raw_optimizer_query_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6"
    assert payload["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V9"
    assert payload["first_move_admission_contract"] == (
        "PROJECT7_DIRECT_TFV_REFINED_FIRST_MOVE_NORMALIZED_ONE_SIDED_ADMISSION_V1"
    )
    assert payload["first_move_panel_contract"] == (
        "PROJECT7_DIRECT_TFV_REFINED_FIRST_MOVE_CALIBRATION_PANEL_V1"
    )
    assert payload["first_move_data_contract"]["minimum_calibration_rainfall_groups"] == 24
    assert payload["first_move_data_contract"]["normalization"] == (
        "SQRT_FIRST_MOVE_CHANGED_FACILITY_COUNT"
    )
    assert payload["objective_contract"]["online_primary_objective"] == (
        "SYSTEM_WIDE_CUMULATIVE_TFV_MINIMIZATION"
    )
    assert payload["objective_contract"]["pfv_role"] == "REPORT_ONLY_SECONDARY_RISK_METRIC"
    assert payload["objective_contract"]["pfv_enters_step3_objective"] is False
    assert payload["action_contract"]["hold_online_reference_semantics"] == (
        "HOLD_ACTIVE_SUPERVISORY_TARGET_NOT_NO_CONTROL"
    )
    assert payload["scientific_bottleneck"]["classification"] == (
        "FIRST_MOVE_QUERY_AND_ADMISSION_MISMATCH_AFTER_V10"
    )


def test_execution_registry_routes_first_move_workflow() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert payload["contract"] == "PROJECT7_EXECUTION_REGISTRY_DIRECT_TFV_V12"
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V11"
    assert current["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
    assert current["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V9"
    assert current["first_move_panel_design"] == (
        "scripts/design_direct_tfv_first_move_calibration_current.py"
    )
    assert current["first_move_admission_calibration"] == (
        "scripts/calibrate_direct_tfv_first_move_admission_current.py"
    )
    assert current["development_authoritative_runtime"] == (
        "scripts/run_policy_direct_tfv_first_move_development.py"
    )
    assert current["pfv_report"] == "scripts/add_pfv_to_direct_tfv_comparison_current.py"
    assert current["runtime_enabled"] is False
    assert current["validation_enabled"] is False
    assert current["final_enabled"] is False
    assert current["formal_enabled"] is False
    assert current["policy_lock_enabled"] is False


def test_step2_remains_frozen_v5() -> None:
    assert "run_step2_tfv_value_core_current import main" in CURRENT_STEP2.read_text(encoding="utf-8")
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v4" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct


def test_current_v11_clis_are_exposed() -> None:
    panel = _help(FIRST_MOVE_PANEL)
    for argument in (
        "--checkpoint",
        "--policy-admission",
        "--sequence-support",
        "--fresh-cache-manifest",
        "--fresh-causal-store",
        "--fresh-causal-state-store",
        "--template-d3-manifest",
        "--first-move-maxiter",
        "--out",
    ):
        assert argument in panel
    calibration = _help(FIRST_MOVE_ADMISSION)
    for argument in (
        "--first-move-cache-manifest",
        "--first-move-design-manifest",
        "--coverage",
        "--step2-checkpoint",
        "--sequence-support",
        "--out",
    ):
        assert argument in calibration
    runtime = _help(DEV_RUNTIME)
    for argument in (
        "--inp",
        "--step1",
        "--step2",
        "--policy-admission-calibration",
        "--first-move-admission-calibration",
        "--sequence-support",
        "--first-move-maxiter",
        "--first-move-deadline-seconds",
    ):
        assert argument in runtime
    pfv = _help(PFV_REPORT)
    assert "--comparison-json" in pfv
    assert "--priority-nodes" in pfv
    assert "--out-json" in pfv


def test_current_lint_surface_tracks_v11_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V14"
    paths = set(payload["paths"])
    required = {
        "src/rtc/direct_tfv_first_move.py",
        "src/rtc/direct_tfv_first_move_admission.py",
        "src/rtc/step3_tfv_value_mpc_v9.py",
        "scripts/design_direct_tfv_first_move_calibration_current.py",
        "scripts/calibrate_direct_tfv_first_move_admission_current.py",
        "scripts/run_policy_direct_tfv_first_move_development.py",
        "scripts/add_pfv_to_direct_tfv_comparison_current.py",
        "tests/test_direct_tfv_first_move.py",
    }
    assert required <= paths
    assert all((ROOT / path).is_file() for path in paths)


def test_production_and_untouched_evaluation_remain_fail_closed() -> None:
    for path in (CURRENT_POLICY, CURRENT_SEVEN):
        help_text = _help(path)
        assert "--promotion-status" in help_text
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode != 0
        assert "production" in result.stderr.lower()
