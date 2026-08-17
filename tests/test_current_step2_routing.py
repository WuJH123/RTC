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
STEP3_RUNNER = ROOT / "scripts" / "run_step3_direct_tfv_solver_current.py"
DEV_RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_development.py"
DEV_AUDIT = ROOT / "scripts" / "audit_direct_tfv_closed_loop_current.py"
CURRENT_POLICY = ROOT / "scripts" / "run_policy_current.py"
CURRENT_SEVEN = ROOT / "scripts" / "run_seven_strategies_current.py"


def _help(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    return result.stdout


def test_current_contract_is_direct_tfv_core_v4() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V4"
    assert payload["selected_implementation_contract"] == "PROJECT7_DIRECT_109ACT_PAIRWISE_VALUE_TO_DELTA_TFV_V2"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V4"
    assert payload["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V3"
    assert payload["step2_current"]["109_facility_learning_gate"].startswith("full DEV fails closed only")
    assert payload["step2_current"]["d4_role"].startswith("reference-shift stress diagnostic")
    assert payload["step2_current"]["top1_policy"].startswith("exact cached-candidate top1 is diagnostic only")
    assert payload["step3_current"]["admission"].startswith("no separate calibrated threshold")
    assert payload["action_contract"]["all_writable_actuators_screened_every_decision"] is True


def test_execution_registry_routes_directly_from_step2_to_step3() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V4"
    assert current["training_contract"] == "PROJECT7_DIRECT_TFV_CORE_TRAINING_V4"
    assert "selection_calibration" not in current
    assert current["step3_solver_audit"] == "scripts/run_step3_direct_tfv_solver_current.py"
    assert current["runtime_enabled"] is False
    selected = payload["selected_internal_implementation"]
    assert selected["step2_training"].startswith("src/rtc/step2_tfv_value_training_v4.py")
    assert selected["step3_development"] == "src/rtc/step3_tfv_value_mpc_v3.py"


def test_current_step2_wrapper_routes_to_core_runner() -> None:
    text = CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_step2_tfv_value_core_current import main" in text
    assert "run_step2_tfv_value_selection_aware_current" not in text
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v4" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct
    assert "control_training_reference_family" in direct


def test_current_cli_surfaces_are_explicit_and_threshold_free() -> None:
    step2 = _help(CURRENT_STEP2)
    for argument in (
        "--profile {smoke,dev}",
        "--graph",
        "--cache-manifest",
        "--d4-fit-cache",
        "--d4-audit-cache",
        "--causal-store",
        "--causal-state-store",
        "--out-dir",
        "--control-epochs",
    ):
        assert argument in step2
    step3 = _help(STEP3_RUNNER)
    assert "--checkpoint" in step3
    assert "--selection-report" not in step3
    assert "--max-groups" in step3
    assert "--active-facilities" in step3
    runtime = _help(DEV_RUNTIME)
    for argument in ("--inp", "--step1", "--step2", "--sensors", "--active-facilities"):
        assert argument in runtime
    assert "--selection-report" not in runtime
    audit = _help(DEV_AUDIT)
    assert "--metadata" in audit
    assert "--baseline-node-statistics" in audit


def test_current_lint_surface_tracks_core_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V5"
    paths = set(payload["paths"])
    required = {
        "scripts/run_step2_tfv_value_core_current.py",
        "scripts/run_step3_direct_tfv_solver_current.py",
        "scripts/run_policy_direct_tfv_development.py",
        "scripts/audit_direct_tfv_closed_loop_current.py",
        "src/rtc/checkpoint_direct_tfv.py",
        "src/rtc/controller_direct_tfv.py",
        "src/rtc/runtime_controller_guard.py",
        "src/rtc/step2_tfv_value_training_v4.py",
        "src/rtc/step2_tfv_support.py",
        "src/rtc/step3_tfv_value_mpc_v3.py",
        "tests/test_direct_tfv_core_training.py",
        "tests/test_direct_tfv_step3_core.py",
        "tests/test_direct_tfv_runtime_adapter.py",
        "tests/test_temporal_control_continuity_v069.py",
    }
    assert required <= paths
    assert all((ROOT / path).is_file() for path in paths)


def test_runtime_and_seven_strategy_remain_fail_closed() -> None:
    for path in (CURRENT_POLICY, CURRENT_SEVEN):
        help_text = _help(path)
        assert "--promotion-status" in help_text
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "production" in result.stderr.lower()
