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
DIRECT_RUNNER = ROOT / "scripts" / "run_step2_tfv_value_selection_aware_current.py"
SELECTION_RUNNER = ROOT / "scripts" / "run_step2_selection_threshold_current.py"
STEP3_RUNNER = ROOT / "scripts" / "run_step3_direct_tfv_solver_current.py"
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


def test_current_contract_is_selection_aware_direct_tfv_v3() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V3"
    assert payload["selected_implementation_contract"] == "PROJECT7_DIRECT_109ACT_PAIRWISE_VALUE_TO_DELTA_TFV_V2"
    assert payload["training_contract"] == "PROJECT7_DIRECT_TFV_SELECTION_AWARE_TRAINING_V3"
    assert payload["selection_contract"] == "PROJECT7_DIRECT_TFV_HOLD_ACTION_THRESHOLD_V2"
    assert payload["step3_contract"] == "PROJECT7_DIRECT_TFV_109ACT_SCREENED_TRUST_REGION_MPC_V2"
    assert payload["step2_current"]["109_facility_learning_gate"].startswith("full DEV fails closed")
    assert payload["step2_current"]["hydraulic_trajectory_primary_target"] is False
    assert payload["step2_current"]["gradient_label_used"] is False
    assert payload["step3_current"]["screening"].startswith("evaluate first-move")
    assert payload["action_contract"]["all_writable_actuators_screened_every_decision"] is True


def test_execution_registry_routes_v3_training_selection_and_step3() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V3"
    assert current["training_contract"] == "PROJECT7_DIRECT_TFV_SELECTION_AWARE_TRAINING_V3"
    assert current["selection_calibration"] == "scripts/run_step2_selection_threshold_current.py"
    assert current["step3_solver_audit"] == "scripts/run_step3_direct_tfv_solver_current.py"
    assert current["runtime_enabled"] is False
    selected = payload["selected_internal_implementation"]
    assert selected["step2_training"] == "src/rtc/step2_tfv_value_training_v3.py explicit HOLD/action + oracle-choice loss"
    assert selected["step3_development"] == "src/rtc/step3_tfv_value_mpc_v2.py"


def test_current_step2_wrapper_routes_to_selection_aware_runner() -> None:
    text = CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_step2_tfv_value_selection_aware_current import main" in text
    assert "run_step2_action_identifiable_current" not in text
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model_v3" in direct
    assert "derive_direct_tfv_action_support" in direct
    assert "single_facility_coverage_count" in direct
    assert "selection_aware_training" in direct


def test_current_cli_surfaces_are_explicit() -> None:
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
        "--selection-epochs",
    ):
        assert argument in step2
    selection = _help(SELECTION_RUNNER)
    assert "--checkpoint" in selection
    assert "--d4-audit-cache" in selection
    assert "--alpha" not in selection
    step3 = _help(STEP3_RUNNER)
    assert "--checkpoint" in step3
    assert "--selection-report" in step3
    assert "--max-groups" in step3


def test_current_lint_surface_tracks_v3_paths() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V3"
    paths = set(payload["paths"])
    required = {
        "scripts/run_step2_tfv_value_selection_aware_current.py",
        "scripts/run_step2_selection_threshold_current.py",
        "scripts/run_step3_direct_tfv_solver_current.py",
        "src/rtc/step2_tfv_value_training_v3.py",
        "src/rtc/step2_tfv_support.py",
        "src/rtc/step2_tfv_selection_v2.py",
        "src/rtc/step3_tfv_value_mpc_v2.py",
        "tests/test_direct_tfv_selection_aware.py",
        "tests/test_direct_tfv_step3_trust_region.py",
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
