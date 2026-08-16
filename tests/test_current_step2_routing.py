from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "configs" / "step2_current_contract.json"
REGISTRY = ROOT / "configs" / "project7_execution_registry.json"
LINT = ROOT / "configs" / "project7_current_lint_surface.json"
GUIDE = ROOT / "CODEX_START_HERE.md"
CURRENT_STEP2 = ROOT / "scripts" / "run_step2_current.py"
DIRECT_RUNNER = ROOT / "scripts" / "run_step2_tfv_value_current.py"
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


def test_current_contract_is_direct_tfv_first() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V1"
    assert payload["status"] == "CURRENT_DEVELOPMENT_IMPLEMENTATION_NOT_POLICY_LOCKED"
    assert payload["selected_implementation_contract"] == "PROJECT7_DIRECT_109ACT_ACTION_TO_DELTA_TFV_VALUE_V1"
    assert payload["core_pipeline"]["step1"].startswith("reconstruct the current")
    assert "delta TFV" in payload["core_pipeline"]["step2"]
    assert "minimize Step2 predicted delta TFV" in payload["core_pipeline"]["step3"]
    assert payload["step2_current"]["hydraulic_trajectory_primary_target"] is False
    assert payload["step2_current"]["gradient_label_used"] is False
    assert payload["step2_current"]["zero_action_contract"].endswith("exactly zero")
    assert payload["scientific_boundaries"]["D4_AUDIT_training"] is False
    assert payload["scientific_boundaries"]["runtime_current_enabled"] is False


def test_execution_registry_routes_current_training_to_direct_value_surface() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert current["research_contract"] == "PROJECT7_CURRENT_DIRECT_TFV_CONTROL_V1"
    assert current["implementation_contract"] == "PROJECT7_DIRECT_109ACT_ACTION_TO_DELTA_TFV_VALUE_V1"
    assert current["step2_training"] == "scripts/run_step2_current.py"
    assert current["enabled_step2_profiles"] == ["smoke", "dev"]
    assert current["runtime_enabled"] is False
    selected = payload["selected_internal_implementation"]
    assert selected["step2_value_model"] == "src/rtc/step2_tfv_value.py"
    assert selected["step2_training"] == "src/rtc/step2_tfv_value_training.py"
    assert selected["step3_development"] == "src/rtc/step3_tfv_value_mpc.py"
    assert payload["legacy_v128"]["status"].startswith("retained")


def test_current_step2_wrapper_routes_to_direct_runner() -> None:
    text = CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_step2_tfv_value_current import main" in text
    assert "run_step2_action_identifiable_current" not in text
    direct = DIRECT_RUNNER.read_text(encoding="utf-8")
    assert "train_direct_tfv_value_model" in direct
    assert "evaluate_direct_tfv_value_model" in direct
    assert "load_causal_state_store_v127" in direct
    assert "load_causal_forecast_store_v123" in direct


def test_current_step2_help_is_small_and_does_not_require_edge_physics() -> None:
    help_text = _help(CURRENT_STEP2)
    assert "--profile {smoke,dev}" in help_text
    assert "--graph" in help_text
    assert "--cache-manifest" in help_text
    assert "--d4-fit-cache" in help_text
    assert "--d4-audit-cache" in help_text
    assert "--causal-store" in help_text
    assert "--causal-state-store" in help_text
    assert "--out-dir" in help_text
    assert "--edge-physics" not in help_text
    assert "--resume-from" not in help_text
    assert "--stop-after-stage" not in help_text


def test_current_lint_surface_tracks_direct_path_not_legacy_v128_curriculum() -> None:
    payload = json.loads(LINT.read_text(encoding="utf-8"))
    assert payload["contract"] == "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V1"
    assert payload["rule_select"] == ["E4", "E7", "E9", "F"]
    paths = set(payload["paths"])
    required = {
        "scripts/run_step2_current.py",
        "scripts/run_step2_tfv_value_current.py",
        "scripts/audit_facility_tfv_influence_current.py",
        "src/rtc/step2_tfv_value.py",
        "src/rtc/step2_tfv_value_training.py",
        "src/rtc/step3_tfv_value_mpc.py",
        "tests/test_direct_tfv_value.py",
    }
    assert required <= paths
    assert "src/rtc/step2_counterfactual_training_v5.py" not in paths
    assert all((ROOT / path).is_file() for path in paths)


def test_guide_states_the_three_core_steps_and_direct_smoke_command() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert "Step1 reconstruct CURRENT full-network hydraulic state" in text
    assert "Step2 learn 109-facility ACTION -> future delta TFV" in text
    assert "Step3 choose lower-TFV 109-facility action sequence every 10 min" in text
    assert "--profile smoke" in text
    assert "--edge-physics" not in text
    assert "STEP2_DIRECT_TFV_VALUE_REPORT.json" in text


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
