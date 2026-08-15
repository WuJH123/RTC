from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "configs" / "step2_current_contract.json"
GUIDE = ROOT / "CODEX_START_HERE.md"
REGISTRY = ROOT / "configs" / "project7_execution_registry.json"
V128_EXECUTION = ROOT / "configs" / "v128_control_execution.json"
PYPROJECT = ROOT / "pyproject.toml"
PROFILE_RUNNER = ROOT / "scripts" / "run_step2_v128_current_profiles.py"
OBSOLETE_V128_RUNNER = ROOT / "scripts" / "run_step2_v128_control_4060.py"
V128_SEVEN = ROOT / "scripts" / "run_seven_strategies_v128.py"
CURRENT_STEP2 = ROOT / "scripts" / "run_step2_current.py"
CURRENT_POLICY = ROOT / "scripts" / "run_policy_current.py"
CURRENT_SEVEN = ROOT / "scripts" / "run_seven_strategies_current.py"
STEP1_ATTENTION_TRAINER = ROOT / "scripts" / "train_step1_global_attention_dev.py"
STEP1_ATTENTION_AUDIT = ROOT / "scripts" / "audit_step1_global_attention_current.py"
STEP2_SPATIAL_AUDIT = ROOT / "scripts" / "audit_step2_spatial_current.py"
STEP2_DEV_GRADIENT_AUDIT = ROOT / "scripts" / "audit_step2_gradient_current_dev.py"
EDGE_DEV = ROOT / "scripts" / "run_step2_edge_aware_dev.py"
EDGE_SPATIAL = ROOT / "scripts" / "audit_step2_edge_spatial_current.py"
OBSOLETE_OBJECTIVE = ROOT / "src" / "rtc" / "step2_train_v128.py"
VERSIONED_START_GUIDES = (
    ROOT / "CODEX_START_HERE_V069.md",
    ROOT / "CODEX_START_HERE_V127.md",
    ROOT / "CODEX_START_HERE_V128.md",
)
OBSOLETE_ROOT_PIPELINES = (
    ROOT / "FORMAL_PIPELINE_LATEST.md",
    ROOT / "FORMAL_PIPELINE_V2.md",
)


def _script_help(path: Path) -> str:
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


def test_current_contract_routes_only_user_entrypoints_to_unversioned_surface() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["status"] == "CURRENT_DEVELOPMENT_IMPLEMENTATION_NOT_POLICY_LOCKED"
    entrypoints = payload["canonical_entrypoints"]
    assert entrypoints["guide"] == "CODEX_START_HERE.md"
    assert entrypoints["preflight"] == "rtc-current-preflight"
    assert entrypoints["existing_data_training"] == "scripts/run_step2_current.py"
    assert entrypoints["step2_spatial_audit"] == "scripts/audit_step2_spatial_current.py"
    assert entrypoints["step2_development_gradient_audit"] == "scripts/audit_step2_gradient_current_dev.py"
    assert entrypoints["step1_attention_trainer"] == "scripts/train_step1_global_attention_dev.py"
    assert entrypoints["step1_attention_ablation"] == "scripts/audit_step1_global_attention_current.py"
    assert entrypoints["runtime"] == "scripts/run_policy_current.py"
    assert entrypoints["seven_strategy_comparison"] == "scripts/run_seven_strategies_current.py"
    assert payload["step2_current"]["objective_module"] == "src/rtc/step2_train_v128_exact.py"
    assert payload["step2_current"]["execution_profiles"] == ["smoke", "dev", "full"]


def test_project7_registry_has_one_current_user_surface_and_complete_dev_diagnostics() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert current["guide"] == "CODEX_START_HERE.md"
    assert current["preflight"] == "rtc-current-preflight"
    assert current["step2_training"] == "scripts/run_step2_current.py"
    assert current["runtime"] == "scripts/run_policy_current.py"
    assert current["seven_strategy"] == "scripts/run_seven_strategies_current.py"
    assert current["status"] == "CURRENT_DEVELOPMENT_IMPLEMENTATION_NOT_POLICY_LOCKED"
    diagnostics = payload["development_diagnostics"]
    assert diagnostics["step1_attention_trainer"] == "scripts/train_step1_global_attention_dev.py"
    assert diagnostics["step1_distance_attention_ablation"] == "scripts/audit_step1_global_attention_current.py"
    assert diagnostics["step2_spatial_action_effect"] == "scripts/audit_step2_spatial_current.py"
    assert diagnostics["step2_development_gradient"] == "scripts/audit_step2_gradient_current_dev.py"


def test_v128_execution_config_distinguishes_dev_and_full_gradient_surfaces() -> None:
    payload = json.loads(V128_EXECUTION.read_text(encoding="utf-8"))
    entrypoints = payload["entrypoints"]
    assert entrypoints["development_gradient"] == "scripts/audit_step2_gradient_current_dev.py"
    assert entrypoints["full_d2"] == "scripts/audit_step2_v128_d2_gradients_fast.py"
    assert entrypoints["step1_attention_training"] == "scripts/train_step1_global_attention_dev.py"
    assert entrypoints["step1_attention_audit"] == "scripts/audit_step1_global_attention_current.py"


def test_current_preflight_alias_is_installed() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert 'rtc-current-preflight = "rtc.v128_preflight:main"' in text


def test_current_wrappers_pin_the_selected_v128_implementation() -> None:
    assert "run_step2_v128_current_profiles import main" in CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_policy_v128 import main" in CURRENT_POLICY.read_text(encoding="utf-8")
    assert "run_seven_strategies_v128 import main" in CURRENT_SEVEN.read_text(encoding="utf-8")


def test_current_step2_help_requires_explicit_cost_profile() -> None:
    help_text = _script_help(CURRENT_STEP2)
    assert "--profile {smoke,dev,full}" in help_text
    assert "--resume-from" in help_text
    assert "--stop-after-stage" in help_text
    assert "--profile-one-group" in help_text
    assert "--torch-profiler" in help_text
    assert "--out-dir" in help_text
    assert "--cache-manifest" in help_text
    assert "--causal-state-store" in help_text


def test_current_development_diagnostic_clis_have_help() -> None:
    for path in (
        STEP1_ATTENTION_TRAINER,
        STEP1_ATTENTION_AUDIT,
        STEP2_SPATIAL_AUDIT,
        STEP2_DEV_GRADIENT_AUDIT,
        EDGE_DEV,
        EDGE_SPATIAL,
    ):
        _script_help(path)


def test_current_profile_runner_uses_typed_stage_a_exact_objective_and_nonfinal_stages() -> None:
    text = PROFILE_RUNNER.read_text(encoding="utf-8")
    assert "build_v128_model_from_graph" in text
    assert "train_hydraulic_stage_streaming_v128" in text
    assert "train_objective_stage_streaming_v128" in text
    assert "save_stage_checkpoint_v128" in text
    assert '"--profile"' in text
    assert 'choices=("smoke", "dev", "full")' in text
    assert "required=True" in text
    assert "if profile.final_checkpoint_allowed:" in text


def test_current_seven_strategy_help_exposes_current_evidence_contract() -> None:
    help_text = _script_help(CURRENT_SEVEN)
    assert "--continuous-evidence" in help_text
    assert "--continuous-gate" not in help_text
    assert "--engineering-envelope" in help_text


def test_v128_seven_strategy_translates_current_evidence_only_inside_shared_boundary() -> None:
    text = V128_SEVEN.read_text(encoding="utf-8")
    assert 'p.add_argument("--continuous-evidence", required=True)' in text
    assert '"--continuous-gate", str(args.continuous_evidence)' in text
    assert "_build_current_parser().parse_args()" in text


def test_obsolete_current_surfaces_are_removed() -> None:
    assert GUIDE.is_file()
    assert not OBSOLETE_OBJECTIVE.exists()
    assert not OBSOLETE_V128_RUNNER.exists()
    assert all(not path.exists() for path in VERSIONED_START_GUIDES)
    assert all(not path.exists() for path in OBSOLETE_ROOT_PIPELINES)
