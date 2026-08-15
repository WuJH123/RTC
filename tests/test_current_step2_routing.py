from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "configs" / "step2_current_contract.json"
GUIDE = ROOT / "CODEX_START_HERE.md"
REGISTRY = ROOT / "configs" / "project7_execution_registry.json"
V128_RUNNER = ROOT / "scripts" / "run_step2_v128_control_4060.py"
CURRENT_STEP2 = ROOT / "scripts" / "run_step2_current.py"
CURRENT_POLICY = ROOT / "scripts" / "run_policy_current.py"
CURRENT_SEVEN = ROOT / "scripts" / "run_seven_strategies_current.py"
OBSOLETE_OBJECTIVE = ROOT / "src" / "rtc" / "step2_train_v128.py"
VERSIONED_START_GUIDES = (
    ROOT / "CODEX_START_HERE_V069.md",
    ROOT / "CODEX_START_HERE_V127.md",
    ROOT / "CODEX_START_HERE_V128.md",
)


def test_current_contract_routes_only_user_entrypoints_to_unversioned_surface() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert payload["status"] == "CURRENT_DEVELOPMENT_IMPLEMENTATION_NOT_POLICY_LOCKED"
    entrypoints = payload["canonical_entrypoints"]
    assert entrypoints["guide"] == "CODEX_START_HERE.md"
    assert entrypoints["existing_data_training"] == "scripts/run_step2_current.py"
    assert entrypoints["runtime"] == "scripts/run_policy_current.py"
    assert entrypoints["seven_strategy_comparison"] == "scripts/run_seven_strategies_current.py"
    assert payload["step2_current"]["objective_module"] == "src/rtc/step2_train_v128_exact.py"


def test_project7_registry_has_one_current_user_surface() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = payload["current"]
    assert payload["contract"] == "PROJECT7_EXECUTION_REGISTRY_V7_SINGLE_CURRENT_SURFACE"
    assert current["guide"] == "CODEX_START_HERE.md"
    assert current["step2_training"] == "scripts/run_step2_current.py"
    assert current["runtime"] == "scripts/run_policy_current.py"
    assert current["seven_strategy"] == "scripts/run_seven_strategies_current.py"
    assert current["status"] == "CURRENT_DEVELOPMENT_IMPLEMENTATION_NOT_POLICY_LOCKED"


def test_current_wrappers_pin_the_selected_v128_implementation() -> None:
    assert "run_step2_v128_control_4060 import main" in CURRENT_STEP2.read_text(encoding="utf-8")
    assert "run_policy_v128 import main" in CURRENT_POLICY.read_text(encoding="utf-8")
    assert "run_seven_strategies_v128 import main" in CURRENT_SEVEN.read_text(encoding="utf-8")


def test_v128_runner_uses_typed_stage_a_and_exact_pairwise_objective() -> None:
    text = V128_RUNNER.read_text(encoding="utf-8")
    assert "from rtc.step2_train_v128_exact import" in text
    assert "from rtc.step2_train_v128 import" not in text
    assert "train_hydraulic_stage_streaming_v128" in text
    assert "runner.train_hydraulic_stage_streaming_v127 = train_hydraulic_stage_streaming_v128" in text
    assert "train_objective_stage_streaming_v128" in text
    assert "runner.train_objective_stage_streaming_v127 = train_objective_stage_streaming_v128" in text
    assert "exact_two_pass_full_pairwise_first_order_gradient" in text


def test_obsolete_current_surfaces_are_removed() -> None:
    assert GUIDE.is_file()
    assert not OBSOLETE_OBJECTIVE.exists()
    assert all(not path.exists() for path in VERSIONED_START_GUIDES)
