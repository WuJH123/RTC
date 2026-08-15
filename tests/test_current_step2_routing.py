from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "configs" / "step2_current_contract.json"
GUIDE = ROOT / "CODEX_START_HERE_V127.md"
REGISTRY = ROOT / "configs" / "project7_execution_registry.json"
V128_RUNNER = ROOT / "scripts" / "run_step2_v128_control_4060.py"


def test_current_contract_routes_to_control_streaming_trainer() -> None:
    payload = json.loads(CURRENT.read_text(encoding="utf-8"))
    entrypoints = payload["canonical_entrypoints"]
    assert entrypoints["existing_data_training"] == (
        "scripts/run_step2_v127_control_streaming.py"
    )
    assert entrypoints["historical_noncanonical_training"] == (
        "scripts/run_step2_v127.py"
    )


def test_canonical_guide_does_not_instruct_historical_base_trainer() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert "python scripts/run_step2_v127_control_streaming.py" in text
    assert "python scripts/run_step2_v127.py `" not in text
    assert "scripts/run_step2_v127.py` is a preserved historical implementation" in text


def test_project7_registry_separates_production_v127_from_experimental_v128() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    production = payload["production"]
    experimental = payload["experimental_candidate"]
    assert production["base_training"] == "scripts/run_step2_v127_control_streaming.py"
    assert production["historical_trainer_forbidden_as_current"] == "scripts/run_step2_v127.py"
    assert experimental["base_training"] == "scripts/run_step2_v128_control_4060.py"
    assert experimental["runtime"] == "scripts/run_policy_v128.py"
    assert experimental["status"] == "DEVELOPMENT_ONLY_NOT_POLICY_LOCKED"
    assert production["project_contract"] != experimental["project_contract"]


def test_v128_runner_versions_both_teacher_forcing_and_objective() -> None:
    text = V128_RUNNER.read_text(encoding="utf-8")
    assert "train_hydraulic_stage_streaming_v128" in text
    assert "runner.train_hydraulic_stage_streaming_v127 = train_hydraulic_stage_streaming_v128" in text
    assert "train_objective_stage_streaming_v128" in text
    assert "runner.train_objective_stage_streaming_v127 = train_objective_stage_streaming_v128" in text
