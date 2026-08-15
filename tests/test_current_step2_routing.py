from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "configs" / "step2_current_contract.json"
GUIDE = ROOT / "CODEX_START_HERE_V127.md"


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
