from __future__ import annotations

from pathlib import Path

import pytest

from rtc.pipeline_ledger_cli import record_pipeline_stage
from rtc.tfv_pipeline import TFVPipelineLedger


def _evidence(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_pipeline_ledger_records_ordered_hashed_stages(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    first = _evidence(tmp_path / "preflight.json", "preflight")
    second = _evidence(tmp_path / "split.json", "split")

    record_pipeline_stage(
        ledger_path=ledger,
        stage="inp_preflight",
        evidence_paths=[first],
        passed=True,
    )
    record_pipeline_stage(
        ledger_path=ledger,
        stage="rainfall_split",
        evidence_paths=[second],
        passed=True,
    )

    loaded = TFVPipelineLedger.from_json(ledger)
    loaded.verify_integrity()
    assert loaded.stages["inp_preflight"].passed is True
    assert loaded.stages["rainfall_split"].passed is True


def test_pipeline_ledger_rejects_skipped_prerequisite(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "phase0.json", "phase0")
    with pytest.raises(ValueError, match="prerequisite inp_preflight"):
        record_pipeline_stage(
            ledger_path=tmp_path / "ledger.json",
            stage="phase0_timescale",
            evidence_paths=[evidence],
            passed=True,
        )
