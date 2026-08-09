from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtc.code_contract import rtc_source_tree_sha256
from rtc.formalize_run import _verify_strategy_execution


def _inp(path: Path, *, controls: bool) -> Path:
    path.write_text(
        "[JUNCTIONS]\nN1 0 1 0 0 0\n[CONTROLS]\n"
        + ("RULE R1\nIF NODE N1 DEPTH > 1\nTHEN PUMP P1 SETTING = 1\n" if controls else ""),
        encoding="utf-8",
    )
    return path


def _metadata(root: Path, *, controls: bool, controller: bool, rows: list[dict]) -> tuple[dict, Path]:
    inp = _inp(root / "runtime.inp", controls=controls)
    decisions = root / "run.decisions.jsonl"
    decisions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = {
        "inp_path": str(inp),
        "decision_file": decisions.name,
        "controller_present": controller,
    }
    meta_path = root / "run.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata, meta_path


def test_no_control_and_internal_labels_are_checked_from_runtime(tmp_path: Path) -> None:
    no_control, no_path = _metadata(tmp_path, controls=False, controller=False, rows=[])
    assert _verify_strategy_execution(
        metadata=no_control, metadata_path=no_path, strategy="no_control"
    )["passed"] is True
    with pytest.raises(ValueError, match="Internal-RTC"):
        _verify_strategy_execution(
            metadata=no_control, metadata_path=no_path, strategy="internal_rtc"
        )


def test_all_open_label_requires_every_recorded_setting_open(tmp_path: Path) -> None:
    metadata, path = _metadata(
        tmp_path,
        controls=False,
        controller=True,
        rows=[{"source": "ALL_OPEN", "settings": {"A1": 1.0, "A2": 0.9}}],
    )
    with pytest.raises(ValueError, match="did not command"):
        _verify_strategy_execution(metadata=metadata, metadata_path=path, strategy="all_open")


def test_proposed_label_requires_public_guard_artifact_lineage(tmp_path: Path) -> None:
    metadata, path = _metadata(
        tmp_path,
        controls=False,
        controller=True,
        rows=[{"source": "MPC", "settings": {"A1": 0.5}}],
    )
    metadata.update(
        {
            "strategy": "proposed",
            "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        }
    )
    with pytest.raises(ValueError, match="controller_config_sha256"):
        _verify_strategy_execution(metadata=metadata, metadata_path=path, strategy="proposed")

    metadata.update(
        {
            "controller_config_sha256": "c",
            "graph_schema_sha256": "g",
            "step1_model_sha256": "s1",
            "step2_model_sha256": "s2",
        }
    )
    assert _verify_strategy_execution(
        metadata=metadata, metadata_path=path, strategy="proposed"
    )["passed"] is True
