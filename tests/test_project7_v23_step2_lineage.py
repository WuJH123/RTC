from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from rtc.project7_v23_step2_lineage import (
    V23_STEP2_CHECKPOINT_SHA256,
    V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT,
    V23_STEP2_LINEAGE_EVIDENCE_CONTRACT,
    V23_V15_CHECKPOINT_SHA256,
    V23_V21_CHECKPOINT_SHA256,
    validate_v23_step2_component_diagnostic,
    validate_v23_step2_lineage_evidence,
)


REPO = Path(__file__).resolve().parents[1]


def _lineage() -> dict[str, object]:
    return {
        "contract": V23_STEP2_LINEAGE_EVIDENCE_CONTRACT,
        "lineage_pass": True,
        "step2_checkpoint_path": "C:/study/direct_tfv_core_v5_dev_20260817_144117/step2_direct_tfv_value_dev.pt",
        "step2_checkpoint_sha256": V23_STEP2_CHECKPOINT_SHA256,
        "asset_manifest_step2_sha256": V23_STEP2_CHECKPOINT_SHA256,
        "v15_checkpoint_sha256": V23_V15_CHECKPOINT_SHA256,
        "v15_base_step2_sha256": V23_STEP2_CHECKPOINT_SHA256,
        "v21_checkpoint_sha256": V23_V21_CHECKPOINT_SHA256,
        "v21_base_step2_sha256": V23_STEP2_CHECKPOINT_SHA256,
        "v21_rank_source_checkpoint_sha256": V23_V15_CHECKPOINT_SHA256,
        "step2_retrained_for_formal": False,
    }


def test_v23_step2_lineage_accepts_only_frozen_v5_chain() -> None:
    payload = _lineage()
    validate_v23_step2_lineage_evidence(payload)
    payload["v15_base_step2_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="V15 rank"):
        validate_v23_step2_lineage_evidence(payload)


def test_component_diagnostic_rejects_other_step2() -> None:
    payload = {
        "contract": V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT,
        "metric_role": "LEGACY_COMPONENT_DIAGNOSTIC_NOT_POLICY_LOCK_HARD_GATE",
        "step2_checkpoint_sha256": "0" * 64,
        "new_swmm_truth_generated": False,
        "step2_retrained": False,
    }
    with pytest.raises(RuntimeError, match="V23 V5"):
        validate_v23_step2_component_diagnostic(payload)


def test_v5_component_compiler_requires_v5_bound_source(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps(_lineage()) + "\n", encoding="utf-8")
    wrong_source = tmp_path / "v41_diagnostic.json"
    wrong_source.write_text('{"version":"V41"}\n', encoding="utf-8")
    out = tmp_path / "diagnostic.json"
    command = [
        sys.executable,
        str(REPO / "scripts" / "compile_project7_v23_step2_v5_component_diagnostic_current.py"),
        "--step2-lineage-evidence",
        str(lineage),
        "--tfv-exact-truth-rank-correlation",
        "0.865",
        "--source-evidence",
        str(wrong_source),
        "--out",
        str(out),
    ]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "do not substitute V4/V41" in completed.stderr
    assert not out.exists()


def test_v5_component_compiler_accepts_source_that_declares_v5_sha(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps(_lineage()) + "\n", encoding="utf-8")
    source = tmp_path / "v5_metric.json"
    source.write_text(json.dumps({"step2_checkpoint_sha256": V23_STEP2_CHECKPOINT_SHA256}) + "\n", encoding="utf-8")
    out = tmp_path / "diagnostic.json"
    command = [
        sys.executable,
        str(REPO / "scripts" / "compile_project7_v23_step2_v5_component_diagnostic_current.py"),
        "--step2-lineage-evidence",
        str(lineage),
        "--tfv-exact-truth-rank-correlation",
        "0.55",
        "--query-balanced-top1",
        "0.6",
        "--source-evidence",
        str(source),
        "--out",
        str(out),
    ]
    subprocess.run(command, cwd=REPO, check=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["step2_checkpoint_sha256"] == V23_STEP2_CHECKPOINT_SHA256
    assert payload["standalone_acceptance_pass"] is False
    assert payload["step2_retrained"] is False
