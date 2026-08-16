from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage_checkpoint_audit_current.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage_checkpoint_audit_compat", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compat_help_exposes_required_chunk_and_audit_selector() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--audit {flow,hydraulic,gradient}" in result.stdout
    assert "--hydraulic-branch-chunk" in result.stdout
    assert "source-strict" in result.stdout.lower()


def test_memory_design_override_changes_only_hydraulic_chunk() -> None:
    compat = _load_module()

    @dataclass(frozen=True)
    class FakeMemoryDesign:
        hydraulic_branch_chunk: int = 4
        rollout_candidates_per_group: int = 2
        objective_candidate_chunk: int = 2
        evaluation_branch_chunk: int = 4
        residual_sample_rows: int = 131072

    fake_module = SimpleNamespace(
        __name__="fake_audit",
        V127StreamingMemoryDesign=FakeMemoryDesign,
    )
    original = compat._install_memory_design_override(
        fake_module, hydraulic_branch_chunk=2
    )
    changed = fake_module.V127StreamingMemoryDesign(
        hydraulic_branch_chunk=4,
        rollout_candidates_per_group=7,
        objective_candidate_chunk=3,
        evaluation_branch_chunk=5,
        residual_sample_rows=123,
    )

    assert original is FakeMemoryDesign
    assert changed.hydraulic_branch_chunk == 2
    assert changed.rollout_candidates_per_group == 7
    assert changed.objective_candidate_chunk == 3
    assert changed.evaluation_branch_chunk == 5
    assert changed.residual_sample_rows == 123


def test_memory_design_override_rejects_nonpositive_chunk() -> None:
    compat = _load_module()
    fake_module = SimpleNamespace(__name__="fake_audit", V127StreamingMemoryDesign=object)
    with pytest.raises(ValueError, match="must be positive"):
        compat._install_memory_design_override(fake_module, hydraulic_branch_chunk=0)
