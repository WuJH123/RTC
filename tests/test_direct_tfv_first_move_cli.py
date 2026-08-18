from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rtc.checkpoint_direct_tfv import direct_tfv_first_move_source_sha256


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "scripts" / "design_direct_tfv_first_move_calibration_current.py"
MERGE = ROOT / "scripts" / "merge_direct_tfv_first_move_panel_shards.py"
CALIBRATE = ROOT / "scripts" / "calibrate_direct_tfv_first_move_admission_current.py"
RUNTIME = ROOT / "scripts" / "run_policy_direct_tfv_first_move_development.py"
PFV = ROOT / "scripts" / "add_pfv_to_direct_tfv_comparison_current.py"


def _help(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_panel_exposes_four_way_gpu_sharding() -> None:
    text = _help(PANEL)
    assert "--shard-count" in text
    assert "--shard-index" in text
    assert "--first-move-maxiter" in text


def test_merge_calibration_runtime_and_pfv_surfaces_are_routable() -> None:
    merge = _help(MERGE)
    assert "--manifest" in merge and "--summary" in merge and "--summary-out" in merge
    calibration = _help(CALIBRATE)
    assert "--first-move-cache-manifest" in calibration
    runtime = _help(RUNTIME)
    assert "--first-move-admission-calibration" in runtime
    assert "--first-move-deadline-seconds" in runtime
    pfv = _help(PFV)
    assert "--priority-nodes" in pfv


def test_first_move_source_fingerprint_is_byte_level_and_routed_end_to_end() -> None:
    fingerprint = direct_tfv_first_move_source_sha256()
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")
    panel_source = PANEL.read_text(encoding="utf-8")
    calibration_source = CALIBRATE.read_text(encoding="utf-8")
    runtime_source = RUNTIME.read_text(encoding="utf-8")
    for source in (panel_source, calibration_source, runtime_source):
        assert "first_move_source_sha256" in source
    assert "direct_tfv_first_move_source_sha256" in panel_source
    assert "direct_tfv_first_move_source_sha256" in calibration_source
    assert "direct_tfv_first_move_source_sha256" in runtime_source
