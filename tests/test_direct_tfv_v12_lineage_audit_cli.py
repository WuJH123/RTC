from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v12_lineage_audit_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/audit_direct_tfv_v12_admission_lineage_current.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--require-compatible" in completed.stdout
