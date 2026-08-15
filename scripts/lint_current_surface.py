"""Run Ruff only on the active Project7 execution/development surface.

The repository intentionally retains historical implementation files for provenance and
shared orchestration. Their pre-existing lint debt must not block smoke/dev debugging.
This gate is the authoritative lint check for the current Project7 surface and is also run
by GitHub Actions.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "project7_current_lint_surface.json"
EXPECTED_CONTRACT = "PROJECT7_CURRENT_LINT_SURFACE_V1_ACTIVE_ONLY"


def _load_paths() -> list[str]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if payload.get("contract") != EXPECTED_CONTRACT:
        raise RuntimeError(f"unexpected current lint contract: {payload.get('contract')!r}")
    if payload.get("full_repository_ruff_is_gate") is not False:
        raise RuntimeError("current lint contract must not turn archival Ruff debt into a stop-gate")
    if payload.get("current_surface_ruff_is_gate") is not True:
        raise RuntimeError("current lint surface must remain a fail-closed gate")

    raw = payload.get("paths")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("current lint surface contains no paths")
    paths = [str(value) for value in raw]
    if len(paths) != len(set(paths)):
        raise RuntimeError("current lint surface contains duplicate paths")

    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"current lint surface references missing files: {missing}")
    non_python = [path for path in paths if not path.endswith(".py")]
    if non_python:
        raise RuntimeError(f"current lint surface contains non-Python files: {non_python}")
    return paths


def main() -> int:
    paths = _load_paths()
    print(f"{EXPECTED_CONTRACT}: checking {len(paths)} active Python files")
    command = [sys.executable, "-m", "ruff", "check", *paths]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
