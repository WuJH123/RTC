"""Run the high-signal Ruff gate for the maintained Project7 surface."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "project7_current_lint_surface.json"
EXPECTED_CONTRACT = "PROJECT7_CURRENT_LINT_SURFACE_DIRECT_TFV_V12"
EXPECTED_RULES = ("E4", "E7", "E9", "F")


def _load_contract() -> tuple[list[str], tuple[str, ...]]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if payload.get("contract") != EXPECTED_CONTRACT:
        raise RuntimeError(f"unexpected current lint contract: {payload.get('contract')!r}")
    if payload.get("full_repository_ruff_is_gate") is not False:
        raise RuntimeError("current lint contract must not turn repository-wide Ruff debt into a stop-gate")
    if payload.get("current_surface_ruff_is_gate") is not True:
        raise RuntimeError("current lint surface must remain a fail-closed gate")
    rules_raw = payload.get("rule_select")
    if not isinstance(rules_raw, list):
        raise RuntimeError("current lint contract lacks rule_select")
    rules = tuple(str(value) for value in rules_raw)
    if rules != EXPECTED_RULES:
        raise RuntimeError(f"current lint rules must remain {list(EXPECTED_RULES)}, got {list(rules)}")
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
    return paths, rules


def main() -> int:
    paths, rules = _load_contract()
    select = ",".join(rules)
    print(f"{EXPECTED_CONTRACT}: checking {len(paths)} maintained Python files with --select {select}")
    command = [sys.executable, "-m", "ruff", "check", "--select", select, *paths]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
