from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_v122_wrapper_rejects_gradient_search_candidate_policy(tmp_path: Path) -> None:
    """Exercise the real script entrypoint without assuming ``scripts`` is importable.

    GitHub Actions installs the project from ``src/`` and invokes the ``pytest`` console
    script, so repository-root helper scripts are not guaranteed to be importable as a
    Python package.  The production behavior under test is the CLI fail-closed guard;
    invoking that CLI in a subprocess is both more faithful and path independent.
    """
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"v120_contract": "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1"}),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle.pt"
    import torch

    torch.save(
        {
            "candidate_policy": {"continuous_gradient_search": True},
            "value_gate": {"passed": True},
        },
        bundle,
    )
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_policy_v122.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--step2",
            str(bundle),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    combined = completed.stdout + "\n" + completed.stderr
    assert "continuous gradient search disabled" in combined
