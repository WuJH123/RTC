"""Fail closed unless a V9 state-sufficiency report matches the current git HEAD."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rtc.step2_v90_evidence import validate_state_sufficiency_evidence_v90


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve current git HEAD for V9 evidence validation") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate current-head Project7 Step2 V9 A/B/C evidence"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--expected-git-head",
        default=None,
        help="40-character SHA; defaults to git rev-parse HEAD",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    expected = args.expected_git_head or _git_head()
    accepted = validate_state_sufficiency_evidence_v90(
        payload, expected_git_head=expected
    )
    text = json.dumps(accepted, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
