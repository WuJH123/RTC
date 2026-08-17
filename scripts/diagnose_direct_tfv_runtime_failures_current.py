"""Separate accelerator/environment fallbacks from controller or policy failures in a Direct-TFV run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.runtime_failure_diagnostics import summarize_runtime_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Direct-TFV metadata must be a JSON object")
    decision_name = metadata.get("decision_file")
    if not decision_name:
        raise ValueError("Direct-TFV metadata lacks decision_file")
    decision_path = metadata_path.parent / str(decision_name)
    rows = [
        json.loads(line)
        for line in decision_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("decision log contains a non-object row")
    payload = summarize_runtime_failures(rows)
    payload.update(
        {
            "metadata_path": str(metadata_path),
            "decision_path": str(decision_path.resolve()),
            "decisions": len(rows),
        }
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
