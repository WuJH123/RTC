from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tfv_pipeline import TFV_STAGES, TFVPipelineLedger


def record_pipeline_stage(
    *,
    ledger_path: str | Path,
    stage: str,
    evidence_paths: list[str | Path],
    passed: bool,
    notes: str = "",
) -> dict[str, object]:
    if stage not in TFV_STAGES:
        raise ValueError(f"unknown pipeline stage: {stage}")
    ledger_file = Path(ledger_path)
    if ledger_file.is_file():
        ledger = TFVPipelineLedger.from_json(ledger_file)
    else:
        ledger = TFVPipelineLedger()
    if passed and not evidence_paths:
        raise ValueError("a passed pipeline stage requires at least one evidence file")
    ledger.record_files(stage, evidence_paths, passed=passed, notes=notes)
    ledger.to_json(ledger_file)
    return {
        "contract": ledger.contract,
        "stage": stage,
        "passed": bool(passed),
        "evidence_files": [str(Path(p)) for p in evidence_paths],
        "ledger": str(ledger_file),
        "recorded_stages": list(ledger.stages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one ordered, hash-bound RTC scientific stage in the Policy-Lock ledger"
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--stage", required=True, choices=list(TFV_STAGES))
    parser.add_argument("--evidence", action="append", default=[])
    outcome = parser.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--passed", action="store_true")
    outcome.add_argument("--failed", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    payload = record_pipeline_stage(
        ledger_path=args.ledger,
        stage=args.stage,
        evidence_paths=args.evidence,
        passed=bool(args.passed),
        notes=args.notes,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
