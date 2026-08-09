from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineLedger, create_policy_lock, evidence_from_files, sha256_file


_REQUIRED_PHYSICAL = {
    "frozen_inp",
    "priority_nodes",
    "sensor_layout",
    "time_scale_config",
}

_REQUIRED_SCIENTIFIC = {
    "step1_model",
    "step2_model",
    "graph_schema",
    "state_schema",
    "actuator_catalog",
    "split_registry",
    "model_acceptance_contract",
    "step1_acceptance",
    "step2_acceptance",
    "gradient_acceptance",
    "candidate_ranking_acceptance",
    "safety_calibration",
    "safety_audit",
    "controller_config",
    "rainfall_forecast_config",
    "fallback_policy",
    "baseline_plan",
}


def _require_passed(path: str | Path, name: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError(f"formal Policy Lock requires passed evidence: {name}")


def create_formal_policy_lock(
    *,
    ledger_path: str | Path,
    artefacts_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    ledger = PipelineLedger.from_json(ledger_path)
    ledger.verify_integrity()
    artefacts = json.loads(Path(artefacts_path).read_text(encoding="utf-8"))
    if not isinstance(artefacts, dict):
        raise ValueError("artefacts JSON must map stable names to file paths")
    required = _REQUIRED_PHYSICAL | _REQUIRED_SCIENTIFIC
    missing = sorted(required - set(artefacts))
    if missing:
        raise ValueError(f"formal Policy Lock missing artefacts: {missing}")
    for name in sorted(required):
        path = Path(str(artefacts[name]))
        if not path.is_file():
            raise ValueError(f"formal Policy Lock artefact missing on disk: {name}: {path}")
    for name in (
        "step1_acceptance",
        "step2_acceptance",
        "gradient_acceptance",
        "candidate_ranking_acceptance",
        "safety_audit",
    ):
        _require_passed(artefacts[name], name)

    split = Path(str(artefacts["split_registry"]))
    if split.stat().st_size == 0:
        raise ValueError("locked split_registry is empty")
    priority = [
        line.strip()
        for line in Path(str(artefacts["priority_nodes"])).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError(f"formal project contract requires exactly 8 unique observed priority nodes, got {len(priority)}")

    lock = create_policy_lock(ledger=ledger, artefacts=artefacts, output_path=output_path)
    # Extend the lock payload with explicit physical fingerprints for quick audits. These
    # hashes are already part of the canonical policy hash via create_policy_lock.
    lock["physical_contract"] = {
        name: {"path": str(artefacts[name]), "sha256": sha256_file(artefacts[name])}
        for name in sorted(_REQUIRED_PHYSICAL)
    }
    lock["formal_contract"] = "WUHAN_RTC_FORMAL_POLICY_LOCK_V2"
    Path(output_path).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Record only after the final V2 file bytes exist, otherwise the ledger hash would be stale.
    ledger.record(
        evidence_from_files(
            "policy_lock",
            [output_path],
            passed=True,
            notes=str(lock["policy_sha256"]),
        )
    )
    ledger.to_json(ledger_path)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict physical+scientific Policy Lock")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = create_formal_policy_lock(
        ledger_path=args.ledger,
        artefacts_path=args.artifacts,
        output_path=args.out,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
