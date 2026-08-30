"""Re-audit a frozen Q27 publication panel without running SWMM.

This command is intended for publication-resume decisions after an interrupted Final.  It verifies
that the Policy Lock is exactly the set of prepared events minus blocking controller-outcome exposure,
that the locked panel SHA is stable, and that the lock still points to the supplied prepared registry
and exposure ledger bytes.  It deliberately does not inspect TFV/PFV outcomes and therefore cannot
change Final membership based on performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from rtc.project7_q27_publication import (
    canonical_sha256,
    event_id,
    select_outcome_unexposed_events,
    sha256_file,
)


CONTRACT = "PROJECT7_Q27_PUBLICATION_LOCKED_PANEL_RESUME_AUDIT_V1"


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if event_id(value):
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _prepared_ids(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("rt", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        rows = list(_walk(json.loads(path.read_text(encoding="utf-8"))))
    else:
        raise ValueError(f"unsupported prepared registry: {path}")
    ids = [event_id(row) for row in rows if isinstance(row, dict) and event_id(row)]
    if len(ids) != len(set(ids)):
        raise ValueError("prepared registry contains duplicate event IDs")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-events", required=True)
    parser.add_argument("--outcome-exposure-ledger", required=True)
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prepared_path = Path(args.prepared_events).resolve()
    ledger_path = Path(args.outcome_exposure_ledger).resolve()
    lock_path = Path(args.policy_lock).resolve()
    for path in (prepared_path, ledger_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    prepared = _prepared_ids(prepared_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    exposures = ledger.get("exposures", [])
    if not isinstance(exposures, list):
        raise ValueError("outcome exposure ledger lacks exposures list")
    expected, selection = select_outcome_unexposed_events(prepared, exposures)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = [str(value).strip() for value in lock.get("locked_final_event_ids", []) if str(value).strip()]
    if len(locked) != len(set(locked)):
        raise ValueError("Policy Lock contains duplicate event IDs")

    expected_set = set(expected)
    locked_set = set(locked)
    locked_not_expected = sorted(locked_set - expected_set)
    expected_not_locked = sorted(expected_set - locked_set)
    panel_sha_expected = canonical_sha256(sorted(expected))
    panel_sha_locked = str(lock.get("locked_final_panel_sha256", "")).strip().lower()

    artifact_sha = lock.get("artifact_sha256", {})
    if not isinstance(artifact_sha, dict):
        artifact_sha = {}
    prepared_sha = sha256_file(prepared_path)
    ledger_sha = sha256_file(ledger_path)
    lock_prepared_sha = str(artifact_sha.get("prepared_events", "")).strip().lower()
    lock_ledger_sha = str(artifact_sha.get("outcome_exposure_ledger", "")).strip().lower()

    panel_consistency_pass = (
        not locked_not_expected
        and not expected_not_locked
        and len(locked) == len(expected)
        and panel_sha_locked == panel_sha_expected
        and lock_prepared_sha == prepared_sha
        and lock_ledger_sha == ledger_sha
    )

    payload = {
        "contract": CONTRACT,
        "prepared_registry_path": str(prepared_path),
        "prepared_registry_sha256": prepared_sha,
        "outcome_exposure_ledger_path": str(ledger_path),
        "outcome_exposure_ledger_sha256": ledger_sha,
        "policy_lock_path": str(lock_path),
        "policy_lock_sha256": sha256_file(lock_path),
        "policy_lock_git_head": str(lock.get("git_head", "")),
        "prepared_event_count": len(prepared),
        "expected_eligible_event_count": len(expected),
        "locked_event_count": len(locked),
        "selection": selection,
        "locked_not_expected": locked_not_expected,
        "expected_not_locked": expected_not_locked,
        "expected_panel_sha256": panel_sha_expected,
        "locked_panel_sha256": panel_sha_locked,
        "lock_prepared_registry_sha256": lock_prepared_sha,
        "lock_exposure_ledger_sha256": lock_ledger_sha,
        "panel_consistency_pass": panel_consistency_pass,
        "safe_to_resume_locked_final_panel": panel_consistency_pass,
        "outcomes_inspected_for_panel_selection": False,
        "policy_modified": False,
        "model_modified": False,
        "new_swmm_runs": 0,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not panel_consistency_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
