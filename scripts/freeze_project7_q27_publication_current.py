"""Freeze the Project7 policy-consistent Q27 publication panel and immutable lineage.

This is a zero-SWMM operation.  Final events are selected only from the supplied prepared-event
registry after excluding controller-outcome exposures.  Prior Step1/Step2 use does not itself exclude
an event.  All eligible events are retained; no hydraulic/performance metric is used for selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from rtc.project7_q27_publication import (
    POLICY_LOCK_CONTRACT,
    PUBLICATION_PROTOCOL_CONTRACT,
    canonical_sha256,
    event_id,
    select_outcome_unexposed_events,
    sha256_file,
)


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


def _event_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("rt", encoding="utf-8-sig", newline="") as handle:
            return [event_id(row) for row in csv.DictReader(handle) if event_id(row)]
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [event_id(row) for row in rows if isinstance(row, dict) and event_id(row)]
    if path.suffix.lower() == ".json":
        return [event_id(row) for row in _walk(json.loads(path.read_text(encoding="utf-8")))]
    raise ValueError(f"unsupported prepared registry: {path}")


def _git_head(repo: Path) -> str:
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if len(value) != 40:
        raise RuntimeError("could not resolve Git HEAD")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prepared-events", required=True)
    parser.add_argument("--outcome-exposure-ledger", required=True)
    parser.add_argument("--q27-checkpoint", required=True)
    parser.add_argument("--q27-report", required=True)
    parser.add_argument("--policy-consistent-manifest", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--matched-baseline-audit", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {
        key: Path(value).resolve()
        for key, value in {
            "protocol": args.protocol,
            "prepared_events": args.prepared_events,
            "outcome_exposure_ledger": args.outcome_exposure_ledger,
            "q27_checkpoint": args.q27_checkpoint,
            "q27_report": args.q27_report,
            "policy_consistent_manifest": args.policy_consistent_manifest,
            "asset_manifest": args.asset_manifest,
            "matched_baseline_audit": args.matched_baseline_audit,
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Policy Lock inputs missing: {missing}")

    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if protocol.get("contract") != PUBLICATION_PROTOCOL_CONTRACT:
        raise ValueError("unexpected Q27 publication protocol")
    target = str(protocol["policy"]["target_continuation_policy_sha256"]).lower()
    view = json.loads(paths["policy_consistent_manifest"].read_text(encoding="utf-8"))
    if str(view.get("target_continuation_policy_sha256", "")).lower() != target:
        raise ValueError("policy-consistent view continuation differs from publication protocol")
    if view.get("leakage_audit", {}).get("passed") is not True:
        raise ValueError("policy-consistent view leakage audit did not pass")

    ledger = json.loads(paths["outcome_exposure_ledger"].read_text(encoding="utf-8"))
    exposures = ledger.get("exposures", [])
    if not isinstance(exposures, list):
        raise ValueError("outcome exposure ledger lacks exposures list")
    prepared = _event_ids(paths["prepared_events"])
    selected, selection = select_outcome_unexposed_events(prepared, exposures)
    if not selected:
        raise RuntimeError("no outcome-unexposed prepared events remain for publication Final")

    artifact_sha = {name: sha256_file(path) for name, path in paths.items()}
    panel_sha = canonical_sha256(selected)
    payload = {
        "contract": POLICY_LOCK_CONTRACT,
        "git_head": _git_head(Path(args.repo).resolve()),
        "target_continuation_policy_sha256": target,
        "locked_final_event_ids": selected,
        "locked_final_event_count": len(selected),
        "locked_final_panel_sha256": panel_sha,
        "selection": selection,
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "artifact_sha256": artifact_sha,
        "final_event_selection_used_controller_performance": False,
        "all_eligible_events_retained": True,
        "post_lock_training_or_tuning_forbidden": True,
        "final_outcomes_opened_before_lock": False,
        "new_rainfall_generated": False,
        "new_swmm_runs": 0,
        "ready_for_final": True,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
