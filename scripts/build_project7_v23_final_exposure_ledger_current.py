"""Build an identifier-only pre-lock exposure ledger for Project7 V23 Final remediation.

The ledger answers one question: has an already-prepared event identity appeared in any model-learning
or development-steering evidence before Policy Lock? It intentionally ignores PFV/TFV/depth/control
values. Historical evidence is never deleted or relabelled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rtc.project7_v23_final_remediation import FINAL_EXPOSURE_LEDGER_CONTRACT, normalise_event_id
from rtc.project7_v23_formal_reuse import validate_frozen_split


_REQUIRED_COLUMNS = {
    "event_id",
    "return_period_year",
    "duration_minutes",
    "inp_path",
    "prepared_inp_sha256",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"candidate registry must contain {sorted(_REQUIRED_COLUMNS)}; got {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("candidate registry is empty")
    seen: dict[str, str] = {}
    for row in rows:
        event_id = normalise_event_id(row.get("event_id"))
        digest = str(row.get("prepared_inp_sha256", "")).strip().lower()
        if not event_id or len(digest) != 64:
            raise ValueError("candidate registry contains invalid event identity/hash")
        previous = seen.get(event_id)
        if previous is not None and previous != digest:
            raise ValueError(f"candidate registry has conflicting bytes for {event_id}")
        seen[event_id] = digest
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        unique[normalise_event_id(row["event_id"])] = row
    return list(unique.values())


def _category_files(values: list[str]) -> list[Path]:
    paths = [Path(value).resolve() for value in values]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _find_event_ids(path: Path, candidate_ids: set[str]) -> set[str]:
    # Identifier-presence only. Numeric hydraulic/controller values are neither parsed nor returned.
    text = path.read_bytes().decode("utf-8", errors="ignore")
    found: set[str] = set()
    for event_id in candidate_ids:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(event_id)}(?![A-Za-z0-9_])"
        if re.search(pattern, text):
            found.add(event_id)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-split-contract", required=True)
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--policy-return-records-jsonl", action="append", required=True)
    parser.add_argument("--step1-exposure-file", action="append", required=True)
    parser.add_argument("--step2-exposure-file", action="append", required=True)
    parser.add_argument("--development-exposure-file", action="append", required=True)
    parser.add_argument("--additional-exposure-file", action="append", default=[])
    parser.add_argument("--exposed-event-id", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_path = Path(args.original_split_contract).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    roles = validate_frozen_split(split)
    registry_path = Path(args.candidate_registry).resolve()
    rows = _registry(registry_path)
    candidate_ids = {normalise_event_id(row["event_id"]) for row in rows}

    categories = {
        "policy_return": _category_files(args.policy_return_records_jsonl),
        "step1_learning_or_validation": _category_files(args.step1_exposure_file),
        "step2_learning_or_validation": _category_files(args.step2_exposure_file),
        "development_steering_or_controller_truth": _category_files(args.development_exposure_file),
        "additional_prelock_exposure": _category_files(args.additional_exposure_file),
    }
    exposures: dict[str, list[dict[str, str]]] = {event_id: [] for event_id in candidate_ids}
    evidence_manifest: dict[str, list[dict[str, str]]] = {}
    for category, paths in categories.items():
        evidence_manifest[category] = []
        for path in paths:
            digest = _sha(path)
            found = _find_event_ids(path, candidate_ids)
            evidence_manifest[category].append(
                {"path": str(path), "sha256": digest, "matched_candidate_count": str(len(found))}
            )
            for event_id in found:
                exposures[event_id].append(
                    {"category": category, "path": str(path), "sha256": digest}
                )
    for raw in args.exposed_event_id:
        event_id = normalise_event_id(raw)
        if event_id not in candidate_ids:
            raise ValueError(f"explicit exposed event is absent from candidate registry: {event_id}")
        exposures[event_id].append(
            {"category": "explicit_prelock_exposure", "path": "<CLI>", "sha256": ""}
        )

    original_role: dict[str, str] = {}
    for role, values in roles.items():
        for value in values:
            original_role[normalise_event_id(value)] = role
    validation = {normalise_event_id(value) for value in roles["development_validation"]}

    event_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: normalise_event_id(item["event_id"])):
        event_id = normalise_event_id(row["event_id"])
        source = exposures[event_id]
        exposed = bool(source)
        event_rows.append(
            {
                "event_id": event_id,
                "return_period_year": int(float(row["return_period_year"])),
                "duration_minutes": int(float(row["duration_minutes"])),
                "inp_path": str(row["inp_path"]),
                "prepared_inp_sha256": str(row["prepared_inp_sha256"]).strip().lower(),
                "original_role": original_role.get(event_id, "outside_original_split"),
                "exposed_prelock": exposed,
                "protected_original_validation": event_id in validation,
                "eligible_for_reblind_final": bool(not exposed and event_id not in validation),
                "exposure_sources": source,
            }
        )

    eligible = [row for row in event_rows if row["eligible_for_reblind_final"]]
    payload = {
        "contract": FINAL_EXPOSURE_LEDGER_CONTRACT,
        "source_original_split_contract": str(split.get("contract")),
        "source_original_split_sha256": _sha(split_path),
        "candidate_registry_path": str(registry_path),
        "candidate_registry_sha256": _sha(registry_path),
        "candidate_event_count": len(event_rows),
        "eligible_event_count": len(eligible),
        "eligible_duration_minutes": sorted({int(row["duration_minutes"]) for row in eligible}),
        "evidence_categories_complete": True,
        "required_exposure_categories": [
            "policy_return",
            "step1_learning_or_validation",
            "step2_learning_or_validation",
            "development_steering_or_controller_truth",
        ],
        "evidence_manifest": evidence_manifest,
        "identifier_presence_only": True,
        "hydraulic_outcomes_used_for_eligibility": False,
        "controller_performance_used_for_eligibility": False,
        "historical_records_deleted_or_relabeled": False,
        "new_rainfall_generated": False,
        "new_training_data_generated": False,
        "swmm_started": False,
        "events": event_rows,
    }
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ledger_path": str(destination),
                "ledger_sha256": _sha(destination),
                "candidate_event_count": len(event_rows),
                "eligible_event_count": len(eligible),
                "eligible_duration_minutes": payload["eligible_duration_minutes"],
                "hydraulic_outcomes_used_for_eligibility": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
