"""Freeze the active six-event Project7 Development-Validation prepared-INP manifest.

The active Formal split may be the original v0.6.9 split or its contamination-remediated successor.
Only path/hash/common-clock metadata are recorded; no SWMM run is started.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.event_clock import inspect_prepared_event_clock
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES
from rtc.project7_v23_formal_reuse import validate_frozen_split


VALIDATION_EVENT_MANIFEST_CONTRACT = "PROJECT7_V23_FROZEN_DEVELOPMENT_VALIDATION6_MANIFEST_V1"


def _parse_event(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--event must use EVENT_ID=PATH")
    event_id, raw_path = value.split("=", 1)
    path = Path(raw_path.strip()).resolve()
    if not event_id.strip() or not path.is_file():
        raise ValueError(f"invalid Validation event mapping: {value}")
    return event_id.strip(), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-split-contract", required=True)
    parser.add_argument("--event", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_path = Path(args.formal_split_contract).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    roles = validate_frozen_split(split)
    mappings = dict(_parse_event(value) for value in args.event)
    expected = set(roles["development_validation"])
    if set(mappings) != expected:
        raise ValueError(
            "Validation event mapping differs from frozen split; "
            f"missing={sorted(expected-set(mappings))}, extra={sorted(set(mappings)-expected)}"
        )
    events: list[dict[str, Any]] = []
    shas: set[str] = set()
    for event_id in roles["development_validation"]:
        path = mappings[event_id]
        digest = sha256_file(path)
        if digest in shas:
            raise ValueError("two Validation event IDs resolve to the same INP bytes")
        shas.add(digest)
        clock = inspect_prepared_event_clock(path)
        if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
            raise ValueError(f"Validation event {event_id} violates the 120-min warm-up")
        events.append(
            {
                "event_id": event_id,
                "inp_path": str(path),
                "inp_sha256": digest,
                "prepared_event_clock": clock,
            }
        )
    payload = {
        "contract": VALIDATION_EVENT_MANIFEST_CONTRACT,
        "event_count": 6,
        "event_ids": list(roles["development_validation"]),
        "events": events,
        "source_split_contract": str(split.get("contract", "")),
        "selection_basis": "ACTIVE_FROZEN_SPLIT_DEVELOPMENT_VALIDATION",
        "final_truth_opened": False,
        "new_rainfall_generated": False,
        "new_training_data_generated": False,
        "formal_split_contract_sha256": sha256_file(split_path),
    }
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"validation_manifest_path": str(destination), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
