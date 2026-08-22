"""Freeze the six prepared Project7 Final event INPs without opening any hydraulic outcome.

Pass exactly six ``--event EVENT_ID=PATH`` pairs. Event IDs must equal the preregistered Final cohort.
The command records immutable INP hashes and common-clock metadata only; it never starts SWMM.
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


FINAL_EVENT_MANIFEST_CONTRACT = "PROJECT7_V23_FROZEN_FINAL6_EVENT_MANIFEST_V1"


def _parse_event(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--event must use EVENT_ID=PATH")
    event_id, raw_path = value.split("=", 1)
    event_id = event_id.strip()
    path = Path(raw_path.strip()).resolve()
    if not event_id or not path.is_file():
        raise ValueError(f"invalid Final event mapping: {value}")
    return event_id, path


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
    if set(mappings) != set(roles["final"]):
        missing = sorted(set(roles["final"]) - set(mappings))
        extra = sorted(set(mappings) - set(roles["final"]))
        raise ValueError(f"Final event mapping differs from frozen split; missing={missing}, extra={extra}")

    events: list[dict[str, Any]] = []
    shas: set[str] = set()
    for event_id in roles["final"]:
        path = mappings[event_id]
        digest = sha256_file(path)
        if digest in shas:
            raise ValueError("two Final event IDs resolve to the same INP bytes")
        shas.add(digest)
        clock = inspect_prepared_event_clock(path)
        if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
            raise ValueError(f"Final event {event_id} does not have the frozen 120-min warm-up")
        events.append(
            {
                "event_id": event_id,
                "inp_path": str(path),
                "inp_sha256": digest,
                "prepared_event_clock": clock,
            }
        )

    payload = {
        "contract": FINAL_EVENT_MANIFEST_CONTRACT,
        "final_event_count": 6,
        "event_ids": list(roles["final"]),
        "events": events,
        "selection_basis": "FROZEN_PROJECT7_V069_FORCING_ONLY_SPLIT",
        "hydraulic_outcomes_opened": False,
        "proposed_performance_used_for_selection": False,
        "new_rainfall_generated": False,
        "new_training_data_generated": False,
        "formal_split_contract_sha256": sha256_file(split_path),
    }
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "final_manifest_path": str(destination),
                "final_manifest_sha256": sha256_file(destination),
                "event_ids": payload["event_ids"],
                "hydraulic_outcomes_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
