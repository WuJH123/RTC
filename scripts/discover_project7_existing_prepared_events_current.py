"""Discover already-prepared Project7 event INPs for contamination-safe Final reblinding.

The command does not generate rainfall and never starts SWMM. It only inventories existing INP files,
computes hashes, validates the common event clock, and parses forcing descriptors from event IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rtc.event_clock import inspect_prepared_event_clock
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES


REGISTRY_CONTRACT = "PROJECT7_EXISTING_PREPARED_EVENT_REGISTRY_V1"
_PATTERNS = (
    re.compile(r"^T(?P<rp>\d+)_D(?P<duration>\d+)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)RP(?P<rp>\d+)_D(?P<duration>\d+)(?:_|$)", re.IGNORECASE),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forcing(event_id: str) -> tuple[int, int] | None:
    for pattern in _PATTERNS:
        match = pattern.search(event_id)
        if match:
            return int(match.group("rp")), int(match.group("duration"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-root", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--include-clock-incompatible", action="store_true")
    args = parser.parse_args()

    rows: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for root_value in args.events_root:
        root = Path(root_value).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        for path in sorted(root.rglob("*.inp")):
            event_id = path.stem
            forcing = _forcing(event_id)
            if forcing is None:
                continue
            rp, duration = forcing
            digest = _sha(path)
            clock = inspect_prepared_event_clock(path)
            compatible = (
                abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) <= 1.0e-6
            )
            if not compatible and not args.include_clock_incompatible:
                skipped.append({"event_id": event_id, "path": str(path), "reason": "CLOCK_INCOMPATIBLE"})
                continue
            row = {
                "event_id": event_id,
                "return_period_year": rp,
                "duration_minutes": duration,
                "inp_path": str(path),
                "prepared_inp_sha256": digest,
                "effective_warmup_minutes": float(clock["effective_warmup_minutes"]),
                "clock_compatible": compatible,
            }
            previous = rows.get(event_id)
            if previous is not None and previous["prepared_inp_sha256"] != digest:
                raise RuntimeError(
                    f"same event_id resolves to different INP bytes: {event_id}\n"
                    f"{previous['inp_path']}\n{path}"
                )
            rows[event_id] = row

    if not rows:
        raise RuntimeError("no parseable prepared Project7 event INPs found")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "PROJECT7_EXISTING_PREPARED_EVENT_REGISTRY.csv"
    fields = [
        "event_id",
        "return_period_year",
        "duration_minutes",
        "inp_path",
        "prepared_inp_sha256",
        "effective_warmup_minutes",
        "clock_compatible",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: str(row["event_id"])))
    payload = {
        "contract": REGISTRY_CONTRACT,
        "new_rainfall_generated": False,
        "swmm_started": False,
        "events_root": [str(Path(value).resolve()) for value in args.events_root],
        "event_count": len(rows),
        "clock_incompatible_skipped_count": len(skipped),
        "registry_csv_path": str(csv_path),
        "registry_csv_sha256": _sha(csv_path),
        "skipped": skipped,
    }
    json_path = out_dir / "PROJECT7_EXISTING_PREPARED_EVENT_REGISTRY.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "registry_json_path": str(json_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
