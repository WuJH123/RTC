"""Build a zero-SWMM controller-outcome exposure ledger for Project7 Q27 publication.

Use repeated ``--source KIND=PATH`` arguments.  PATH may be CSV, JSON, or JSONL.  Event identity is
read from ``event_id``, ``event``, or ``rainfall_group``.  Prepared/Step1/Step2 use is non-blocking
unless the caller explicitly supplies a blocking controller-outcome kind.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from rtc.project7_q27_publication import (
    BLOCKING_OUTCOME_EXPOSURES,
    NONBLOCKING_EXPOSURES,
    event_id,
    exposure_counts,
    sha256_file,
)


CONTRACT = "PROJECT7_Q27_OUTCOME_EXPOSURE_LEDGER_V1"
VALID_KINDS = BLOCKING_OUTCOME_EXPOSURES | NONBLOCKING_EXPOSURES


def _source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be KIND=PATH")
    kind, raw = value.split("=", 1)
    kind = kind.strip().upper()
    path = Path(raw.strip()).resolve()
    if kind not in VALID_KINDS:
        raise argparse.ArgumentTypeError(f"unsupported exposure kind: {kind}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"source does not exist: {path}")
    return kind, path


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


def _rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("rt", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle) if event_id(row)]
    if suffix == ".jsonl":
        out: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw)
            if isinstance(value, dict) and event_id(value):
                out.append(value)
        return out
    if suffix == ".json":
        return list(_walk(json.loads(path.read_text(encoding="utf-8"))))
    raise ValueError(f"unsupported source format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=_source)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exposures: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for kind, path in args.source:
        rows = _rows(path)
        source_sha = sha256_file(path)
        sources.append({"kind": kind, "path": str(path), "sha256": source_sha, "event_rows": len(rows)})
        for row in rows:
            eid = event_id(row)
            key = (eid, kind, source_sha)
            if key in seen:
                continue
            seen.add(key)
            exposures.append(
                {"event_id": eid, "kind": kind, "source_path": str(path), "source_sha256": source_sha}
            )

    payload = {
        "contract": CONTRACT,
        "exposure_count": len(exposures),
        "unique_event_count": len({row["event_id"] for row in exposures}),
        "exposure_kind_counts": exposure_counts(exposures),
        "blocking_kinds": sorted(BLOCKING_OUTCOME_EXPOSURES),
        "nonblocking_kinds": sorted(NONBLOCKING_EXPOSURES),
        "prepared_step1_step2_use_is_not_implicitly_blocking": True,
        "new_swmm_runs": 0,
        "sources": sources,
        "exposures": sorted(exposures, key=lambda row: (row["event_id"], row["kind"])),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "exposures"}, indent=2))


if __name__ == "__main__":
    main()
