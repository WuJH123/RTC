"""Freeze five representative existing Project7 Development events for operational iteration.

No rainfall or SWMM truth is generated.  Either pass exactly five ``--event ID=PATH`` values, or
provide ``--candidate-dir`` and let the script choose five existing prepared INPs using filename-only
forcing descriptors (RP, duration, family).  Proposed performance is never used for selection.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

from rtc.operational_benchmark_v1 import build_event, write_benchmark_manifest


_EVENT_RE = re.compile(r"(?:^|[_-])RP(?P<rp>\d+)(?:[_-])D(?P<duration>\d+)(?:[_-](?P<family>[A-Za-z0-9-]+))?", re.I)
_TARGET_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _descriptor(path: Path) -> tuple[int, int, str]:
    match = _EVENT_RE.search(path.stem)
    if match is None:
        raise ValueError(
            f"cannot derive forcing-only RP/duration descriptor from event filename: {path.name}"
        )
    family = str(match.group("family") or "UNSPECIFIED").upper()
    return int(match.group("rp")), int(match.group("duration")), family


def _auto_select(paths: list[Path]) -> list[Path]:
    if len(paths) < 5:
        raise ValueError("candidate directory contains fewer than five prepared INPs")
    items: list[tuple[Path, int, int, str]] = []
    for path in sorted(paths):
        try:
            rp, duration, family = _descriptor(path)
        except ValueError:
            continue
        items.append((path, rp, duration, family))
    if len(items) < 5:
        raise ValueError("fewer than five candidate INPs expose RP/D forcing descriptors")
    rp_values = [math.log1p(item[1]) for item in items]
    duration_values = [float(item[2]) for item in items]
    rp_min, rp_max = min(rp_values), max(rp_values)
    d_min, d_max = min(duration_values), max(duration_values)

    def norm(value: float, lo: float, hi: float) -> float:
        return 0.5 if hi <= lo else (value - lo) / (hi - lo)

    scored = []
    for item, rp_log, duration in zip(items, rp_values, duration_values, strict=True):
        severity = 0.7 * norm(rp_log, rp_min, rp_max) + 0.3 * norm(duration, d_min, d_max)
        scored.append((*item, severity))
    selected: list[tuple[Path, int, int, str, float]] = []
    used: set[Path] = set()
    used_family: set[str] = set()
    for target in _TARGET_QUANTILES:
        available = [row for row in scored if row[0] not in used]
        # Primary criterion is forcing-severity quantile coverage. A small deterministic family
        # penalty improves temporal-pattern diversity without using any controller outcome.
        chosen = min(
            available,
            key=lambda row: (
                abs(row[4] - target) + (0.08 if row[3] in used_family else 0.0),
                row[1], row[2], row[3], row[0].name,
            ),
        )
        selected.append(chosen)
        used.add(chosen[0])
        used_family.add(chosen[3])
    return [row[0] for row in selected]


def _parse_explicit(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--event must be EVENT_ID=PATH")
        event_id, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((event_id.strip(), path))
    if len(result) != 5:
        raise ValueError("explicit operational benchmark requires exactly five --event values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--candidate-dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.event) == bool(args.candidate_dir):
        raise ValueError("choose exactly one of repeated --event or --candidate-dir")

    if args.event:
        explicit = _parse_explicit(list(args.event))
        chosen = []
        for event_id, path in explicit:
            try:
                rp, duration, family = _descriptor(path)
            except ValueError:
                rp = duration = None
                family = None
            chosen.append(
                build_event(
                    event_id=event_id,
                    inp_path=path,
                    rainfall_family=family,
                    return_period_code=rp,
                    duration_code=duration,
                )
            )
        basis = "USER_FROZEN_FIVE_EXISTING_DEVELOPMENT_EVENTS_NO_PROPOSED_OUTCOME_SELECTION"
    else:
        root = Path(args.candidate_dir).resolve()
        paths = [path for path in root.rglob("*.inp") if path.is_file()]
        selected_paths = _auto_select(paths)
        chosen = []
        for path in selected_paths:
            rp, duration, family = _descriptor(path)
            chosen.append(
                build_event(
                    event_id=path.stem,
                    inp_path=path,
                    rainfall_family=family,
                    return_period_code=rp,
                    duration_code=duration,
                )
            )
        basis = "FORCING_ONLY_RP_DURATION_FAMILY_STRATIFIED_EXISTING_PREPARED_EVENTS"

    payload = write_benchmark_manifest(chosen, args.out, selection_basis=basis)
    for event in payload["events"]:
        print(
            f"{event['event_id']} | RP={event['return_period_code']} | "
            f"D={event['duration_code']} | family={event['rainfall_family']} | {event['inp_path']}"
        )
    print(Path(args.out).resolve())


if __name__ == "__main__":
    main()
