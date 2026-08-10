from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .inp_lineage import scientific_event_contract_sha256
from .inp_runtime import sha256_file


EVENT_PREPARATION_CONTRACT = "RTC_EVENT_PREPARATION_V1_DRY_PREFIX_RECOVERY_TAIL"
_DATE_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y")
_TIME_FORMATS = ("%H:%M:%S", "%H:%M")


def _parse_date(value: str):
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unsupported SWMM date token: {value}")


def _parse_time(value: str):
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"unsupported SWMM time token: {value}")


def _section_rows(lines: list[str], section_name: str) -> list[tuple[int, str]]:
    target = section_name.strip().upper()
    section = ""
    rows: list[tuple[int, str]] = []
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            continue
        if section == target:
            rows.append((i, raw))
    return rows


def _option_values(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for _, raw in _section_rows(lines, "OPTIONS"):
        body = raw.split(";", 1)[0].strip().split()
        if len(body) >= 2:
            values[body[0].upper()] = body[1]
    return values


def _set_option(lines: list[str], key: str, value: str) -> None:
    target = key.upper()
    for i, raw in _section_rows(lines, "OPTIONS"):
        body, sep, comment = raw.partition(";")
        tokens = body.split()
        if tokens and tokens[0].upper() == target:
            newline = "\n" if raw.endswith("\n") else ""
            lines[i] = f"{key:<20}{value}" + ((" ;" + comment.rstrip("\n")) if sep else "") + newline
            return
    raise ValueError(f"[OPTIONS] lacks required field {key}")


def _simulation_start(lines: list[str]) -> datetime:
    options = _option_values(lines)
    for key in ("START_DATE", "START_TIME", "END_DATE", "END_TIME"):
        if key not in options:
            raise ValueError(f"event INP lacks [OPTIONS] {key}")
    return datetime.combine(_parse_date(options["START_DATE"]), _parse_time(options["START_TIME"]))


def _raingage_series(lines: list[str]) -> tuple[set[str], int]:
    series: set[str] = set()
    intervals: list[int] = []
    for _, raw in _section_rows(lines, "RAINGAGES"):
        tokens = raw.split(";", 1)[0].strip().split()
        if not tokens:
            continue
        upper = [x.upper() for x in tokens]
        if "TIMESERIES" not in upper:
            if "FILE" in upper:
                raise ValueError(
                    "event preparation does not rewrite external rainfall FILE clocks; "
                    "materialize the rainfall series into [TIMESERIES] or prepare the external file explicitly"
                )
            continue
        pos = upper.index("TIMESERIES")
        if pos + 1 >= len(tokens):
            raise ValueError("RAINGAGES TIMESERIES source lacks a series name")
        series.add(tokens[pos + 1])
        if len(tokens) >= 3:
            try:
                t = _parse_time(tokens[2])
                intervals.append(t.hour * 60 + t.minute + (1 if t.second else 0))
            except ValueError:
                pass
    if not series:
        raise ValueError("no TIMESERIES rainfall source found in event INP")
    positive = [x for x in intervals if x > 0]
    interval_minutes = min(positive) if positive else 5
    return series, interval_minutes


def _canonicalize_rainfall_clock(
    lines: list[str], *, old_start: datetime, rain_series: set[str]
) -> tuple[datetime, datetime]:
    """Rewrite rain series rows to explicit absolute timestamps without moving the storm."""

    first_nonzero: datetime | None = None
    last_nonzero: datetime | None = None
    last_by_series: dict[str, datetime] = {}
    for i, raw in _section_rows(lines, "TIMESERIES"):
        body, sep, comment = raw.partition(";")
        tokens = body.strip().split()
        if not tokens or tokens[0] not in rain_series:
            continue
        name = tokens[0]
        if len(tokens) == 3:
            date_value = old_start.date()
            time_value = _parse_time(tokens[1])
            value_tokens = tokens[2:]
            stamp = datetime.combine(date_value, time_value)
            previous = last_by_series.get(name)
            while previous is not None and stamp <= previous:
                stamp += timedelta(days=1)
        elif len(tokens) >= 4:
            stamp = datetime.combine(_parse_date(tokens[1]), _parse_time(tokens[2]))
            value_tokens = tokens[3:]
        else:
            raise ValueError(f"unsupported rainfall TIMESERIES row: {body.strip()}")
        last_by_series[name] = stamp
        try:
            value = float(value_tokens[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"rainfall TIMESERIES value is not numeric: {body.strip()}") from exc
        if value > 0:
            first_nonzero = stamp if first_nonzero is None else min(first_nonzero, stamp)
            last_nonzero = stamp if last_nonzero is None else max(last_nonzero, stamp)
        newline = "\n" if raw.endswith("\n") else ""
        rebuilt = (
            f"{name:<20}{stamp.strftime('%m/%d/%Y')} {stamp.strftime('%H:%M:%S')} "
            + " ".join(value_tokens)
        )
        if sep:
            rebuilt += " ;" + comment.rstrip("\n")
        lines[i] = rebuilt + newline
    if first_nonzero is None or last_nonzero is None:
        raise ValueError("rainfall TIMESERIES contains no positive rainfall values")
    return first_nonzero, last_nonzero


def prepare_event_inp(
    source: str | Path,
    destination: str | Path,
    *,
    warmup_minutes: int,
    post_rain_tail_minutes: int,
) -> dict[str, object]:
    """Add an antecedent dry/DWF prefix and a recovery tail without changing storm clock/shape.

    Rainfall rows are canonicalized to explicit absolute dates/times anchored to the original
    event clock. The simulation start is moved earlier by ``warmup_minutes`` so the same storm
    arrives after a causal dry/DWF history. The simulation end is moved to the end of the final
    non-zero rainfall interval plus ``post_rain_tail_minutes``. DWF patterns, rainfall intensities,
    hydraulic geometry, initial device definitions and any existing policy section are otherwise
    left untouched.
    """

    if warmup_minutes <= 0:
        raise ValueError("warmup_minutes must be positive")
    if post_rain_tail_minutes <= 0:
        raise ValueError("post_rain_tail_minutes must be positive")
    src = Path(source).resolve()
    dst = Path(destination)
    if not src.is_file():
        raise ValueError(f"event INP is missing: {src}")
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    old_start = _simulation_start(lines)
    rain_series, rain_interval_minutes = _raingage_series(lines)
    first_rain, last_rain = _canonicalize_rainfall_clock(
        lines, old_start=old_start, rain_series=rain_series
    )
    if first_rain < old_start:
        raise ValueError("rainfall begins before the original event simulation start")

    new_start = old_start - timedelta(minutes=int(warmup_minutes))
    rain_end = last_rain + timedelta(minutes=rain_interval_minutes)
    new_end = rain_end + timedelta(minutes=int(post_rain_tail_minutes))
    _set_option(lines, "START_DATE", new_start.strftime("%m/%d/%Y"))
    _set_option(lines, "START_TIME", new_start.strftime("%H:%M:%S"))
    _set_option(lines, "REPORT_START_DATE", new_start.strftime("%m/%d/%Y"))
    _set_option(lines, "REPORT_START_TIME", new_start.strftime("%H:%M:%S"))
    _set_option(lines, "END_DATE", new_end.strftime("%m/%d/%Y"))
    _set_option(lines, "END_TIME", new_end.strftime("%H:%M:%S"))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(lines), encoding="utf-8")
    onset_elapsed = (first_rain - new_start).total_seconds() / 60.0
    end_elapsed = (rain_end - new_start).total_seconds() / 60.0
    return {
        "contract": EVENT_PREPARATION_CONTRACT,
        "source_inp": str(src),
        "source_inp_sha256": sha256_file(src),
        "prepared_inp": str(dst.resolve()),
        "prepared_inp_sha256": sha256_file(dst),
        "source_scientific_event_sha256": scientific_event_contract_sha256(src),
        "prepared_scientific_event_sha256": scientific_event_contract_sha256(dst),
        "warmup_minutes": int(warmup_minutes),
        "post_rain_tail_minutes": int(post_rain_tail_minutes),
        "rainfall_onset_elapsed_minutes": float(onset_elapsed),
        "rainfall_end_elapsed_minutes": float(end_elapsed),
        "rainfall_interval_minutes": int(rain_interval_minutes),
        "storm_absolute_clock_preserved": True,
        "dwf_clock_phase_at_storm_preserved": True,
    }


def prepare_event_registry(
    events: pd.DataFrame,
    *,
    output_dir: str | Path,
    warmup_minutes: int,
    post_rain_tail_minutes: int,
) -> pd.DataFrame:
    required = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"event registry lacks required columns: {missing}")
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("event registry must contain one row per event_id")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, row in events.iterrows():
        event_id = str(row["event_id"])
        dest = out_dir / f"{event_id}.inp"
        evidence = prepare_event_inp(
            str(row["inp_path"]),
            dest,
            warmup_minutes=warmup_minutes,
            post_rain_tail_minutes=post_rain_tail_minutes,
        )
        result = row.to_dict()
        result["source_inp_path"] = str(Path(str(row["inp_path"])).resolve())
        result["inp_path"] = str(dest.resolve())
        result["event_preparation_contract"] = EVENT_PREPARATION_CONTRACT
        result["source_inp_sha256"] = evidence["source_inp_sha256"]
        result["prepared_inp_sha256"] = evidence["prepared_inp_sha256"]
        result["pre_rain_warmup_minutes"] = evidence["rainfall_onset_elapsed_minutes"]
        result["post_rain_tail_minutes"] = int(post_rain_tail_minutes)
        result["rainfall_onset_elapsed_minutes"] = evidence["rainfall_onset_elapsed_minutes"]
        result["rainfall_end_elapsed_minutes"] = evidence["rainfall_end_elapsed_minutes"]
        result["prepared_scientific_event_sha256"] = evidence[
            "prepared_scientific_event_sha256"
        ]
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare event INPs with a causal dry/DWF prefix and explicit recovery tail"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-registry", required=True)
    parser.add_argument("--warmup-minutes", type=int, default=60)
    parser.add_argument("--post-rain-tail-minutes", type=int, default=360)
    args = parser.parse_args()
    prepared = prepare_event_registry(
        pd.read_csv(args.events),
        output_dir=args.out_dir,
        warmup_minutes=args.warmup_minutes,
        post_rain_tail_minutes=args.post_rain_tail_minutes,
    )
    out = Path(args.out_registry)
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out, index=False)
    summary = {
        "contract": EVENT_PREPARATION_CONTRACT,
        "events": int(len(prepared)),
        "rainfall_groups": int(prepared["rainfall_group"].astype(str).nunique()),
        "warmup_minutes": int(args.warmup_minutes),
        "post_rain_tail_minutes": int(args.post_rain_tail_minutes),
        "storm_absolute_clock_preserved": True,
        "dwf_clock_phase_at_storm_preserved": True,
        "output_registry": str(out.resolve()),
        "output_registry_sha256": sha256_file(out),
    }
    out.with_suffix(out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
