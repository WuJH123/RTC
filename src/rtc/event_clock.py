from __future__ import annotations

from pathlib import Path

from .event_preparation import (
    _canonicalize_rainfall_clock,
    _raingage_series,
    _simulation_start,
)


def inspect_prepared_event_clock(inp_path: str | Path) -> dict[str, float | str]:
    """Read the effective pre-rain prefix from an already-prepared event INP.

    The helper operates on an in-memory copy of the INP text. Rainfall clock
    canonicalization therefore cannot mutate the user's file.
    """

    path = Path(inp_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"event INP is missing: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    start = _simulation_start(lines)
    rain_series, interval_minutes = _raingage_series(lines)
    first_rain, last_rain = _canonicalize_rainfall_clock(
        lines, old_start=start, rain_series=rain_series
    )
    if first_rain < start:
        raise ValueError("prepared rainfall begins before SWMM simulation START")
    effective = (first_rain - start).total_seconds() / 60.0
    return {
        "inp_path": str(path),
        "simulation_start": start.isoformat(),
        "first_positive_rainfall": first_rain.isoformat(),
        "last_positive_rainfall": last_rain.isoformat(),
        "rainfall_interval_minutes": float(interval_minutes),
        "effective_warmup_minutes": float(effective),
    }
