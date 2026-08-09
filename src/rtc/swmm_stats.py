from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Mapping

from .units import flow_rate_to_m3s, length_to_m, volume_to_m3


_SELECTED = (
    "flooding_volume",
    "peak_flooding_rate",
    "max_depth",
    "flooding_duration",
    "surcharge_duration",
)


def snapshot_node_statistics(node_objects: Mapping[str, object]) -> dict[str, dict[str, float]]:
    """Read the current cumulative SWMM node statistics exposed by PySWMM."""

    result: dict[str, dict[str, float]] = {}
    for node_id, obj in node_objects.items():
        raw = obj.statistics
        result[str(node_id)] = {name: float(raw.get(name, 0.0)) for name in _SELECTED}
    return result


def write_node_statistics(
    path: str | Path,
    *,
    end_statistics: Mapping[str, Mapping[str, float]],
    system_units: str,
    flow_units: str,
    start_statistics: Mapping[str, Mapping[str, float]] | None = None,
) -> Path:
    """Write cumulative/delta node statistics with explicit SI conversions.

    ``start_statistics`` is useful for D2/D3 branches: subtracting the identical causal
    prefix makes post-action flooding volume exact without numerically integrating sampled
    rates. For a full-event run, omit it and cumulative values are measured from zero.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "node_id",
                "start_flooding_volume_native",
                "end_flooding_volume_native",
                "delta_flooding_volume_native",
                "delta_flooding_volume_m3",
                "peak_flooding_rate_native",
                "peak_flooding_rate_m3s",
                "max_depth_native",
                "max_depth_m",
                "flooding_duration",
                "surcharge_duration",
                "system_units",
                "flow_units",
            ]
        )
        for node_id in sorted(end_statistics):
            end = end_statistics[node_id]
            start = start_statistics.get(node_id, {}) if start_statistics is not None else {}
            start_volume = float(start.get("flooding_volume", 0.0))
            end_volume = float(end.get("flooding_volume", 0.0))
            delta_volume = max(0.0, end_volume - start_volume)
            writer.writerow(
                [
                    node_id,
                    start_volume,
                    end_volume,
                    delta_volume,
                    float(volume_to_m3(delta_volume, system_units)),
                    float(end.get("peak_flooding_rate", 0.0)),
                    float(flow_rate_to_m3s(end.get("peak_flooding_rate", 0.0), flow_units)),
                    float(end.get("max_depth", 0.0)),
                    float(length_to_m(end.get("max_depth", 0.0), system_units)),
                    float(end.get("flooding_duration", 0.0)),
                    float(end.get("surcharge_duration", 0.0)),
                    system_units,
                    flow_units,
                ]
            )
    return out
