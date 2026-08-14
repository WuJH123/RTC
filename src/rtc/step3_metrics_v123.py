"""Authoritative V12.3 reporting metrics derived from SWMM node statistics."""
from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path
from typing import Sequence

V123_PFV_REPORT_CONTRACT = "PROJECT7_V123_PRIORITY_PFV_FROM_SWMM_NODE_STATISTICS_V1"


def priority_flood_volume_from_node_statistics_v123(
    path: str | Path,
    *,
    priority_nodes: Sequence[str],
) -> float:
    """Sum exact full-event SWMM flooding volume over the frozen priority-node set.

    ``write_node_statistics`` already converts cumulative/delta flooding volume to SI in
    ``delta_flooding_volume_m3``.  This helper deliberately consumes that authoritative
    column instead of re-integrating sampled flooding rates.
    """
    priority = tuple(str(node) for node in priority_nodes)
    if not priority or len(set(priority)) != len(priority):
        raise ValueError("V123 PFV reporting requires a non-empty unique priority list")
    wanted = set(priority)
    found: dict[str, float] = {}
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"node_id", "delta_flooding_volume_m3"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("V123 node-statistics file lacks PFV columns")
        for row in reader:
            node = str(row["node_id"])
            if node not in wanted:
                continue
            if node in found:
                raise ValueError(f"duplicate priority node in statistics: {node}")
            value = float(row["delta_flooding_volume_m3"])
            if not math.isfinite(value) or value < -1.0e-8:
                raise ValueError(f"invalid flooding volume for priority node {node}")
            found[node] = max(value, 0.0)
    missing = [node for node in priority if node not in found]
    if missing:
        raise ValueError(f"priority nodes missing from SWMM statistics: {missing}")
    return float(sum(found[node] for node in priority))


__all__ = [
    "V123_PFV_REPORT_CONTRACT",
    "priority_flood_volume_from_node_statistics_v123",
]
