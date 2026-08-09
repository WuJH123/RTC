from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .units import flow_rate_to_m3s, length_to_m


@dataclass(frozen=True)
class EventMetrics:
    tfv_m3: float
    priority_flood_volume_m3: float
    global_peak_flood_rate_m3s: float
    priority_max_depth_m: dict[str, float]


def _integrate_group_rate(group: pd.DataFrame, *, flow_units: str) -> float:
    ordered = group.sort_values("elapsed_seconds")
    t = ordered["elapsed_seconds"].to_numpy(dtype=float)
    q = flow_rate_to_m3s(ordered["flooding"].to_numpy(dtype=float), flow_units)
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("node samples must have strictly increasing elapsed_seconds")
    return float((np.clip(q[:-1], 0.0, None) * dt).sum())


def compile_event_metrics(
    node_frame: pd.DataFrame,
    *,
    priority_nodes: tuple[str, ...],
    flow_units: str,
    system_units: str = "SI",
    post_action_only: bool = True,
) -> EventMetrics:
    """Derive diagnostic metrics from sampled SWMM node outputs.

    Authoritative Formal PFV/TFV should prefer cumulative SWMM node statistics. This helper
    is retained for diagnostics. Global Peak is the maximum over time of the *simultaneous
    network sum* of flooding rates, not the maximum flooding rate at one individual node.
    """

    required = {"elapsed_seconds", "node_id", "depth", "flooding"}
    missing = sorted(required - set(node_frame.columns))
    if missing:
        raise ValueError(f"node frame missing columns: {missing}")
    frame = node_frame.copy()
    if post_action_only and "phase" in frame.columns:
        frame = frame[frame["phase"] == "POST_ACTION"]
    if frame.empty:
        raise ValueError("no evaluation samples available")

    volumes = {
        str(node_id): _integrate_group_rate(group, flow_units=flow_units)
        for node_id, group in frame.groupby("node_id", sort=False)
    }
    tfv = float(sum(volumes.values()))
    priority_set = set(priority_nodes)
    priority_volume = float(sum(v for n, v in volumes.items() if n in priority_set))

    converted = frame[["elapsed_seconds", "flooding"]].copy()
    converted["flooding_m3s"] = flow_rate_to_m3s(
        converted["flooding"].to_numpy(dtype=float), flow_units
    )
    simultaneous = converted.groupby("elapsed_seconds", sort=True)["flooding_m3s"].sum()
    global_peak = float(np.clip(simultaneous.to_numpy(dtype=float), 0.0, None).max(initial=0.0))

    max_depth: dict[str, float] = {}
    for node in priority_nodes:
        subset = frame.loc[frame["node_id"].astype(str) == node, "depth"]
        max_depth[node] = (
            float(length_to_m(float(subset.max()), system_units)) if not subset.empty else float("nan")
        )
    return EventMetrics(
        tfv_m3=tfv,
        priority_flood_volume_m3=priority_volume,
        global_peak_flood_rate_m3s=global_peak,
        priority_max_depth_m=max_depth,
    )


def load_branch_node_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")
