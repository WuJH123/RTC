from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import compile_event_metrics


def verify_untouched_final_groups(
    final_groups: Iterable[str],
    *,
    development_groups: Iterable[str],
    calibration_groups: Iterable[str],
    safety_audit_groups: Iterable[str],
) -> None:
    final = set(map(str, final_groups))
    used = set(map(str, development_groups)) | set(map(str, calibration_groups)) | set(map(str, safety_audit_groups))
    overlap = sorted(final & used)
    if overlap:
        raise ValueError(f"final rainfall groups were previously touched: {overlap[:20]}")


def compile_closed_loop_run_index(
    run_index: pd.DataFrame,
    *,
    priority_nodes: tuple[str, ...],
) -> pd.DataFrame:
    """Compile event-strategy KPI rows only from authoritative closed-loop SWMM files."""

    required = {"event_id", "rainfall_group", "strategy", "metadata_path"}
    missing = sorted(required - set(run_index.columns))
    if missing:
        raise ValueError(f"run index missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for _, item in run_index.iterrows():
        meta_path = Path(str(item["metadata_path"]))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        node_path = meta_path.parent / str(meta["node_file"])
        node = pd.read_csv(node_path, compression="infer")
        metrics = compile_event_metrics(
            node,
            priority_nodes=priority_nodes,
            flow_units=str(meta["flow_units"]),
            post_action_only=False,
        )
        row: dict[str, object] = {
            "event_id": str(item["event_id"]),
            "rainfall_group": str(item["rainfall_group"]),
            "strategy": str(item["strategy"]),
            "tfv_m3": metrics.tfv_m3,
            "priority_flood_volume_m3": metrics.priority_flood_volume_m3,
            "global_peak_flood_rate_m3s": metrics.global_peak_flood_rate_m3s,
            "inp_sha256": str(meta["inp_sha256"]),
            "flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        }
        for node_id, depth in metrics.priority_max_depth_m.items():
            row[f"priority_max_depth_m:{node_id}"] = depth
        rows.append(row)
    detail = pd.DataFrame.from_records(rows)
    duplicate = detail.duplicated(["event_id", "strategy"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate event/strategy closed-loop runs in final index")
    return detail


def event_balanced_summary(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "tfv_m3",
        "priority_flood_volume_m3",
        "global_peak_flood_rate_m3s",
        "flow_routing_error_pct",
    ]
    missing = sorted(set(metrics + ["strategy", "event_id"]) - set(detail.columns))
    if missing:
        raise ValueError(f"detail missing columns: {missing}")
    return detail.groupby("strategy", as_index=False)[metrics].mean(numeric_only=True)


def paired_strategy_comparison(
    detail: pd.DataFrame,
    *,
    proposed: str = "proposed",
    reference: str = "native_rules",
) -> pd.DataFrame:
    """Return paired event-level deltas; negative flood-volume delta favours proposed."""

    metrics = ["tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s"]
    left = detail[detail["strategy"] == proposed].set_index("event_id")
    right = detail[detail["strategy"] == reference].set_index("event_id")
    common = sorted(set(left.index) & set(right.index))
    if not common:
        raise ValueError(f"no paired events for {proposed} vs {reference}")
    rows: list[dict[str, object]] = []
    for event in common:
        row: dict[str, object] = {"event_id": event, "proposed": proposed, "reference": reference}
        for metric in metrics:
            row[f"delta_{metric}"] = float(left.loc[event, metric] - right.loc[event, metric])
            denom = float(right.loc[event, metric])
            row[f"reduction_{metric}_pct"] = float(100.0 * (right.loc[event, metric] - left.loc[event, metric]) / denom) if abs(denom) > 1e-12 else float("nan")
        rows.append(row)
    return pd.DataFrame.from_records(rows)
