from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import load_priority_nodes
from .pipeline import sha256_file


def _verified_lock(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != "WUHAN_RTC_POLICY_LOCK_V1":
        raise ValueError("Final requires WUHAN_RTC_POLICY_LOCK_V1")
    artefacts = payload.get("artefacts")
    hashes = payload.get("sha256")
    if not isinstance(artefacts, dict) or not isinstance(hashes, dict):
        raise ValueError("policy lock is missing artefact/hash maps")
    for name, raw_path in artefacts.items():
        p = Path(str(raw_path))
        if not p.is_file():
            raise RuntimeError(f"locked artefact disappeared: {name}: {p}")
        if sha256_file(p) != str(hashes.get(name, "")):
            raise RuntimeError(f"locked artefact changed after Policy Lock: {name}: {p}")
    return payload


def _strategy_plan(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    strategies = tuple(str(x) for x in payload.get("strategies", []))
    if not strategies or "proposed" not in strategies:
        raise ValueError("baseline plan must contain a non-empty strategies list including proposed")
    if len(set(strategies)) != len(strategies):
        raise ValueError("baseline plan contains duplicate strategy IDs")
    return strategies


def _exact_run_metrics(metadata_path: str | Path, priority_nodes: tuple[str, ...]) -> dict[str, object]:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("exact_global_peak") is not True:
        raise ValueError(f"formal Final run did not use routing-step exact global peak: {meta_path}")
    if "global_peak_flood_rate_m3s" not in meta:
        raise ValueError(f"formal Final metadata lacks global_peak_flood_rate_m3s: {meta_path}")
    stats_name = meta.get("node_statistics_file")
    if not stats_name:
        raise ValueError(f"formal Final metadata lacks node_statistics_file: {meta_path}")
    stats_path = meta_path.parent / str(stats_name)
    if not stats_path.is_file():
        raise ValueError(f"formal Final node statistics file is missing: {stats_path}")
    stats = pd.read_csv(stats_path, compression="infer")
    required = {"node_id", "delta_flooding_volume_m3", "max_depth_m"}
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ValueError(f"node statistics missing formal truth columns: {missing}")
    stats = stats.copy()
    stats["node_id"] = stats["node_id"].astype(str)
    if stats["node_id"].duplicated().any():
        raise ValueError(f"duplicate node rows in formal node statistics: {stats_path}")
    indexed = stats.set_index("node_id")
    missing_priority = sorted(set(priority_nodes) - set(indexed.index))
    if missing_priority:
        raise ValueError(f"priority nodes missing from formal node statistics: {missing_priority}")
    flood = indexed["delta_flooding_volume_m3"].astype(float).clip(lower=0.0)
    row: dict[str, object] = {
        "tfv_m3": float(flood.sum()),
        "priority_flood_volume_m3": float(flood.reindex(priority_nodes).sum()),
        "global_peak_flood_rate_m3s": float(meta["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "inp_sha256": str(meta["inp_sha256"]),
        "truth_source_flood_volume": "SWMM_NODE_STATISTICS_CUMULATIVE",
        "truth_source_global_peak": "ROUTING_STEP_SYNCHRONIZED_NETWORK_SUM",
        "metadata_sha256": sha256_file(meta_path),
        "node_statistics_sha256": sha256_file(stats_path),
    }
    for node in priority_nodes:
        row[f"priority_flood_volume_m3:{node}"] = float(flood.loc[node])
        row[f"priority_max_depth_m:{node}"] = float(indexed.loc[node, "max_depth_m"])
    return row


def compile_formal_final(
    *,
    policy_lock_path: str | Path,
    run_index_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    lock = _verified_lock(policy_lock_path)
    artefacts = lock["artefacts"]
    assert isinstance(artefacts, dict)
    for name in ("split_registry", "baseline_plan", "priority_nodes", "frozen_inp"):
        if name not in artefacts:
            raise ValueError(f"Policy Lock lacks formal Final artefact: {name}")
    priority = load_priority_nodes(str(artefacts["priority_nodes"]))
    strategies = _strategy_plan(str(artefacts["baseline_plan"]))
    frozen_inp_sha = sha256_file(str(artefacts["frozen_inp"]))

    split_registry = pd.read_csv(str(artefacts["split_registry"]))
    if not {"rainfall_group", "scientific_split"}.issubset(split_registry.columns):
        raise ValueError("locked split_registry requires rainfall_group and scientific_split")
    roles = split_registry[["rainfall_group", "scientific_split"]].drop_duplicates().copy()
    roles["rainfall_group"] = roles["rainfall_group"].astype(str)
    if roles.groupby("rainfall_group")["scientific_split"].nunique().max() != 1:
        raise ValueError("locked split_registry contains rainfall-group leakage")
    role_map = roles.set_index("rainfall_group")["scientific_split"].astype(str).to_dict()

    index = pd.read_csv(run_index_path)
    required = {"event_id", "rainfall_group", "strategy", "metadata_path"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"formal Final run index missing columns: {missing}")
    if index.duplicated(["event_id", "strategy"]).any():
        raise ValueError("formal Final contains duplicate event/strategy runs")
    wrong_groups = sorted(
        group for group in set(index["rainfall_group"].astype(str)) if role_map.get(group) != "final"
    )
    if wrong_groups:
        raise ValueError(f"formal Final contains non-final or unknown rainfall groups: {wrong_groups[:20]}")
    expected = set(strategies)
    for event, group in index.groupby("event_id", sort=False):
        present = set(group["strategy"].astype(str))
        if present != expected:
            raise ValueError(
                f"incomplete Final strategy matrix for {event}: "
                f"missing={sorted(expected-present)}, extra={sorted(present-expected)}"
            )
        if group["rainfall_group"].astype(str).nunique() != 1:
            raise ValueError(f"Final event {event} maps to multiple rainfall groups")

    detail_rows: list[dict[str, object]] = []
    for _, item in index.iterrows():
        metrics = _exact_run_metrics(str(item["metadata_path"]), priority)
        if str(metrics["inp_sha256"]) != frozen_inp_sha:
            # passive_no_rtc intentionally has a derived INP with [CONTROLS] removed.
            # Its source lineage must be declared explicitly in the run index.
            if str(item["strategy"]) != "passive_no_rtc":
                raise ValueError(
                    f"Final run INP hash differs from locked frozen INP: {item['event_id']} / {item['strategy']}"
                )
        detail_rows.append(
            {
                "event_id": str(item["event_id"]),
                "rainfall_group": str(item["rainfall_group"]),
                "strategy": str(item["strategy"]),
                **metrics,
            }
        )
    detail = pd.DataFrame(detail_rows)
    metric_cols = ["tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s", "flow_routing_error_pct"]
    summary = detail.groupby("strategy", as_index=False)[metric_cols].mean(numeric_only=True)

    pairwise: dict[str, pd.DataFrame] = {}
    proposed = detail[detail["strategy"] == "proposed"].set_index("event_id")
    for reference in strategies:
        if reference == "proposed":
            continue
        base = detail[detail["strategy"] == reference].set_index("event_id")
        if set(proposed.index) != set(base.index):
            raise ValueError(f"paired Final events differ for proposed vs {reference}")
        rows: list[dict[str, object]] = []
        for event in sorted(proposed.index):
            row: dict[str, object] = {"event_id": event, "reference": reference}
            for metric in ("tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s"):
                p = float(proposed.loc[event, metric])
                b = float(base.loc[event, metric])
                row[f"delta_{metric}"] = p - b
                row[f"reduction_{metric}_pct"] = 100.0 * (b - p) / b if abs(b) > 1e-12 else np.nan
            rows.append(row)
        pairwise[reference] = pd.DataFrame(rows)
    return detail, summary, pairwise


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile exact policy-locked untouched Final SWMM evidence")
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    detail, summary, pairwise = compile_formal_final(
        policy_lock_path=args.policy_lock,
        run_index_path=args.run_index,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail_path = out / "formal_final_detail.csv"
    summary_path = out / "formal_final_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    for reference, frame in pairwise.items():
        frame.to_csv(out / f"proposed_vs_{reference}.csv", index=False)
    print(json.dumps({
        "detail": str(detail_path),
        "summary": str(summary_path),
        "events": int(detail["event_id"].nunique()),
        "strategies": sorted(detail["strategy"].unique().tolist()),
    }, indent=2))


if __name__ == "__main__":
    main()
