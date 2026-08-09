from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import load_priority_nodes
from .inp_lineage import physical_contract_sha256
from .pipeline import sha256_file


def _verified_lock(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != "WUHAN_RTC_POLICY_LOCK_V1":
        raise ValueError("Final requires WUHAN_RTC_POLICY_LOCK_V1")
    if payload.get("formal_contract") != "WUHAN_RTC_FORMAL_POLICY_LOCK_V2":
        raise ValueError("Final requires the strict WUHAN_RTC_FORMAL_POLICY_LOCK_V2 wrapper")
    artefacts, hashes = payload.get("artefacts"), payload.get("sha256")
    if not isinstance(artefacts, dict) or not isinstance(hashes, dict):
        raise ValueError("policy lock is missing artefact/hash maps")
    for name, raw_path in artefacts.items():
        p = Path(str(raw_path))
        if not p.is_file():
            raise RuntimeError(f"locked artefact disappeared: {name}: {p}")
        if sha256_file(p) != str(hashes.get(name, "")):
            raise RuntimeError(f"locked artefact changed after Policy Lock: {name}: {p}")
    return payload


def _strategies(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = tuple(str(x) for x in payload.get("strategies", []))
    if not values or "proposed" not in values or len(set(values)) != len(values):
        raise ValueError("baseline plan must contain unique strategy IDs including proposed")
    return values


def _exact_metrics(metadata_path: str | Path, priority: tuple[str, ...]) -> dict[str, object]:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("exact_global_peak") is not True:
        raise ValueError(f"formal Final requires exact_global_peak=true: {meta_path}")
    stats_name = meta.get("node_statistics_file")
    if not stats_name:
        raise ValueError(f"formal Final requires node_statistics_file: {meta_path}")
    stats_path = meta_path.parent / str(stats_name)
    stats = pd.read_csv(stats_path, compression="infer")
    required = {"node_id", "delta_flooding_volume_m3", "max_depth_m"}
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ValueError(f"formal node statistics missing columns: {missing}")
    stats = stats.copy()
    stats["node_id"] = stats["node_id"].astype(str)
    if stats["node_id"].duplicated().any():
        raise ValueError(f"duplicate node statistics rows: {stats_path}")
    table = stats.set_index("node_id")
    missing_priority = sorted(set(priority) - set(table.index))
    if missing_priority:
        raise ValueError(f"priority nodes missing from exact SWMM statistics: {missing_priority}")
    flood = table["delta_flooding_volume_m3"].astype(float).clip(lower=0.0)
    inp_path = Path(str(meta["inp_path"]))
    if not inp_path.is_file():
        raise ValueError(f"run INP disappeared; cannot verify physical lineage: {inp_path}")
    row: dict[str, object] = {
        "tfv_m3": float(flood.sum()),
        "priority_flood_volume_m3": float(flood.reindex(priority).sum()),
        "global_peak_flood_rate_m3s": float(meta["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "full_inp_sha256": str(meta["inp_sha256"]),
        "physical_contract_sha256": physical_contract_sha256(inp_path),
        "metadata_sha256": sha256_file(meta_path),
        "node_statistics_sha256": sha256_file(stats_path),
        "truth_source_flood_volume": "SWMM_NODE_STATISTICS_CUMULATIVE",
        "truth_source_global_peak": "ROUTING_STEP_SYNCHRONIZED_NETWORK_SUM",
    }
    for node in priority:
        row[f"priority_flood_volume_m3:{node}"] = float(flood.loc[node])
        row[f"priority_max_depth_m:{node}"] = float(table.loc[node, "max_depth_m"])
    return row


def compile_formal_final_v2(
    *, policy_lock_path: str | Path, run_index_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    lock = _verified_lock(policy_lock_path)
    artefacts = lock["artefacts"]
    assert isinstance(artefacts, dict)
    for name in ("frozen_inp", "priority_nodes", "split_registry", "baseline_plan"):
        if name not in artefacts:
            raise ValueError(f"strict Policy Lock lacks {name}")
    physical_sha = physical_contract_sha256(str(artefacts["frozen_inp"]))
    priority = load_priority_nodes(str(artefacts["priority_nodes"]))
    strategies = _strategies(str(artefacts["baseline_plan"]))

    split = pd.read_csv(str(artefacts["split_registry"]))
    if not {"rainfall_group", "scientific_split"}.issubset(split.columns):
        raise ValueError("locked split registry lacks rainfall_group/scientific_split")
    roles = split[["rainfall_group", "scientific_split"]].drop_duplicates().copy()
    roles["rainfall_group"] = roles["rainfall_group"].astype(str)
    if roles.groupby("rainfall_group")["scientific_split"].nunique().max() != 1:
        raise ValueError("rainfall-group leakage exists in the locked split registry")
    role_map = roles.set_index("rainfall_group")["scientific_split"].astype(str).to_dict()

    index = pd.read_csv(run_index_path)
    required = {"event_id", "rainfall_group", "strategy", "metadata_path"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Final run index missing columns: {missing}")
    if index.duplicated(["event_id", "strategy"]).any():
        raise ValueError("duplicate event/strategy Final run")
    wrong = sorted(g for g in set(index["rainfall_group"].astype(str)) if role_map.get(g) != "final")
    if wrong:
        raise ValueError(f"Final contains non-final/unknown rainfall groups: {wrong[:20]}")
    expected = set(strategies)
    for event, group in index.groupby("event_id", sort=False):
        present = set(group["strategy"].astype(str))
        if present != expected:
            raise ValueError(
                f"incomplete Final matrix at {event}: missing={sorted(expected-present)}, extra={sorted(present-expected)}"
            )
        if group["rainfall_group"].astype(str).nunique() != 1:
            raise ValueError(f"Final event {event} maps to multiple rainfall groups")

    records: list[dict[str, object]] = []
    for _, run in index.iterrows():
        metrics = _exact_metrics(str(run["metadata_path"]), priority)
        if metrics["physical_contract_sha256"] != physical_sha:
            raise ValueError(
                f"physical network changed in Final: {run['event_id']} / {run['strategy']}"
            )
        records.append(
            {
                "event_id": str(run["event_id"]),
                "rainfall_group": str(run["rainfall_group"]),
                "strategy": str(run["strategy"]),
                **metrics,
            }
        )
    detail = pd.DataFrame(records)
    metrics = ["tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s", "flow_routing_error_pct"]
    summary = detail.groupby("strategy", as_index=False)[metrics].mean(numeric_only=True)

    pairs: dict[str, pd.DataFrame] = {}
    proposed = detail[detail["strategy"] == "proposed"].set_index("event_id")
    for reference in strategies:
        if reference == "proposed":
            continue
        base = detail[detail["strategy"] == reference].set_index("event_id")
        if set(proposed.index) != set(base.index):
            raise ValueError(f"unpaired Final events for proposed vs {reference}")
        rows: list[dict[str, object]] = []
        for event in sorted(proposed.index):
            row: dict[str, object] = {"event_id": event, "reference": reference}
            for metric in ("tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s"):
                p, b = float(proposed.loc[event, metric]), float(base.loc[event, metric])
                row[f"delta_{metric}"] = p - b
                row[f"reduction_{metric}_pct"] = 100.0 * (b - p) / b if abs(b) > 1e-12 else np.nan
            rows.append(row)
        pairs[reference] = pd.DataFrame(rows)
    return detail, summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile exact, physical-lineage-verified untouched Final")
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    detail, summary, pairs = compile_formal_final_v2(
        policy_lock_path=args.policy_lock, run_index_path=args.run_index
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "formal_final_detail.csv", index=False)
    summary.to_csv(out / "formal_final_summary.csv", index=False)
    for reference, frame in pairs.items():
        frame.to_csv(out / f"proposed_vs_{reference}.csv", index=False)
    print(json.dumps({
        "events": int(detail["event_id"].nunique()),
        "strategies": sorted(detail["strategy"].unique().tolist()),
        "detail": str(out / "formal_final_detail.csv"),
        "summary": str(out / "formal_final_summary.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
