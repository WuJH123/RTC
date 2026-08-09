from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import load_priority_nodes
from .inp_lineage import physical_contract_sha256
from .pipeline import sha256_file


def _json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _verified_lock(path: str | Path) -> dict[str, object]:
    payload = _json(path)
    if payload.get("contract") != "WUHAN_RTC_POLICY_LOCK_V1":
        raise ValueError("Formal Final requires WUHAN_RTC_POLICY_LOCK_V1")
    if payload.get("formal_contract") != "WUHAN_RTC_FORMAL_POLICY_LOCK_V2":
        raise ValueError("Formal Final requires the strict formal Policy Lock wrapper")
    artefacts, hashes = payload.get("artefacts"), payload.get("sha256")
    if not isinstance(artefacts, dict) or not isinstance(hashes, dict):
        raise ValueError("Policy Lock lacks artefact/hash maps")
    for name, raw in artefacts.items():
        p = Path(str(raw))
        if not p.is_file():
            raise RuntimeError(f"locked artefact disappeared: {name}: {p}")
        if sha256_file(p) != str(hashes.get(name, "")):
            raise RuntimeError(f"locked artefact changed after Policy Lock: {name}: {p}")
    return payload


def _strategies(path: str | Path) -> tuple[str, ...]:
    payload = _json(path)
    values = tuple(str(x) for x in payload.get("strategies", []))
    if not values or "proposed" not in values or len(set(values)) != len(values):
        raise ValueError("baseline plan must contain unique strategy IDs including proposed")
    return values


def _verify_formal_run(
    manifest_path: str | Path,
    *,
    priority: tuple[str, ...],
    physical_sha: str,
    model_step_seconds: int,
    control_update_seconds: int,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    run = _json(manifest_path)
    if run.get("contract") != "FORMAL_CLOSED_LOOP_RUN_MANIFEST_V3":
        raise ValueError(f"not a formal run V3 manifest: {manifest_path}")
    if str(run.get("physical_network_sha256", "")) != physical_sha:
        raise ValueError(f"physical network changed in run: {manifest_path}")
    if int(run.get("model_step_seconds", -1)) != model_step_seconds:
        raise ValueError(f"model-step cadence differs from Policy Lock: {manifest_path}")
    if int(run.get("control_update_seconds", -1)) != control_update_seconds:
        raise ValueError(f"control-update cadence differs from Policy Lock: {manifest_path}")

    bound = {
        "main_metadata_path": "main_metadata_sha256",
        "node_statistics_path": "node_statistics_sha256",
        "decision_log_path": "decision_log_sha256",
        "peak_replay_path": "peak_replay_sha256",
    }
    for path_key, hash_key in bound.items():
        p = Path(str(run[path_key]))
        if not p.is_file() or sha256_file(p) != str(run[hash_key]):
            raise RuntimeError(f"formal run evidence changed: {path_key}: {p}")

    meta = _json(str(run["main_metadata_path"]))
    if meta.get("exact_global_peak") is not False:
        raise ValueError("main causal run must preserve fixed observation/control cadence")
    if sha256_file(str(run["decision_log_path"])) != str(run["decision_log_sha256"]):
        raise RuntimeError("decision log changed after peak replay")
    replay = _json(str(run["peak_replay_path"]))
    if replay.get("contract") != "ROUTING_STEP_GLOBAL_PEAK_REPLAY_V1":
        raise ValueError("formal run lacks a routing-step peak replay")
    if str(replay.get("source_main_metadata_sha256")) != str(run["main_metadata_sha256"]):
        raise ValueError("peak replay is bound to a different main run")
    if str(replay.get("decision_log_sha256")) != str(run["decision_log_sha256"]):
        raise ValueError("peak replay is bound to a different decision schedule")

    stats = pd.read_csv(str(run["node_statistics_path"]), compression="infer")
    required = {"node_id", "delta_flooding_volume_m3", "max_depth_m"}
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ValueError(f"formal node statistics missing columns: {missing}")
    stats = stats.copy()
    stats["node_id"] = stats["node_id"].astype(str)
    if stats["node_id"].duplicated().any():
        raise ValueError("formal node statistics contain duplicate nodes")
    table = stats.set_index("node_id")
    missing_priority = sorted(set(priority) - set(table.index))
    if missing_priority:
        raise ValueError(f"priority nodes missing from formal run: {missing_priority}")
    flood = table["delta_flooding_volume_m3"].astype(float).clip(lower=0.0)
    result: dict[str, object] = {
        "event_id": str(run["event_id"]),
        "rainfall_group": str(run["rainfall_group"]),
        "strategy": str(run["strategy"]),
        "tfv_m3": float(flood.sum()),
        "priority_flood_volume_m3": float(flood.reindex(priority).sum()),
        "global_peak_flood_rate_m3s": float(replay["routing_step_global_peak_flood_rate_m3s"]),
        "main_flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "peak_replay_flow_routing_error_pct": float(replay["flow_routing_error_pct"]),
        "physical_network_sha256": physical_sha,
        "formal_run_manifest_sha256": sha256_file(manifest_path),
        "truth_source_flood_volume": "SWMM_NODE_STATISTICS_CUMULATIVE_MAIN_RUN",
        "truth_source_global_peak": "ROUTING_STEP_FROZEN_DECISION_REPLAY",
    }
    for node in priority:
        result[f"priority_flood_volume_m3:{node}"] = float(flood.loc[node])
        result[f"priority_max_depth_m:{node}"] = float(table.loc[node, "max_depth_m"])
    return result


def compile_final_v3(
    *, policy_lock_path: str | Path, run_index_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    lock = _verified_lock(policy_lock_path)
    artefacts = lock["artefacts"]
    assert isinstance(artefacts, dict)
    for name in ("frozen_inp", "priority_nodes", "split_registry", "baseline_plan", "controller_config"):
        if name not in artefacts:
            raise ValueError(f"Policy Lock lacks {name}")
    physical_sha = physical_contract_sha256(str(artefacts["frozen_inp"]))
    priority = load_priority_nodes(str(artefacts["priority_nodes"]))
    strategies = _strategies(str(artefacts["baseline_plan"]))
    controller = _json(str(artefacts["controller_config"]))
    model_step = int(controller["model_step_seconds"])
    control_update = int(controller["control_update_seconds"])

    split = pd.read_csv(str(artefacts["split_registry"]))
    if not {"rainfall_group", "scientific_split"}.issubset(split.columns):
        raise ValueError("locked split registry lacks rainfall_group/scientific_split")
    roles = split[["rainfall_group", "scientific_split"]].drop_duplicates().copy()
    roles["rainfall_group"] = roles["rainfall_group"].astype(str)
    if roles.groupby("rainfall_group")["scientific_split"].nunique().max() != 1:
        raise ValueError("locked split registry contains rainfall-group leakage")
    role_map = roles.set_index("rainfall_group")["scientific_split"].astype(str).to_dict()

    index = pd.read_csv(run_index_path)
    required = {"event_id", "rainfall_group", "strategy", "formal_manifest_path"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Final run index missing columns: {missing}")
    if index.duplicated(["event_id", "strategy"]).any():
        raise ValueError("duplicate event/strategy Formal Final run")
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

    rows: list[dict[str, object]] = []
    for _, item in index.iterrows():
        result = _verify_formal_run(
            str(item["formal_manifest_path"]),
            priority=priority,
            physical_sha=physical_sha,
            model_step_seconds=model_step,
            control_update_seconds=control_update,
        )
        if result["event_id"] != str(item["event_id"]):
            raise ValueError("Final index event_id differs from bound formal run manifest")
        if result["rainfall_group"] != str(item["rainfall_group"]):
            raise ValueError("Final index rainfall_group differs from bound formal run manifest")
        if result["strategy"] != str(item["strategy"]):
            raise ValueError("Final index strategy differs from bound formal run manifest")
        rows.append(result)
    detail = pd.DataFrame(rows)
    metric_cols = [
        "tfv_m3",
        "priority_flood_volume_m3",
        "global_peak_flood_rate_m3s",
        "main_flow_routing_error_pct",
        "peak_replay_flow_routing_error_pct",
    ]
    summary = detail.groupby("strategy", as_index=False)[metric_cols].mean(numeric_only=True)

    pairs: dict[str, pd.DataFrame] = {}
    proposed = detail[detail["strategy"] == "proposed"].set_index("event_id")
    for reference in strategies:
        if reference == "proposed":
            continue
        base = detail[detail["strategy"] == reference].set_index("event_id")
        if set(proposed.index) != set(base.index):
            raise ValueError(f"unpaired Final events for proposed vs {reference}")
        pair_rows: list[dict[str, object]] = []
        for event in sorted(proposed.index):
            row: dict[str, object] = {"event_id": event, "reference": reference}
            for metric in ("tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s"):
                p, b = float(proposed.loc[event, metric]), float(base.loc[event, metric])
                row[f"delta_{metric}"] = p - b
                row[f"reduction_{metric}_pct"] = 100.0 * (b - p) / b if abs(b) > 1e-12 else np.nan
            pair_rows.append(row)
        pairs[reference] = pd.DataFrame(pair_rows)
    return detail, summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile cadence-preserving exact Formal Final V3")
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    detail, summary, pairs = compile_final_v3(
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
