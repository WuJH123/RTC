from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .code_contract import rtc_source_tree_sha256
from .pipeline import sha256_file


def read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def verify_formal_run_v4(
    manifest_path: str | Path,
    *,
    priority: tuple[str, ...],
    physical_sha: str,
    model_step_seconds: int,
    control_update_seconds: int,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    run = read_json(manifest_path)
    implementation_sha = rtc_source_tree_sha256()
    if run.get("contract") != "FORMAL_CLOSED_LOOP_RUN_MANIFEST_V4_CODE_BOUND":
        raise ValueError(f"not a Formal run V4 manifest: {manifest_path}")
    if run.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError(f"Formal run uses an incompatible implementation contract: {manifest_path}")
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

    meta = read_json(str(run["main_metadata_path"]))
    if meta.get("exact_global_peak") is not False:
        raise ValueError("main causal run must preserve fixed observation/control cadence")
    if meta.get("rtc_source_tree_sha256") is not None and meta.get(
        "rtc_source_tree_sha256"
    ) != implementation_sha:
        raise ValueError("main policy run uses an incompatible implementation contract")
    if sha256_file(str(run["decision_log_path"])) != str(run["decision_log_sha256"]):
        raise RuntimeError("decision log changed after peak replay")

    replay = read_json(str(run["peak_replay_path"]))
    if replay.get("contract") != "ROUTING_STEP_GLOBAL_PEAK_REPLAY_V3_WRITE_CADENCE_PRESERVED":
        raise ValueError(
            "formal run lacks routing-step Global Peak replay with original Python write cadence"
        )
    if replay.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError("Global Peak replay uses an incompatible implementation contract")
    if replay.get("control_write_cadence_preserved") is not True:
        raise ValueError("Global Peak replay did not preserve the original Python write cadence")
    if replay.get("engine_files_retained") is not False:
        raise ValueError("Formal Global Peak replay must not retain SWMM engine files")
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
        "global_peak_flood_rate_m3s": float(
            replay["routing_step_global_peak_flood_rate_m3s"]
        ),
        "main_flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "peak_replay_flow_routing_error_pct": float(replay["flow_routing_error_pct"]),
        "physical_network_sha256": physical_sha,
        "rtc_source_tree_sha256": implementation_sha,
        "formal_run_manifest_sha256": sha256_file(manifest_path),
        "truth_source_flood_volume": "SWMM_NODE_STATISTICS_CUMULATIVE_MAIN_RUN",
        "truth_source_global_peak": "ROUTING_STEP_FROZEN_DECISION_REPLAY_WRITE_CADENCE_PRESERVED",
    }
    for node in priority:
        result[f"priority_flood_volume_m3:{node}"] = float(flood.loc[node])
        result[f"priority_max_depth_m:{node}"] = float(table.loc[node, "max_depth_m"])
    return result
