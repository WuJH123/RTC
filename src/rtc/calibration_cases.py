from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .contracts import load_priority_nodes
from .dataset_compile import compile_branch_tensors
from .production_cli import _load_graph, _load_step2
from .units import length_to_m


def _join(manifest: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_action_sha256"]
    for key in ("event_id", "checkpoint_id", "checkpoint_minutes"):
        if key in manifest.columns and key in runs.columns:
            keys.append(key)
    merged = manifest.merge(runs, on=keys, how="inner", suffixes=("", "_run"))
    if merged.empty:
        raise ValueError(f"D2 manifest/run summary have no matches on {keys}")
    return merged


def _exact_flood_volume(metadata_path: str | Path, node_ids: tuple[str, ...]) -> np.ndarray:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats_name = meta.get("node_statistics_file")
    if not stats_name:
        raise ValueError(f"calibration requires exact node_statistics_file: {meta_path}")
    stats = pd.read_csv(meta_path.parent / str(stats_name), compression="infer")
    table = stats.assign(node_id=stats["node_id"].astype(str)).set_index("node_id")
    missing = [n for n in node_ids if n not in table.index]
    if missing:
        raise ValueError(f"node statistics missing nodes: {missing[:20]}")
    return table.reindex(node_ids)["delta_flooding_volume_m3"].to_numpy(dtype=float)


def _post_action_max_depth(metadata_path: str | Path, node_ids: tuple[str, ...]) -> np.ndarray:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    node = pd.read_csv(meta_path.parent / str(meta["node_file"]), compression="infer")
    node = node[node["phase"].astype(str) != "PRE_ACTION_CHECKPOINT"].copy()
    if node.empty:
        raise ValueError(f"branch has no post-action node samples: {meta_path}")
    node["node_id"] = node["node_id"].astype(str)
    table = node.groupby("node_id")["depth"].max()
    missing = [n for n in node_ids if n not in table.index]
    if missing:
        raise ValueError(f"post-action depth truth missing nodes: {missing[:20]}")
    native = table.reindex(node_ids).to_numpy(dtype=float)
    return length_to_m(native, str(meta["system_units"]))


def _predict(model, graph, metadata_path: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    branch = compile_branch_tensors(metadata_path)
    if branch.node_ids != graph.node_ids or branch.actuator_ids != graph.actuator_ids:
        raise ValueError("calibration branch schema differs from graph schema")
    dt = np.diff(branch.elapsed_seconds).astype(np.float32)
    if np.any(dt <= 0):
        raise ValueError("branch time grid must be strictly increasing")
    with torch.no_grad():
        rollout = model.rollout(
            torch.as_tensor(branch.initial_state[None], dtype=torch.float32, device=device),
            torch.as_tensor(branch.rainfall[None], dtype=torch.float32, device=device),
            torch.as_tensor(branch.settings[None], dtype=torch.float32, device=device),
            torch.as_tensor(branch.previous_actuator_flow[None], dtype=torch.float32, device=device),
            torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
            torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
            torch.as_tensor(graph.actuator_physics[None], dtype=torch.float32, device=device),
            torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
            torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        )
        states = rollout.states[0]
        rate = states[..., 2].clamp_min(0.0)
        volume = (
            rate * torch.as_tensor(dt, dtype=torch.float32, device=device).view(-1, 1)
        ).sum(dim=0)
        max_depth = states[..., 0].amax(dim=0)
    return volume.cpu().numpy(), max_depth.cpu().numpy()


def build_calibration_cases(
    *,
    manifest_path: str | Path,
    run_summary_path: str | Path,
    graph_path: str | Path,
    step2_path: str | Path,
    priority_path: str | Path,
    device: str | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    runs = pd.read_csv(run_summary_path)
    merged = _join(manifest, runs)
    if "scientific_split" not in merged.columns:
        raise ValueError("D2 manifest must carry scientific_split lineage")
    merged = merged[merged["scientific_split"].astype(str) == "calibration"].copy()
    if merged.empty:
        raise ValueError("no D2 calibration branches after split filtering")
    if "rainfall_group" not in merged.columns:
        raise ValueError("calibration D2 manifest requires rainfall_group")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(graph_path)
    model = _load_step2(step2_path, dev)
    priority = load_priority_nodes(priority_path)
    missing = sorted(set(priority) - set(graph.node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from graph: {missing}")
    pidx = np.asarray([graph.node_ids.index(node) for node in priority], dtype=int)

    # Build one unique run record per checkpoint/action. D2 center actions are intentionally
    # repeated in the design for each probed actuator, but they represent one physical branch.
    action_keys = ["checkpoint_id", "candidate_action_sha256"]
    if "event_id" in merged.columns:
        action_keys.insert(0, "event_id")
    unique = merged.drop_duplicates(action_keys).copy()
    run_by_sha: dict[tuple[str, str], pd.Series] = {}
    for _, row in unique.iterrows():
        scope = str(row.get("event_id", "")) + "|" + str(row["checkpoint_id"])
        run_by_sha[(scope, str(row["candidate_action_sha256"]))] = row

    pred_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    flood_cache: dict[str, np.ndarray] = {}
    depth_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for _, row in unique.iterrows():
        candidate_sha = str(row["candidate_action_sha256"])
        base_sha = str(row.get("base_action_sha256", ""))
        if not base_sha or candidate_sha == base_sha:
            continue
        scope = str(row.get("event_id", "")) + "|" + str(row["checkpoint_id"])
        base_row = run_by_sha.get((scope, base_sha))
        if base_row is None:
            raise ValueError(f"missing D2 base-action branch for {scope}: {base_sha[:12]}")
        candidate_path = str(row["metadata_path"])
        base_path = str(base_row["metadata_path"])
        for path in (candidate_path, base_path):
            if path not in pred_cache:
                pred_cache[path] = _predict(model, graph, path, dev)
            if path not in flood_cache:
                flood_cache[path] = _exact_flood_volume(path, graph.node_ids)
            if path not in depth_cache:
                depth_cache[path] = _post_action_max_depth(path, graph.node_ids)
        pred_c_vol, pred_c_depth = pred_cache[candidate_path]
        pred_b_vol, pred_b_depth = pred_cache[base_path]
        true_c_vol, true_b_vol = flood_cache[candidate_path], flood_cache[base_path]
        true_c_depth, true_b_depth = depth_cache[candidate_path], depth_cache[base_path]
        out: dict[str, object] = {
            "event_id": str(row.get("event_id", "")),
            "checkpoint_id": str(row["checkpoint_id"]),
            "rainfall_group": str(row["rainfall_group"]),
            "scientific_split": "calibration",
            "candidate_action_sha256": candidate_sha,
            "base_action_sha256": base_sha,
            "candidate_metadata_path": candidate_path,
            "base_metadata_path": base_path,
        }
        for node, idx in zip(priority, pidx, strict=True):
            out[f"pred_flood_delta_m3:{node}"] = float(pred_c_vol[idx] - pred_b_vol[idx])
            out[f"true_flood_delta_m3:{node}"] = float(true_c_vol[idx] - true_b_vol[idx])
            out[f"pred_depth_delta_m:{node}"] = float(pred_c_depth[idx] - pred_b_depth[idx])
            out[f"true_depth_delta_m:{node}"] = float(true_c_depth[idx] - true_b_depth[idx])
        rows.append(out)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no non-center calibration candidate/base pairs were produced")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step2-vs-SWMM calibration residual cases from D2 branches")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    frame = build_calibration_cases(
        manifest_path=args.manifest,
        run_summary_path=args.run_summary,
        graph_path=args.graph,
        step2_path=args.step2,
        priority_path=args.priority,
        device=args.device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(json.dumps({
        "rows": len(frame),
        "rainfall_groups": int(frame["rainfall_group"].nunique()),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
