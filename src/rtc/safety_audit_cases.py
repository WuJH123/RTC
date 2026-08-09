from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calibration import SafetyCalibration
from .calibration_cases import _exact_flood_volume, _join, _post_action_max_depth, _predict
from .contracts import load_priority_nodes
from .production_cli import _load_graph, _load_step2


def _site_budget(config: dict[str, object], key: str, nodes: tuple[str, ...]) -> np.ndarray:
    raw = config.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"budget config requires object {key}")
    missing = [node for node in nodes if node not in raw]
    if missing:
        raise ValueError(f"{key} missing priority nodes: {missing}")
    return np.asarray([float(raw[node]) for node in nodes], dtype=float)


def build_selected_action_audit_cases(
    *,
    manifest_path: str | Path,
    run_summary_path: str | Path,
    graph_path: str | Path,
    step2_path: str | Path,
    priority_path: str | Path,
    calibration_path: str | Path,
    budget_config_path: str | Path,
    device: str | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    runs = pd.read_csv(run_summary_path)
    merged = _join(manifest, runs)
    if "scientific_split" not in merged.columns:
        raise ValueError("D2 audit manifest must carry scientific_split")
    merged = merged[merged["scientific_split"].astype(str) == "safety_audit"].copy()
    if merged.empty:
        raise ValueError("no independent safety_audit D2 branches")
    if "rainfall_group" not in merged.columns:
        raise ValueError("safety audit manifest requires rainfall_group")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(graph_path)
    model = _load_step2(step2_path, dev)
    priority = load_priority_nodes(priority_path)
    calibration = SafetyCalibration.from_json(calibration_path)
    if calibration.priority_nodes != priority:
        raise ValueError("calibration priority ordering differs from audit priority file")
    config = json.loads(Path(budget_config_path).read_text(encoding="utf-8"))
    flood_budget = _site_budget(config, "flood_budget_m3", priority)
    depth_budget = _site_budget(config, "depth_budget_m", priority)
    pidx = np.asarray([graph.node_ids.index(node) for node in priority], dtype=int)

    action_keys = ["checkpoint_id", "candidate_action_sha256"]
    group_keys = ["checkpoint_id"]
    if "event_id" in merged.columns:
        action_keys.insert(0, "event_id")
        group_keys.insert(0, "event_id")
    unique = merged.drop_duplicates(action_keys).copy()

    pred_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    true_vol_cache: dict[str, np.ndarray] = {}
    true_depth_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for keys, group in unique.groupby(group_keys, sort=False):
        base_rows = group[
            group["candidate_action_sha256"].astype(str)
            == group["base_action_sha256"].astype(str)
        ]
        if base_rows.empty:
            raise ValueError(f"audit checkpoint lacks base action branch: {keys}")
        base = base_rows.iloc[0]
        base_path = str(base["metadata_path"])
        if base_path not in pred_cache:
            pred_cache[base_path] = _predict(model, graph, base_path, dev)
            true_vol_cache[base_path] = _exact_flood_volume(base_path, graph.node_ids)
            true_depth_cache[base_path] = _post_action_max_depth(base_path, graph.node_ids)
        pred_base_vol, pred_base_depth = pred_cache[base_path]
        true_base_vol, true_base_depth = true_vol_cache[base_path], true_depth_cache[base_path]

        candidates: list[dict[str, object]] = []
        for _, candidate in group.iterrows():
            path = str(candidate["metadata_path"])
            if path not in pred_cache:
                pred_cache[path] = _predict(model, graph, path, dev)
                true_vol_cache[path] = _exact_flood_volume(path, graph.node_ids)
                true_depth_cache[path] = _post_action_max_depth(path, graph.node_ids)
            pred_vol, pred_depth = pred_cache[path]
            true_vol, true_depth = true_vol_cache[path], true_depth_cache[path]
            pred_flood_delta = pred_vol[pidx] - pred_base_vol[pidx]
            pred_depth_delta = pred_depth[pidx] - pred_base_depth[pidx]
            pred_flood_ucb = pred_flood_delta + np.asarray(calibration.flood_error_ucb_m3)
            pred_depth_ucb = pred_depth_delta + np.asarray(calibration.depth_error_ucb_m)
            admitted = bool(
                np.all(pred_flood_ucb <= flood_budget)
                and np.all(pred_depth_ucb <= depth_budget)
            )
            candidates.append(
                {
                    "row": candidate,
                    "path": path,
                    "pred_tfv_m3": float(pred_vol.sum()),
                    "admitted": admitted,
                    "pred_flood_ucb": pred_flood_ucb,
                    "pred_depth_ucb": pred_depth_ucb,
                    "true_flood_delta": true_vol[pidx] - true_base_vol[pidx],
                    "true_depth_delta": true_depth[pidx] - true_base_depth[pidx],
                }
            )
        safe = [item for item in candidates if bool(item["admitted"])]
        fallback_used = not safe
        if safe:
            selected = min(safe, key=lambda item: float(item["pred_tfv_m3"]))
        else:
            selected = next(
                item
                for item in candidates
                if str(item["row"]["candidate_action_sha256"])
                == str(item["row"]["base_action_sha256"])
            )
        selected_row = selected["row"]
        out: dict[str, object] = {
            "event_id": str(selected_row.get("event_id", "")),
            "checkpoint_id": str(selected_row["checkpoint_id"]),
            "rainfall_group": str(selected_row["rainfall_group"]),
            "scientific_split": "safety_audit",
            "selected_action_sha256": str(selected_row["candidate_action_sha256"]),
            "base_action_sha256": str(selected_row["base_action_sha256"]),
            "admitted": bool(selected["admitted"]) if not fallback_used else False,
            "fallback_used": fallback_used,
            "predicted_selected_tfv_m3": float(selected["pred_tfv_m3"]),
            "selection_contract": "D2_HELDOUT_SITEWISE_UCB_TFV_MIN_V1",
        }
        for i, node in enumerate(priority):
            out[f"pred_flood_ucb_m3:{node}"] = float(selected["pred_flood_ucb"][i])
            out[f"true_flood_delta_m3:{node}"] = float(selected["true_flood_delta"][i])
            out[f"pred_depth_ucb_m:{node}"] = float(selected["pred_depth_ucb"][i])
            out[f"true_depth_delta_m:{node}"] = float(selected["true_depth_delta"][i])
        rows.append(out)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no independent selected-action safety audit decisions were produced")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent site-wise UCB selected-action audit cases")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--budget-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    frame = build_selected_action_audit_cases(
        manifest_path=args.manifest,
        run_summary_path=args.run_summary,
        graph_path=args.graph,
        step2_path=args.step2,
        priority_path=args.priority,
        calibration_path=args.calibration,
        budget_config_path=args.budget_config,
        device=args.device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(json.dumps({
        "decisions": len(frame),
        "events": int(frame["event_id"].nunique()),
        "rainfall_groups": int(frame["rainfall_group"].nunique()),
        "fallbacks": int(frame["fallback_used"].astype(bool).sum()),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
