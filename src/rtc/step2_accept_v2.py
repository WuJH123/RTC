from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .acceptance import rank_correlation
from .code_contract import rtc_source_tree_sha256
from .contracts import load_priority_nodes
from .flood_volume import trapezoid_node_flood_volume
from .large_model_cli import _device
from .production_cli import _load_graph, _load_step2
from .step2_shards import load_shard_manifest, sha256_file


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _group() -> dict[str, object]:
    return {
        "depth_sse": 0.0,
        "depth_n": 0,
        "flow_sse": 0.0,
        "flow_n": 0,
        "pred_tfv": [],
        "true_tfv": [],
        "pred_pfv": [],
        "true_pfv": [],
        "negative_depth_count": 0,
        "depth_value_count": 0,
        "negative_flood_rate_count": 0,
        "flood_rate_value_count": 0,
        "negative_node_volume_count": 0,
        "node_volume_value_count": 0,
        "nonfinite_state_count": 0,
        "state_value_count": 0,
        "nonfinite_flow_count": 0,
        "flow_value_count": 0,
    }


def _fraction(count: object, total: object) -> float:
    return float(count) / max(float(total), 1.0)


def accept_step2_large_v2_main() -> None:
    parser = argparse.ArgumentParser(
        description="Accept Step2 on exact SWMM truth with rainfall-group-balanced metrics"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = _device(args.device)
    graph = _load_graph(args.graph)
    model = _load_step2(args.model, device)
    manifest = load_shard_manifest(args.manifest)
    model_meta = dict(getattr(model, "runtime_metadata", {}))
    if int(model_meta.get("model_step_seconds", -1)) != int(manifest["model_step_seconds"]):
        raise ValueError("Step2 validation shard step differs from model checkpoint")
    if int(model_meta.get("horizon_steps", -1)) != int(manifest["horizon_steps"]):
        raise ValueError("Step2 validation shard horizon differs from model checkpoint")

    up = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    down = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    static = torch.as_tensor(
        graph.static_node_features, dtype=torch.float32, device=device
    )
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    physics = torch.as_tensor(
        graph.actuator_physics, dtype=torch.float32, device=device
    )
    pidx = None
    if args.priority:
        priority = load_priority_nodes(args.priority)
        missing = sorted(set(priority) - set(graph.node_ids))
        if missing:
            raise ValueError(f"priority mapping incompatible with graph: {missing}")
        pidx = np.asarray([graph.node_ids.index(n) for n in priority], dtype=int)

    groups: dict[str, dict[str, object]] = {}
    negative_tolerance = -1e-6
    with torch.no_grad():
        for item in manifest["shards"]:
            with np.load(str(item["path"]), allow_pickle=False) as ds:
                if "exact_node_flood_volume_m3" not in ds.files:
                    raise ValueError(
                        "Step2 Formal acceptance requires exact SWMM node flooding-volume truth"
                    )
                if "rainfall_group" not in ds.files:
                    raise ValueError("Step2 Formal acceptance requires rainfall_group provenance")
                if int(ds["model_step_seconds"].item()) != int(manifest["model_step_seconds"]):
                    raise ValueError("Step2 validation shard embedded step mismatch")
                count = ds["initial_state"].shape[0]
                rainfall_groups = ds["rainfall_group"].astype(str)
                for start in range(0, count, args.batch_size):
                    end = min(count, start + args.batch_size)
                    b = end - start
                    initial = torch.as_tensor(
                        ds["initial_state"][start:end], dtype=torch.float32, device=device
                    )
                    rain = torch.as_tensor(
                        ds["rainfall"][start:end], dtype=torch.float32, device=device
                    )
                    settings = torch.as_tensor(
                        ds["settings"][start:end], dtype=torch.float32, device=device
                    )
                    prev = torch.as_tensor(
                        ds["previous_actuator_flow"][start:end],
                        dtype=torch.float32,
                        device=device,
                    )
                    rollout = model.rollout(
                        initial,
                        rain,
                        settings,
                        prev,
                        up,
                        down,
                        physics.unsqueeze(0).expand(b, -1, -1),
                        static,
                        edges,
                    )
                    elapsed = ds["elapsed_seconds"][start:end].astype(np.float32)
                    dt_np = np.diff(elapsed, axis=1)
                    expected_step = float(manifest["model_step_seconds"])
                    if np.any(np.abs(dt_np - expected_step) > 1e-6):
                        raise ValueError("Step2 validation branch violates frozen model step")
                    pred_node = trapezoid_node_flood_volume(
                        initial,
                        rollout.states,
                        flood_rate_index=2,
                        dt_seconds=torch.as_tensor(dt_np, dtype=torch.float32, device=device),
                    ).cpu().numpy()
                    exact = ds["exact_node_flood_volume_m3"][start:end].astype(float)
                    ps = rollout.states.cpu().numpy()
                    pf = rollout.actuator_flows.cpu().numpy()
                    ts = ds["target_states"][start:end]
                    tf = ds["target_actuator_flows"][start:end]
                    for local, group_id in enumerate(rainfall_groups[start:end]):
                        agg = groups.setdefault(str(group_id), _group())
                        depth_err = ps[local, ..., 0] - ts[local, ..., 0]
                        flow_err = pf[local] - tf[local]
                        agg["depth_sse"] = float(agg["depth_sse"]) + float(
                            np.square(depth_err).sum()
                        )
                        agg["depth_n"] = int(agg["depth_n"]) + int(depth_err.size)
                        agg["flow_sse"] = float(agg["flow_sse"]) + float(
                            np.square(flow_err).sum()
                        )
                        agg["flow_n"] = int(agg["flow_n"]) + int(flow_err.size)
                        agg["pred_tfv"].append(float(pred_node[local].sum()))  # type: ignore[union-attr]
                        agg["true_tfv"].append(float(exact[local].sum()))  # type: ignore[union-attr]

                        state_values = ps[local]
                        flow_values = pf[local]
                        depth = state_values[..., 0]
                        flood_rate = state_values[..., 2]
                        node_volume = state_values[..., 3]
                        agg["negative_depth_count"] = int(agg["negative_depth_count"]) + int(
                            np.count_nonzero(np.isfinite(depth) & (depth < negative_tolerance))
                        )
                        agg["depth_value_count"] = int(agg["depth_value_count"]) + int(depth.size)
                        agg["negative_flood_rate_count"] = int(agg["negative_flood_rate_count"]) + int(
                            np.count_nonzero(
                                np.isfinite(flood_rate) & (flood_rate < negative_tolerance)
                            )
                        )
                        agg["flood_rate_value_count"] = int(agg["flood_rate_value_count"]) + int(
                            flood_rate.size
                        )
                        agg["negative_node_volume_count"] = int(agg["negative_node_volume_count"]) + int(
                            np.count_nonzero(
                                np.isfinite(node_volume) & (node_volume < negative_tolerance)
                            )
                        )
                        agg["node_volume_value_count"] = int(agg["node_volume_value_count"]) + int(
                            node_volume.size
                        )
                        agg["nonfinite_state_count"] = int(agg["nonfinite_state_count"]) + int(
                            state_values.size - np.isfinite(state_values).sum()
                        )
                        agg["state_value_count"] = int(agg["state_value_count"]) + int(
                            state_values.size
                        )
                        agg["nonfinite_flow_count"] = int(agg["nonfinite_flow_count"]) + int(
                            flow_values.size - np.isfinite(flow_values).sum()
                        )
                        agg["flow_value_count"] = int(agg["flow_value_count"]) + int(
                            flow_values.size
                        )
                        if pidx is not None:
                            agg["pred_pfv"].append(float(pred_node[local, pidx].sum()))  # type: ignore[union-attr]
                            agg["true_pfv"].append(float(exact[local, pidx].sum()))  # type: ignore[union-attr]
    if not groups:
        raise ValueError("Step2 validation contains no rainfall groups")

    group_metrics: list[dict[str, float | str]] = []
    for group_id, agg in sorted(groups.items()):
        pred_tfv = np.asarray(agg["pred_tfv"], dtype=float)
        true_tfv = np.asarray(agg["true_tfv"], dtype=float)
        row: dict[str, float | str] = {
            "rainfall_group": group_id,
            "depth_rmse_m": float(
                np.sqrt(float(agg["depth_sse"]) / max(int(agg["depth_n"]), 1))
            ),
            "managed_flow_rmse_m3s": float(
                np.sqrt(float(agg["flow_sse"]) / max(int(agg["flow_n"]), 1))
            ),
            "tfv_exact_truth_mae_m3": float(np.mean(np.abs(pred_tfv - true_tfv))),
            "tfv_exact_truth_rank_correlation": float(
                rank_correlation(pred_tfv, true_tfv)
            ),
            "negative_depth_fraction": _fraction(
                agg["negative_depth_count"], agg["depth_value_count"]
            ),
            "negative_flooding_rate_fraction": _fraction(
                agg["negative_flood_rate_count"], agg["flood_rate_value_count"]
            ),
            "negative_node_volume_fraction": _fraction(
                agg["negative_node_volume_count"], agg["node_volume_value_count"]
            ),
            "nonfinite_state_fraction": _fraction(
                agg["nonfinite_state_count"], agg["state_value_count"]
            ),
            "nonfinite_actuator_flow_fraction": _fraction(
                agg["nonfinite_flow_count"], agg["flow_value_count"]
            ),
        }
        if pidx is not None:
            pred_pfv = np.asarray(agg["pred_pfv"], dtype=float)
            true_pfv = np.asarray(agg["true_pfv"], dtype=float)
            row["priority_flood_exact_truth_mae_m3"] = float(
                np.mean(np.abs(pred_pfv - true_pfv))
            )
            row["priority_flood_exact_truth_rank_correlation"] = float(
                rank_correlation(pred_pfv, true_pfv)
            )
        group_metrics.append(row)

    metric_names = [
        "depth_rmse_m",
        "managed_flow_rmse_m3s",
        "tfv_exact_truth_mae_m3",
        "tfv_exact_truth_rank_correlation",
        "negative_depth_fraction",
        "negative_flooding_rate_fraction",
        "negative_node_volume_fraction",
        "nonfinite_state_fraction",
        "nonfinite_actuator_flow_fraction",
    ]
    if pidx is not None:
        metric_names += [
            "priority_flood_exact_truth_mae_m3",
            "priority_flood_exact_truth_rank_correlation",
        ]
    metrics = {
        name: float(np.mean([float(row[name]) for row in group_metrics]))
        for name in metric_names
    }
    payload = {
        "contract": "STEP2_EXACT_TRUTH_ACCEPTANCE_V4_GROUP_BALANCED_TIME_LOCKED",
        "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        "model_sha256": _sha(args.model),
        "manifest_sha256": sha256_file(args.manifest),
        "model_step_seconds": int(manifest["model_step_seconds"]),
        "horizon_steps": int(manifest["horizon_steps"]),
        "rainfall_groups": len(group_metrics),
        "aggregation": "equal_weight_per_rainfall_group",
        "metrics": metrics,
        "priority_diagnostic_only": True,
        "truth_source_tfv_pfv": "SWMM_NODE_STATISTICS_CUMULATIVE_EXACT_HORIZON",
        "prediction_volume_integration": "trapezoid_current_plus_future_flooding_rate",
        "physics_informed_architecture": True,
        "strict_mass_conservation_enforced": False,
        "physical_plausibility_diagnostics": {
            "negative_value_tolerance": negative_tolerance,
            "negative_depth_fraction": "fraction of predicted depth values below tolerance",
            "negative_flooding_rate_fraction": "fraction of predicted flooding-rate values below tolerance",
            "negative_node_volume_fraction": "fraction of predicted node-volume values below tolerance",
            "nonfinite_state_fraction": "fraction of predicted hydraulic-state values that are NaN/Inf",
            "nonfinite_actuator_flow_fraction": "fraction of predicted managed-flow values that are NaN/Inf",
            "interpretation": (
                "The Step2 architecture is physics-informed by actuator setting-to-flow and "
                "directed upstream/downstream flow injection, but it is not a strict hydraulic "
                "continuity solver. These diagnostics quantify basic physical plausibility; "
                "authoritative hydraulic/flood truth remains SWMM."
            ),
        },
        "group_metrics": group_metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    accept_step2_large_v2_main()
