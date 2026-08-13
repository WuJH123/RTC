"""Post-training V10 temporal consistency and deterministic examples audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v80 import prepare_static_v80
from rtc.step2_control_response_v100 import DirectHydraulicEffectSurrogateV100, build_actuator_node_influence_assets_v100
from rtc.step2_hydraulic_objective_v90 import derive_onset_sqrt_positive_weight_v80
from rtc.step2_optimization_v90 import candidate_batch_chunks_v90
from rtc.step2_train_response_v60 import V60TrainCache, derive_input_normalization_v60, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v100_contract import NonlocalHydraulicEffectLossContractV100
from rtc.step2_v100_temporal_consistency import horizon_bucket_indices_v100, slope_pair_v100

from run_step2_v100_nonlocal_d2 import _build_reference, _load_checkpoint, _load_graph, _sha256


CHANNELS = {
    "depth_m": 0,
    "flood_m3s": 2,
    "volume_m3": 3,
    "inflow_m3s": 4,
    "outflow_m3s": 5,
}
TYPE_NAMES = ("pump", "orifice", "weir")


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True).stdout.strip()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _aggregate(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(finite.mean()) if finite.size else float("nan")


def _event_balanced(records: list[dict[str, Any]], getter) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(getter(record))
        if np.isfinite(value):
            by_event[str(record["event_key"])].append(value)
    return _aggregate([float(np.mean(values)) for values in by_event.values() if values])


def _candidate_temporal_metrics(truth: np.ndarray, pred: np.ndarray, times: np.ndarray, buckets: dict[str, np.ndarray]) -> dict[str, Any]:
    truth_t = np.moveaxis(np.asarray(truth, dtype=np.float64), 1, -1)
    pred_t = np.moveaxis(np.asarray(pred, dtype=np.float64), 1, -1)
    true_slope = slope_pair_v100(truth_t, times)
    pred_slope = slope_pair_v100(pred_t, times)
    result: dict[str, Any] = {}
    for bucket, indices in buckets.items():
        if len(indices) == 0:
            continue
        t = true_slope[..., indices]
        p = pred_slope[..., indices]
        flat_t, flat_p = t.reshape(-1), p.reshape(-1)
        sign_mask = np.abs(flat_t) > 1e-8
        result[bucket] = {
            "slope_rmse": float(np.sqrt(np.mean(np.square(flat_p - flat_t)))),
            "slope_sign_agreement": float(np.mean(np.sign(flat_p[sign_mask]) == np.sign(flat_t[sign_mask]))) if np.any(sign_mask) else float("nan"),
            "abrupt_jump_rate": float(np.mean(np.abs(flat_p) > 3.0 * np.maximum(np.abs(flat_t), 1e-8))),
        }
    magnitude_true = np.max(np.abs(truth_t), axis=1)
    magnitude_pred = np.max(np.abs(pred_t), axis=1)
    peak_true = np.argmax(magnitude_true, axis=-1)
    peak_pred = np.argmax(magnitude_pred, axis=-1)
    result["peak_effect_timing_error_min"] = float(np.mean(np.abs(peak_true - peak_pred) * 5.0))
    onset_true = np.argmax(magnitude_true > 1e-8, axis=-1)
    onset_pred = np.argmax(magnitude_pred > 1e-8, axis=-1)
    result["response_onset_timing_error_min"] = float(np.mean(np.abs(onset_true - onset_pred) * 5.0))
    result["true_peak_time_min"] = float(np.mean(peak_true * 5.0))
    result["pred_peak_time_min"] = float(np.mean(peak_pred * 5.0))
    return result


def _load_v100_model(args: argparse.Namespace, cache: V60TrainCache, fit_names: list[str], device: torch.device):
    graph = _load_graph(args.graph)
    cache_lineage = __import__("rtc.step2_shards_v60", fromlist=["validate_v60_cache_lineage"]).validate_v60_cache_lineage(args.cache_manifest)
    normalization = derive_input_normalization_v60(cache, fit_names)
    scales = derive_target_scales_v70(cache, fit_names)
    basis = build_control_basis_v60(graph)
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    reference = _build_reference(graph, cache, fit_names, scales)
    reference.load_state_dict(hydraulic_checkpoint["state_dict"], strict=True)
    influence = build_actuator_node_influence_assets_v100(
        inp_path=args.frozen_inp, expected_inp_sha256=args.expected_inp_sha256,
        node_ids=graph.node_ids, actuator_ids=graph.actuator_ids,
        actuator_upstream=graph.actuator_upstream, actuator_downstream=graph.actuator_downstream,
    )
    prepared_cpu = prepare_static_v60(graph, "cpu")
    model = DirectHydraulicEffectSurrogateV100(
        reference_model=reference, temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=scales.state_delta_scale, flow_delta_scale=scales.flow_delta_scale,
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]), node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_count=len(graph.actuator_ids), influence_assets=influence,
        contract=NonlocalHydraulicEffectLossContractV100(),
    )
    payload = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    prepared = prepare_static_v80(graph, device)
    return graph, cache_lineage, normalization, scales, model.to(device).eval(), prepared


def _node_distances(graph, source_nodes: list[int]) -> np.ndarray:
    adjacency = [[] for _ in graph.node_ids]
    for src, dst in np.asarray(graph.edge_index, dtype=np.int64).T.tolist():
        adjacency[int(src)].append(int(dst)); adjacency[int(dst)].append(int(src))
    distance = np.full(len(graph.node_ids), np.inf, dtype=np.float64)
    queue = list(map(int, source_nodes)); distance[queue] = 0.0
    head = 0
    while head < len(queue):
        node = queue[head]; head += 1
        for nxt in adjacency[node]:
            if distance[nxt] == np.inf:
                distance[nxt] = distance[node] + 1.0; queue.append(nxt)
    return distance


def _run_subset(model, cache, names, normalization, prepared, scales, device, graph, representatives, all_group_records, all_candidate_records, subset_name):
    times_full = None
    with torch.no_grad():
        for name in names:
            entry = cache.entry(name)
            batch = cache.batch(name, normalization, device)
            outputs = []
            for chunk in candidate_batch_chunks_v90(batch, candidate_chunk_size=4):
                outputs.append(model(chunk.initial_state, chunk.rainfall, chunk.reference_settings, chunk.candidate_settings, chunk.previous_actuator_flow, prepared))
            output_state = torch.cat([o.raw_delta_states_physical for o in outputs], dim=1).cpu().numpy()[0]
            output_flow = torch.cat([o.raw_delta_flows_physical for o in outputs], dim=1).cpu().numpy()[0]
            retained = outputs[0].horizon_indices.cpu().numpy().astype(np.int64)
            times = (retained + 1) * 300.0
            if times_full is None:
                times_full = times
            truth_state = batch.true_candidate_states.cpu().numpy()[0][:, retained]
            ref_state = batch.true_reference_states.cpu().numpy()[0][retained]
            truth_flow = batch.true_candidate_flows.cpu().numpy()[0][:, retained]
            ref_flow = batch.true_reference_flows.cpu().numpy()[0][retained]
            true_state_delta = truth_state - ref_state[None]
            true_flow_delta = truth_flow - ref_flow[None]
            buckets = horizon_bucket_indices_v100(retained, times)
            event_key = f"{entry.rainfall_group}::{entry.event_id}"
            group_metrics: dict[str, Any] = {"event_key": event_key, "subset": subset_name, "group": name}
            for channel_name, channel_index in CHANNELS.items():
                truth = true_state_delta[..., channel_index]
                pred = output_state[..., list(CHANNELS).index(channel_name) + (1 if list(CHANNELS).index(channel_name) >= 1 else 0)] if False else output_state[..., channel_index]
                # V100 raw output channels retain the six-channel schema; use exact channel index.
                pred = output_state[..., channel_index]
                group_metrics[channel_name] = _candidate_temporal_metrics(truth, pred, times, buckets)
            group_metrics["managed_flow_m3s"] = _candidate_temporal_metrics(true_flow_delta, output_flow, times, buckets)
            all_group_records.append(group_metrics)
            for candidate_index in range(output_state.shape[0]):
                actuator_delta = np.asarray(entry.arrays["settings"][entry.indices[candidate_index if candidate_index < len(entry.indices) else 0]], dtype=np.float64) if False else None
                candidate_row = {"event_key": event_key, "group": name, "subset": subset_name, "candidate_index": candidate_index}
                for channel_name, channel_index in CHANNELS.items():
                    candidate_row[channel_name] = _candidate_temporal_metrics(true_state_delta[candidate_index:candidate_index + 1, ..., channel_index], output_state[candidate_index:candidate_index + 1, ..., channel_index], times, buckets)
                candidate_row["managed_flow_m3s"] = _candidate_temporal_metrics(true_flow_delta[candidate_index:candidate_index + 1], output_flow[candidate_index:candidate_index + 1], times, buckets)
                all_candidate_records.append(candidate_row)
            for candidate_index in range(output_state.shape[0]):
                candidate_row_index = [i for i in entry.indices if i != entry.reference_index][candidate_index]
                changed_idx = int(np.flatnonzero(np.any(np.abs(np.asarray(entry.arrays["settings"][candidate_row_index], dtype=np.float64) - np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float64)) > 1e-8, axis=0))[0])
                flags = np.asarray(graph.actuator_physics[changed_idx, :3])
                actuator_type = TYPE_NAMES[int(np.argmax(flags))]
                if actuator_type in representatives:
                    continue
                true_candidate = true_state_delta[candidate_index]
                pred_candidate = output_state[candidate_index]
                remote_mask = _node_distances(graph, [int(graph.actuator_upstream[changed_idx]), int(graph.actuator_downstream[changed_idx])]) > 4.0
                remote_score = np.max(np.abs(true_candidate[..., 0]), axis=0)
                remote_score[~remote_mask] = -1.0
                remote_node = int(np.argmax(remote_score))
                selected_positions = [0, min(5, len(times)-1), min(14, len(times)-1), len(times)-1]
                rows = []
                for pos in selected_positions:
                    rows.append({
                        "time_min": float(times[pos] / 60.0),
                        "upstream": {"node_id": graph.node_ids[int(graph.actuator_upstream[changed_idx])], "true_delta_depth_m": float(true_candidate[pos, graph.actuator_upstream[changed_idx], 0]), "pred_delta_depth_m": float(pred_candidate[pos, graph.actuator_upstream[changed_idx], 0]), "true_delta_head_m": float(true_candidate[pos, graph.actuator_upstream[changed_idx], 1]), "pred_delta_head_m": float(pred_candidate[pos, graph.actuator_upstream[changed_idx], 1]), "true_delta_flood_m3s": float(true_candidate[pos, graph.actuator_upstream[changed_idx], 2]), "pred_delta_flood_m3s": float(pred_candidate[pos, graph.actuator_upstream[changed_idx], 2])},
                        "downstream": {"node_id": graph.node_ids[int(graph.actuator_downstream[changed_idx])], "true_delta_depth_m": float(true_candidate[pos, graph.actuator_downstream[changed_idx], 0]), "pred_delta_depth_m": float(pred_candidate[pos, graph.actuator_downstream[changed_idx], 0]), "true_delta_head_m": float(true_candidate[pos, graph.actuator_downstream[changed_idx], 1]), "pred_delta_head_m": float(pred_candidate[pos, graph.actuator_downstream[changed_idx], 1]), "true_delta_flood_m3s": float(true_candidate[pos, graph.actuator_downstream[changed_idx], 2]), "pred_delta_flood_m3s": float(pred_candidate[pos, graph.actuator_downstream[changed_idx], 2])},
                        "remote_strongest": {"node_id": graph.node_ids[remote_node], "true_delta_depth_m": float(true_candidate[pos, remote_node, 0]), "pred_delta_depth_m": float(pred_candidate[pos, remote_node, 0]), "true_delta_head_m": float(true_candidate[pos, remote_node, 1]), "pred_delta_head_m": float(pred_candidate[pos, remote_node, 1]), "true_delta_flood_m3s": float(true_candidate[pos, remote_node, 2]), "pred_delta_flood_m3s": float(pred_candidate[pos, remote_node, 2])},
                        "actuator": {"id": graph.actuator_ids[changed_idx], "type": actuator_type, "true_delta_flow_m3s": float(true_flow_delta[candidate_index, pos, changed_idx]), "pred_delta_flow_m3s": float(output_flow[candidate_index, pos, changed_idx])},
                    })
                representatives[actuator_type] = {"group": name, "subset": subset_name, "actuator": graph.actuator_ids[changed_idx], "rows": rows}
    return times_full


def _aggregate_records(records: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    selected = [r for r in records if r["subset"] == subset]
    event_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        event_groups[str(record["event_key"])].append(record)
    result: dict[str, Any] = {"records": len(selected), "events": len(event_groups)}
    for channel in (*CHANNELS, "managed_flow_m3s"):
        result[channel] = {}
        for metric in ("peak_effect_timing_error_min", "response_onset_timing_error_min", "true_peak_time_min", "pred_peak_time_min"):
            result[channel][metric] = _event_balanced(selected, lambda r, c=channel, m=metric: r[c][m])
        for bucket in ("0_30_min", "30_120_min", "120_360_min"):
            result[channel][bucket] = {
                metric: _event_balanced([r for r in selected if bucket in r[channel]], lambda r, c=channel, b=bucket, m=metric: r[c][b][m])
                for metric in ("slope_rmse", "slope_sign_agreement", "abrupt_jump_rate")
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True); parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True); parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--frozen-inp", required=True); parser.add_argument("--expected-inp-sha256", required=True)
    parser.add_argument("--model-checkpoint", required=True); parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [n for n in fit if n.startswith("D2::")]; holdout_d2 = [n for n in holdout if n.startswith("D2::")]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph, lineage, normalization, scales, model, prepared = _load_v100_model(args, cache, fit, device)
    records: list[dict[str, Any]] = []; candidates: list[dict[str, Any]] = []; reps: dict[str, Any] = {}
    _run_subset(model, cache, fit_d2, normalization, prepared, scales, device, graph, reps, records, candidates, "TrainFit_D2")
    _run_subset(model, cache, holdout_d2, normalization, prepared, scales, device, graph, reps, records, candidates, "TrainInternalHoldout_D2")
    report = {
        "contract": "PROJECT7_STEP2_V100_TEMPORAL_HYDRAULIC_CONSISTENCY_V1",
        "git_head": _git_head(), "model_checkpoint_sha256": _sha256(args.model_checkpoint), "graph_sha256": _sha256(args.graph), "cache_manifest_sha256": _sha256(args.cache_manifest),
        "lineage": lineage, "boundary": {"diagnostic_only": True, "swmm_run": False, "d3_run": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False},
        "schedule": {"seed": 42, "epochs": 4, "retrained": False, "retained_horizon": 72, "control_interval_seconds": 600, "model_step_seconds": 300},
        "metrics": {"TrainFit_D2": _aggregate_records(records, "TrainFit_D2"), "TrainInternalHoldout_D2": _aggregate_records(records, "TrainInternalHoldout_D2")},
        "representative_examples": list(reps.values())[:3],
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "fit_records": len([r for r in records if r["subset"] == "TrainFit_D2"]), "holdout_records": len([r for r in records if r["subset"] == "TrainInternalHoldout_D2"])}, indent=2))


if __name__ == "__main__":
    main()
