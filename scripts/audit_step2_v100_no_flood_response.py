"""Read-only Train-D2 audit of hydraulic effects without flooding changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from rtc.graph import GraphSchema
from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step2_v100_no_flood_audit import (
    HORIZON_BUCKETS_V100,
    NO_FLOOD_EPSILON_V100,
    bucket_slices_v100,
    joint_no_flood_fractions_v100,
    no_flood_mask_v100,
)


CHANNELS = {
    "depth": 0,
    "head": 1,
    "flood": 2,
    "volume": 3,
    "inflow": 4,
    "outflow": 5,
}
_CATEGORY_NAMES = ("upstream_endpoint", "downstream_endpoint", "local_nodes", "remote_nodes")
_TYPE_NAMES = ("pump", "orifice", "weir")
_RESERVOIR_SIZE = 20_000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


class _Reservoir:
    """Bounded deterministic summary for a potentially very large cell stream."""
    def __init__(self, *, capacity: int = _RESERVOIR_SIZE, seed: int = 42) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.active = 0
        self.maximum = 0.0
        self.values: list[float] = []

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        flat = np.abs(np.asarray(values, dtype=np.float64))[np.asarray(mask, dtype=bool)]
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        self.count += int(flat.size)
        self.sum += float(flat.sum())
        self.sum_sq += float(np.square(flat).sum())
        self.active += int(np.count_nonzero(flat > NO_FLOOD_EPSILON_V100))
        self.maximum = max(self.maximum, float(flat.max()))
        if flat.size > self.capacity:
            flat = flat[np.linspace(0, flat.size - 1, self.capacity, dtype=np.int64)]
        combined = np.concatenate((np.asarray(self.values, dtype=np.float64), flat))
        if combined.size > self.capacity:
            combined = combined[np.linspace(0, combined.size - 1, self.capacity, dtype=np.int64)]
        self.values = combined.tolist()

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {
                "count": 0, "mean_abs": None, "median_abs": None, "p90_abs": None,
                "p95_abs": None, "p99_abs": None, "max_abs": None,
                "active_fraction": None, "quantile_method": "empty",
            }
        sample = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(self.count),
            "mean_abs": float(self.sum / self.count),
            "median_abs": float(np.quantile(sample, 0.50)),
            "p90_abs": float(np.quantile(sample, 0.90)),
            "p95_abs": float(np.quantile(sample, 0.95)),
            "p99_abs": float(np.quantile(sample, 0.99)),
            "max_abs": float(self.maximum),
            "active_fraction": float(self.active / self.count),
            "quantile_method": "deterministic_reservoir",
            "reservoir_count": int(sample.size),
        }


def _distance_from_endpoints(graph: GraphSchema) -> np.ndarray:
    node_count = len(graph.node_ids)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in np.asarray(graph.edge_index, dtype=np.int64).T.tolist():
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))
    distances = np.full((len(graph.actuator_ids), node_count), np.inf, dtype=np.float64)
    for actuator in range(len(graph.actuator_ids)):
        queue: deque[int] = deque()
        for endpoint in (int(graph.actuator_upstream[actuator]), int(graph.actuator_downstream[actuator])):
            distances[actuator, endpoint] = 0.0
            queue.append(endpoint)
        while queue:
            node = queue.popleft()
            next_distance = distances[actuator, node] + 1.0
            for neighbour in adjacency[node]:
                if next_distance < distances[actuator, neighbour]:
                    distances[actuator, neighbour] = next_distance
                    queue.append(neighbour)
    return distances


def _type_for_actuator(graph: GraphSchema, actuator: int) -> str:
    flags = np.asarray(graph.actuator_physics[actuator, :3], dtype=np.float64)
    index = int(np.argmax(flags))
    if flags[index] <= 0.5:
        raise ValueError(f"actuator {actuator} has no pump/orifice/weir type flag")
    return _TYPE_NAMES[index]


def _changed_actuator(settings: np.ndarray, ref: int, candidate: int) -> np.ndarray:
    delta = np.asarray(settings[candidate], dtype=np.float64) - np.asarray(settings[ref], dtype=np.float64)
    changed = np.flatnonzero(np.any(np.abs(delta) > NO_FLOOD_EPSILON_V100, axis=0))
    if changed.size != 1:
        raise ValueError(f"canonical D2 requires one changed actuator, got {changed.tolist()}")
    return changed


def _new_stats() -> dict[str, _Reservoir]:
    return {name: _Reservoir(seed=42 + i) for i, name in enumerate((*CHANNELS, "actuator_flow"))}


def _summary_update(
    stats: dict[str, dict[str, dict[str, dict[str, _Reservoir]]]],
    scope: str,
    category: str,
    bucket: str,
    deltas: dict[str, np.ndarray],
    flood_mask: np.ndarray,
) -> None:
    target = stats.setdefault(scope, {}).setdefault(category, {}).setdefault(bucket, _new_stats())
    for name, values in deltas.items():
        mask = flood_mask
        if name == "actuator_flow" and mask.ndim > np.asarray(values).ndim:
            mask = mask.reshape(np.asarray(values).shape[0], np.asarray(values).shape[1], -1).all(axis=-1)
        target[name].update(values, mask)


def _cell_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=bool)[np.asarray(mask, dtype=bool)])) if np.any(mask) else float("nan")


def audit_no_flood_response_v100_fast(graph_path: str | Path, cache_path: str | Path) -> dict[str, Any]:
    """Vectorized group-level implementation used by the full audit."""
    graph = _load_graph(graph_path)
    cache = V60TrainCache(cache_path)
    names = cache.names("D2")
    if len(names) != 144:
        raise ValueError(f"canonical Train D2 group count changed: {len(names)}")
    distances = _distance_from_endpoints(graph)
    buckets = bucket_slices_v100(72, 300)
    stats: dict[str, dict[str, dict[str, dict[str, _Reservoir]]]] = {}
    joint_counts = defaultdict(lambda: {"depth_num": 0, "depth_den": 0, "volume_num": 0, "volume_den": 0, "flow_num": 0, "flow_den": 0})
    type_counts = defaultdict(int)
    temporal_samples: list[dict[str, Any]] = []
    candidate_count = 0
    no_flood_cells = 0
    total_flood_cells = 0
    for group_index, name in enumerate(names):
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        candidates = [i for i in entry.indices if i != ref]
        changed = np.asarray([int(_changed_actuator(arrays["settings"], ref, c)[0]) for c in candidates], dtype=np.int64)
        types = np.asarray([_type_for_actuator(graph, int(a)) for a in changed], dtype=object)
        for type_name in _TYPE_NAMES:
            type_counts[type_name] += int(np.count_nonzero(types == type_name))
        candidate_count += len(candidates)
        state_delta = np.asarray(arrays["target_states"][candidates], dtype=np.float64) - np.asarray(arrays["target_states"][ref], dtype=np.float64)[None, ...]
        flood_mask = no_flood_mask_v100(state_delta[..., CHANNELS["flood"]])
        total_flood_cells += int(flood_mask.size)
        no_flood_cells += int(flood_mask.sum())
        node_delta = {key: state_delta[..., index] for key, index in CHANNELS.items() if key != "flood"}
        flow_delta = np.stack([
            np.asarray(arrays["target_actuator_flows"][candidate, :, actuator], dtype=np.float64)
            - np.asarray(arrays["target_actuator_flows"][ref, :, actuator], dtype=np.float64)
            for candidate, actuator in zip(candidates, changed)
        ], axis=0)
        for bucket, indices in buckets.items():
            idx = np.asarray(indices, dtype=np.int64)
            bucket_flood = flood_mask[:, idx]
            bucket_deltas = {key: value[:, idx] for key, value in node_delta.items()}
            bucket_deltas["actuator_flow"] = flow_delta[:, idx]
            masks = {
                "upstream_endpoint": np.equal(np.arange(len(graph.node_ids))[None, :], graph.actuator_upstream[changed][:, None]),
                "downstream_endpoint": np.equal(np.arange(len(graph.node_ids))[None, :], graph.actuator_downstream[changed][:, None]),
                "local_nodes": distances[changed] <= 4.0,
                "remote_nodes": distances[changed] > 4.0,
            }
            for category, node_masks in masks.items():
                for scope in ("all", *_TYPE_NAMES):
                    scope_mask = np.ones(len(candidates), dtype=bool) if scope == "all" else (types == scope)
                    rows = np.flatnonzero(scope_mask)
                    if rows.size == 0:
                        continue
                    target = stats.setdefault(scope, {}).setdefault(category, {}).setdefault(bucket, _new_stats())
                    for key, values in bucket_deltas.items():
                        value_parts: list[np.ndarray] = []
                        mask_parts: list[np.ndarray] = []
                        for row in rows.tolist():
                            node_mask = node_masks[row]
                            flood_part = bucket_flood[row][..., node_mask]
                            if key == "actuator_flow":
                                value_parts.append(values[row].reshape(-1))
                                mask_parts.append(flood_part.reshape(values[row].shape[0], -1).all(axis=-1).reshape(-1))
                            else:
                                value_parts.append(values[row][..., node_mask].reshape(-1))
                                mask_parts.append(flood_part.reshape(-1))
                        target[key].update(np.concatenate(value_parts), np.concatenate(mask_parts))
                    joint = joint_no_flood_fractions_v100(
                        bucket_deltas["depth"][rows], bucket_deltas["volume"][rows], bucket_deltas["actuator_flow"][rows], bucket_flood[rows], epsilon=NO_FLOOD_EPSILON_V100
                    )
                    counts = joint_counts[(scope, bucket)]
                    counts["depth_num"] += int(joint["depth_numerator"])
                    counts["depth_den"] += int(joint["depth_denominator"])
                    counts["volume_num"] += int(joint["volume_numerator"])
                    counts["volume_den"] += int(joint["volume_denominator"])
                    counts["flow_num"] += int(joint["flow_numerator"])
                    counts["flow_den"] += int(joint["flow_denominator"])
        if group_index < 3:
            candidate, actuator = candidates[0], int(changed[0])
            elapsed = np.asarray(arrays["elapsed_seconds"][ref])
            temporal_samples.append({
                "group": name,
                "checkpoint_seconds": float(elapsed[0]),
                "control_action_interval": [float(elapsed[0]), float(elapsed[0] + 600)],
                "model_step_interval": [float(elapsed[0]), float(elapsed[0] + 300)],
                "target_timestamps_seconds": [float(elapsed[i + 1]) for i in range(2)],
                "changed_actuator": actuator,
                "changed_actuator_id": graph.actuator_ids[actuator],
                "actuator_type": str(types[0]),
                "node0_target_state_first_two": np.asarray(arrays["target_states"][candidate, :2, 0, :], dtype=np.float64).tolist(),
                "changed_flow_first_two": np.asarray(arrays["target_actuator_flows"][candidate, :2, actuator], dtype=np.float64).tolist(),
            })
    summary = {
        scope: {
            category: {
                bucket: {name: value.result() for name, value in channel_data.items()}
                for bucket, channel_data in bucket_data.items()
            }
            for category, bucket_data in categories.items()
        }
        for scope, categories in stats.items()
    }
    joint: dict[str, dict[str, Any]] = {}
    for (scope, bucket), counts in joint_counts.items():
        joint.setdefault(scope, {})[bucket] = {
            "depth_cell_count": int(counts["depth_den"]),
            "volume_cell_count": int(counts["volume_den"]),
            "flow_sample_count": int(counts["flow_den"]),
            "depth_active_and_flood_inactive": float(counts["depth_num"] / max(counts["depth_den"], 1)),
            "volume_active_and_flood_inactive": float(counts["volume_num"] / max(counts["volume_den"], 1)),
            "flow_active_and_flood_inactive": float(counts["flow_num"] / max(counts["flow_den"], 1)),
        }
    all_counts = list(joint.get("all", {}).values())
    total_depth = sum(int(v["depth_cell_count"]) for v in all_counts)
    total_volume = sum(int(v["volume_cell_count"]) for v in all_counts)
    total_flow = sum(int(v["flow_sample_count"]) for v in all_counts)
    headline = {
        "depth_active_and_flood_inactive": float(sum(float(v["depth_active_and_flood_inactive"]) * int(v["depth_cell_count"]) for v in all_counts) / max(total_depth, 1)),
        "volume_active_and_flood_inactive": float(sum(float(v["volume_active_and_flood_inactive"]) * int(v["volume_cell_count"]) for v in all_counts) / max(total_volume, 1)),
        "flow_active_and_flood_inactive": float(sum(float(v["flow_active_and_flood_inactive"]) * int(v["flow_sample_count"]) for v in all_counts) / max(total_flow, 1)),
    }
    return {
        "contract": "PROJECT7_STEP2_V100_NO_FLOOD_HYDRAULIC_RESPONSE_AUDIT_V1",
        "git_head": _git_head(),
        "graph_sha256": _sha256(graph_path),
        "cache_manifest_sha256": _sha256(cache_path),
        "graph": {"nodes": len(graph.node_ids), "edges": int(graph.edge_index.shape[1]), "actuators": len(graph.actuator_ids)},
        "data": {"source": "canonical Train-only D2", "groups": len(names), "candidates": candidate_count, "candidate_per_group": 24, "actuator_type_candidate_counts": dict(type_counts)},
        "units": {"depth": "m", "head": "m", "flood": "m3/s", "volume": "m3", "inflow": "m3/s", "outflow": "m3/s", "actuator_flow": "m3/s", "time": "s"},
        "state_channels": CHANNELS,
        "no_flood_contract": {"epsilon": NO_FLOOD_EPSILON_V100, "definition": "abs(delta_flood) <= epsilon per node/time cell", "total_cells": total_flood_cells, "no_flood_cells": no_flood_cells, "no_flood_fraction": no_flood_cells / max(total_flood_cells, 1)},
        "horizon_buckets": {name: [int(i) for i in indices] for name, indices in buckets.items()},
        "distributions": summary,
        "joint_fractions": {"headline_sample_level": headline, "by_scope_and_bucket": joint, "hydraulic_effect_is_not_flooding_only": bool(any(value > 0.0 for value in headline.values()))},
        "temporal_contract_samples": temporal_samples,
        "boundary": {"swmm_run": False, "validation_accessed": False, "final_accessed": False, "formal_run": False, "uses_future_truth": False, "diagnostic_only": True},
    }


def audit_no_flood_response_v100(graph_path: str | Path, cache_path: str | Path) -> dict[str, Any]:
    return audit_no_flood_response_v100_fast(graph_path, cache_path)
    graph = _load_graph(graph_path)
    cache = V60TrainCache(cache_path)
    names = cache.names("D2")
    if len(names) != 144:
        raise ValueError(f"canonical Train D2 group count changed: {len(names)}")
    distances = _distance_from_endpoints(graph)
    buckets = bucket_slices_v100(72, 300)
    stats: dict[str, dict[str, dict[str, dict[str, _Reservoir]]]] = {}
    joint_counts = defaultdict(lambda: {"total": 0, "depth": 0, "volume": 0, "flow": 0})
    type_counts = defaultdict(int)
    temporal_samples: list[dict[str, Any]] = []
    candidate_count = 0
    no_flood_cells = 0
    total_flood_cells = 0
    for group_index, name in enumerate(names):
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        candidates = [i for i in entry.indices if i != ref]
        for candidate_position, candidate in enumerate(candidates):
            changed = int(_changed_actuator(arrays["settings"], ref, candidate)[0])
            actuator_type = _type_for_actuator(graph, changed)
            type_counts[actuator_type] += 1
            candidate_count += 1
            state_delta = (
                np.asarray(arrays["target_states"][candidate], dtype=np.float64)
                - np.asarray(arrays["target_states"][ref], dtype=np.float64)
            )[None, ...]
            flow_delta = (
                np.asarray(arrays["target_actuator_flows"][candidate, :, changed], dtype=np.float64)
                - np.asarray(arrays["target_actuator_flows"][ref, :, changed], dtype=np.float64)
            )[None, ...]
            flood_mask = no_flood_mask_v100(state_delta[..., CHANNELS["flood"]])
            total_flood_cells += int(flood_mask.size)
            no_flood_cells += int(flood_mask.sum())
            node_delta = {name_: state_delta[..., index] for name_, index in CHANNELS.items() if name_ != "flood"}
            deltas = {**node_delta, "actuator_flow": flow_delta}
            endpoint_masks = {
                "upstream_endpoint": np.equal(np.arange(len(graph.node_ids)), graph.actuator_upstream[changed]),
                "downstream_endpoint": np.equal(np.arange(len(graph.node_ids)), graph.actuator_downstream[changed]),
                "local_nodes": distances[changed] <= 4.0,
                "remote_nodes": distances[changed] > 4.0,
            }
            for bucket, indices in buckets.items():
                idx = np.asarray(indices, dtype=np.int64)
                bucket_flood = flood_mask[:, idx]
                bucket_deltas = {key: value[:, idx] for key, value in deltas.items()}
                for category, node_mask in endpoint_masks.items():
                    if category == "upstream_endpoint":
                        selected = np.asarray([int(graph.actuator_upstream[changed])])
                    elif category == "downstream_endpoint":
                        selected = np.asarray([int(graph.actuator_downstream[changed])])
                    else:
                        selected = np.flatnonzero(node_mask)
                    selected_flood = bucket_flood[..., selected]
                    selected_deltas = {
                        key: (value[..., selected] if key != "actuator_flow" else value)
                        for key, value in bucket_deltas.items()
                    }
                    for scope in ("all", actuator_type):
                        _summary_update(stats, scope, category, bucket, selected_deltas, selected_flood)
                no_flood_sample = bucket_flood.reshape(bucket_flood.shape[0], -1).all(axis=1)
                joint = joint_no_flood_fractions_v100(
                    bucket_deltas["depth"], bucket_deltas["volume"], bucket_deltas["actuator_flow"],
                    bucket_flood, epsilon=NO_FLOOD_EPSILON_V100,
                )
                joint_counts[("all", bucket)]["total"] += int(no_flood_sample.size)
                joint_counts[(actuator_type, bucket)]["total"] += int(no_flood_sample.size)
                for scope in ("all", actuator_type):
                    joint_counts[(scope, bucket)]["depth"] += int(round(joint["depth_active_and_flood_inactive"] * no_flood_sample.size))
                    joint_counts[(scope, bucket)]["volume"] += int(round(joint["volume_active_and_flood_inactive"] * no_flood_sample.size))
                    joint_counts[(scope, bucket)]["flow"] += int(round(joint["flow_active_and_flood_inactive"] * no_flood_sample.size))
            if group_index < 3 and candidate_position == 0:
                elapsed = np.asarray(arrays["elapsed_seconds"][ref])
                temporal_samples.append({
                    "group": name,
                    "checkpoint_seconds": float(elapsed[0]),
                    "action_interval": [float(elapsed[0]), float(elapsed[0] + 300)],
                    "target_timestamps_seconds": [float(elapsed[i + 1]) for i in range(2)],
                    "changed_actuator": changed,
                    "changed_actuator_id": graph.actuator_ids[changed],
                    "actuator_type": actuator_type,
                    "node0_target_state_first_two": np.asarray(arrays["target_states"][candidate, :2, 0, :], dtype=np.float64).tolist(),
                    "changed_flow_first_two": np.asarray(arrays["target_actuator_flows"][candidate, :2, changed], dtype=np.float64).tolist(),
                })
    summary: dict[str, Any] = {}
    for scope, categories in stats.items():
        summary[scope] = {}
        for category, bucket_data in categories.items():
            summary[scope][category] = {}
            for bucket, channel_data in bucket_data.items():
                summary[scope][category][bucket] = {name: value.result() for name, value in channel_data.items()}
    joint = {}
    for (scope, bucket), counts in joint_counts.items():
        denominator = max(int(counts["total"]), 1)
        joint.setdefault(scope, {})[bucket] = {
            "sample_count": int(counts["total"]),
            "depth_active_and_flood_inactive": float(counts["depth"] / denominator),
            "volume_active_and_flood_inactive": float(counts["volume"] / denominator),
            "flow_active_and_flood_inactive": float(counts["flow"] / denominator),
        }
    # Use the all-scope bucket totals for the headline conjunction fractions.
    all_joint = joint.get("all", {})
    all_counts = [v for v in all_joint.values()]
    total_samples = sum(int(v["sample_count"]) for v in all_counts)
    headline = {
        key: float(sum(float(v[key]) * int(v["sample_count"]) for v in all_counts) / max(total_samples, 1))
        for key in ("depth_active_and_flood_inactive", "volume_active_and_flood_inactive", "flow_active_and_flood_inactive")
    }
    return {
        "contract": "PROJECT7_STEP2_V100_NO_FLOOD_HYDRAULIC_RESPONSE_AUDIT_V1",
        "git_head": _git_head(),
        "graph_sha256": _sha256(graph_path),
        "cache_manifest_sha256": _sha256(cache_path),
        "graph": {"nodes": len(graph.node_ids), "edges": int(graph.edge_index.shape[1]), "actuators": len(graph.actuator_ids)},
        "data": {"source": "canonical Train-only D2", "groups": len(names), "candidates": candidate_count, "candidate_per_group": 24, "actuator_type_candidate_counts": dict(type_counts)},
        "units": {"depth": "m", "head": "m", "flood": "m3/s", "volume": "m3", "inflow": "m3/s", "outflow": "m3/s", "actuator_flow": "m3/s", "time": "s"},
        "state_channels": CHANNELS,
        "no_flood_contract": {"epsilon": NO_FLOOD_EPSILON_V100, "definition": "abs(delta_flood) <= epsilon per node/time cell", "total_cells": total_flood_cells, "no_flood_cells": no_flood_cells, "no_flood_fraction": no_flood_cells / max(total_flood_cells, 1)},
        "horizon_buckets": {name: [int(i) for i in indices] for name, indices in buckets.items()},
        "distributions": summary,
        "joint_fractions": {"headline_sample_level": headline, "by_scope_and_bucket": joint, "hydraulic_effect_is_not_flooding_only": bool(any(value > 0.0 for value in headline.values()))},
        "temporal_contract_samples": temporal_samples,
        "boundary": {"swmm_run": False, "validation_accessed": False, "final_accessed": False, "formal_run": False, "uses_future_truth": False, "diagnostic_only": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit_no_flood_response_v100(args.graph, args.cache_manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_json_safe({"out": str(out), "groups": report["data"]["groups"], "candidates": report["data"]["candidates"], "headline": report["joint_fractions"]["headline_sample_level"]}), indent=2))


if __name__ == "__main__":
    main()
