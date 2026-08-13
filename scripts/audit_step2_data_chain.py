"""Read-only Project7 Step2 Train-data forensic audit.

The audit intentionally starts from the frozen V60 mmap cache *and* recomputes a
deterministic set of candidate/reference targets directly from compact SWMM
trajectories.  It never launches SWMM and it rejects any non-development/train row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rtc.dataset_compile import compile_branch_tensors
from rtc.graph import GraphSchema
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70


AUDIT_CONTRACT = "PROJECT7_STEP2_DATA_AUDIT_V1"
EPSILON = 1e-8
HORIZON_BUCKETS = (
    ("0_30_min", 0, 30),
    ("30_120_min", 30, 120),
    ("120_360_min", 120, 360),
)
STATE_CHANNELS = ((0, "depth_m"), (2, "flooding_m3s"), (3, "storage_m3"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_graph(path: Path) -> GraphSchema:
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


def _actuator_type(graph: GraphSchema, actuator: int) -> str:
    names = tuple(graph.actuator_physics_feature_names)
    row = np.asarray(graph.actuator_physics[int(actuator)])
    for name, label in (("is_pump", "pump"), ("is_orifice", "orifice"), ("is_weir", "weir")):
        if name in names and float(row[names.index(name)]) > 0.5:
            return label
    return "other"


def _adjacency(graph: GraphSchema) -> list[list[int]]:
    result = [[] for _ in graph.node_ids]
    for src, dst in np.asarray(graph.edge_index, dtype=np.int64).T:
        result[int(src)].append(int(dst))
        result[int(dst)].append(int(src))
    return result


def _distances(adjacency: list[list[int]], source: int) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    distance[int(source)] = 0.0
    queue: deque[int] = deque([int(source)])
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if not np.isfinite(distance[nxt]):
                distance[nxt] = distance[current] + 1.0
                queue.append(nxt)
    return distance


def _distance_bucket(distance: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "endpoint": distance == 0.0,
        "one_hop": distance == 1.0,
        "two_hop": distance == 2.0,
        "three_hop": distance == 3.0,
        "greater_than_three_hop": np.isfinite(distance) & (distance > 3.0),
        "unreachable": ~np.isfinite(distance),
    }


@dataclass
class _Moments:
    """Exact moments plus a small deterministic, group-balanced quantile sample."""

    count: int = 0
    total: float = 0.0
    square: float = 0.0
    absolute: float = 0.0
    positive: int = 0
    negative: int = 0
    near_zero: int = 0
    active: int = 0
    maximum_abs: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)

    def update(self, values: np.ndarray, *, scale: float | np.ndarray | None = None) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if not flat.size:
            return
        if not np.isfinite(flat).all():
            raise ValueError("non-finite physical effect encountered during data audit")
        self.count += int(flat.size)
        self.total += float(flat.sum())
        self.square += float(np.square(flat).sum())
        self.absolute += float(np.abs(flat).sum())
        self.positive += int((flat > EPSILON).sum())
        self.negative += int((flat < -EPSILON).sum())
        self.near_zero += int((np.abs(flat) <= EPSILON).sum())
        self.maximum_abs = max(self.maximum_abs, float(np.abs(flat).max()))
        if scale is not None:
            scale_array = np.broadcast_to(
                np.maximum(np.asarray(scale, dtype=np.float64), 1e-12), np.asarray(values).shape
            )
            self.active += int((np.abs(values) / scale_array >= 0.25).sum())
        # Group-balanced bounded sample: no one spatially large group can dominate all
        # quantiles.  Exact moments above remain the scientific primary statistics.
        stride = max(1, int(math.ceil(flat.size / 512)))
        self.samples.append(np.asarray(flat[::stride][:512], dtype=np.float64))

    def report(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "median": None}
        sample = np.concatenate(self.samples) if self.samples else np.empty(0, dtype=np.float64)
        mean = self.total / self.count
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(max(self.square / self.count - mean * mean, 0.0)),
            "mean_abs": self.absolute / self.count,
            "rms": math.sqrt(max(self.square / self.count, 0.0)),
            "median": float(np.quantile(sample, 0.50)) if sample.size else None,
            "p90": float(np.quantile(sample, 0.90)) if sample.size else None,
            "p95": float(np.quantile(sample, 0.95)) if sample.size else None,
            "p99": float(np.quantile(sample, 0.99)) if sample.size else None,
            "max_abs": self.maximum_abs,
            "positive_fraction": self.positive / self.count,
            "negative_fraction": self.negative / self.count,
            "near_zero_fraction": self.near_zero / self.count,
            "active_fraction": self.active / self.count,
            "quantile_method": "deterministic_group_balanced_reservoir",
        }


@dataclass
class _ScalarAccumulator:
    values: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        if np.isfinite(value):
            self.values.append(float(value))

    def report(self) -> dict[str, float | int | None]:
        value = np.asarray(self.values, dtype=np.float64)
        if not value.size:
            return {"count": 0, "mean": None, "median": None, "p95": None}
        return {
            "count": int(value.size),
            "mean": float(value.mean()),
            "median": float(np.median(value)),
            "p95": float(np.quantile(value, 0.95)),
            "max": float(value.max()),
        }


def _empty_effect_accumulators() -> dict[str, _Moments]:
    return {label: _Moments() for _, label in STATE_CHANNELS} | {"managed_flow_m3s": _Moments()}


def _entry_key(entry: Any, action: str) -> tuple[str, str, str]:
    return (str(entry.source_kind).upper(), str(entry.checkpoint_id), str(action))


def _cache_rows(cache: V60TrainCache) -> Iterable[tuple[str, Any, int]]:
    for name in cache.names():
        entry = cache.entry(name)
        for index in entry.indices:
            yield name, entry, int(index)


def _metadata_index(run_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in run_rows:
        source = str(row.get("source_kind", "")).upper()
        action = str(row.get("action_or_sequence_sha256") or row.get("candidate_action_sha256"))
        key = (source, str(row.get("checkpoint_id", "")), action)
        if key in result:
            raise ValueError(f"duplicate run-index action lineage: {key}")
        result[key] = row
    return result


def _selected_raw_groups(cache: V60TrainCache, source: str, count: int) -> list[str]:
    names = cache.names(source)
    by_event: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_event[cache.entry(name).event_id].append(name)
    events = sorted(by_event)
    if not events:
        return []
    positions = np.linspace(0, len(events) - 1, min(count, len(events)), dtype=int)
    selected: list[str] = []
    for position in positions.tolist():
        choices = sorted(by_event[events[position]])
        selected.append(choices[len(choices) // 2])
    return selected


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _raw_recompute(
    cache: V60TrainCache,
    metadata: dict[tuple[str, str, str], dict[str, str]],
    *,
    groups_per_source: int,
) -> dict[str, Any]:
    fields = (
        "initial_state",
        "rainfall",
        "settings",
        "previous_actuator_flow",
        "target_states",
        "target_actuator_flows",
        "elapsed_seconds",
        "exact_node_flood_volume_m3",
    )
    max_error = {name: 0.0 for name in fields}
    effect_error = {"state": 0.0, "flow": 0.0}
    mismatch = 0
    branches = 0
    pairs = 0
    timeline_samples: list[dict[str, Any]] = []
    groups_checked: list[str] = []
    for source in ("D2", "D3"):
        for name in _selected_raw_groups(cache, source, groups_per_source):
            entry = cache.entry(name)
            compiled: dict[int, Any] = {}
            groups_checked.append(name)
            for index in entry.indices:
                arrays = entry.arrays
                action = str(arrays["action_or_sequence_sha256"][index])
                row = metadata.get(_entry_key(entry, action))
                if row is None:
                    raise ValueError(f"missing raw lineage row for {name} {action}")
                branch = compile_branch_tensors(Path(str(row["metadata_path"])))
                compiled[int(index)] = branch
                branches += 1
                expected = {
                    "initial_state": arrays["initial_state"][index],
                    "rainfall": arrays["rainfall"][index],
                    "settings": arrays["settings"][index],
                    "previous_actuator_flow": arrays["previous_actuator_flow"][index],
                    "target_states": arrays["target_states"][index],
                    "target_actuator_flows": arrays["target_actuator_flows"][index],
                    "elapsed_seconds": arrays["elapsed_seconds"][index],
                    "exact_node_flood_volume_m3": arrays["exact_node_flood_volume_m3"][index],
                }
                actual = {field: getattr(branch, field) for field in fields}
                for field in fields:
                    error = _max_abs(np.asarray(actual[field]), np.asarray(expected[field]))
                    max_error[field] = max(max_error[field], error)
                    mismatch += int(error > 1e-6)
                if len(timeline_samples) < 6:
                    elapsed = np.asarray(branch.elapsed_seconds, dtype=np.int64)
                    timeline_samples.append(
                        {
                            "source": source,
                            "group": name,
                            "metadata_path": str(row["metadata_path"]),
                            "initial_state_timestamp_seconds": int(elapsed[0]),
                            "action_start_timestamp_seconds": int(elapsed[0]),
                            "first_action_interval_seconds": [int(elapsed[0]), int(elapsed[1])],
                            "target_0_timestamp_seconds": int(elapsed[1]),
                            "target_1_timestamp_seconds": int(elapsed[2]),
                            "last_target_timestamp_seconds": int(elapsed[-1]),
                            "horizon_frames": int(branch.target_states.shape[0]),
                            "model_step_seconds": int(branch.model_step_seconds),
                        }
                    )
            reference = compiled[int(entry.reference_index)]
            for index, branch in compiled.items():
                if index == entry.reference_index:
                    continue
                pairs += 1
                state_error = _max_abs(
                    branch.target_states - reference.target_states,
                    entry.arrays["target_states"][index] - entry.arrays["target_states"][entry.reference_index],
                )
                flow_error = _max_abs(
                    branch.target_actuator_flows - reference.target_actuator_flows,
                    entry.arrays["target_actuator_flows"][index] - entry.arrays["target_actuator_flows"][entry.reference_index],
                )
                effect_error["state"] = max(effect_error["state"], state_error)
                effect_error["flow"] = max(effect_error["flow"], flow_error)
                mismatch += int(state_error > 1e-6 or flow_error > 1e-6)
    return {
        "scope": "deterministic_raw_compact_recompile_of_all_25_rows_per_selected_group",
        "groups_checked": groups_checked,
        "branches_checked": branches,
        "candidate_reference_pairs_checked": pairs,
        "max_abs_error_by_tensor": max_error,
        "candidate_reference_effect_max_abs_error": effect_error,
        "mismatch_comparisons_over_1e-6": mismatch,
        "timeline_samples": timeline_samples,
    }


def _scan_metadata_and_pairing(
    cache: V60TrainCache,
    metadata: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, Any]:
    unit_counts: dict[str, Counter[str]] = {
        "system_units": Counter(),
        "flow_units": Counter(),
        "controller_tensor_units": Counter(),
        "swmm_engine_version": Counter(),
    }
    same_prefix = Counter()
    context_checks = 0
    context_violations: list[dict[str, Any]] = []
    tensor_max = {"initial_state": 0.0, "rainfall": 0.0, "previous_actuator_flow": 0.0, "elapsed_seconds": 0.0}
    action_nonzero = 0
    action_total = 0
    reference_counts = Counter()
    metadata_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, row in metadata.items():
        path = Path(str(row["metadata_path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata_records[key] = payload
        for field in unit_counts:
            unit_counts[field][str(payload.get(field, ""))] += 1
        same_prefix[str(payload.get("same_prefix_verification", {}).get("passed"))] += 1

    lineage_fields = (
        "event_id",
        "rainfall_group",
        "checkpoint_id",
        "reference_compact_sha256",
        "reference_metadata_sha256",
        "runtime_inp_sha256",
        "source_inp_sha256",
        "reference_swmm_engine_version",
    )
    for name in cache.names():
        entry = cache.entry(name)
        arrays = entry.arrays
        ref = int(entry.reference_index)
        reference_counts[entry.source_kind] += 1
        ref_action = str(arrays["action_or_sequence_sha256"][ref])
        ref_meta = metadata_records[_entry_key(entry, ref_action)]
        ref_lineage = dict(ref_meta.get("generation_lineage", {}))
        for index in entry.indices:
            if index == ref:
                continue
            action_total += 1
            action = str(arrays["action_or_sequence_sha256"][index])
            candidate_meta = metadata_records[_entry_key(entry, action)]
            candidate_lineage = dict(candidate_meta.get("generation_lineage", {}))
            reasons: list[str] = []
            if candidate_meta.get("same_prefix_verification", {}).get("passed") is not True:
                reasons.append("same_prefix_verification_failed")
            for field in lineage_fields:
                if str(candidate_lineage.get(field, "")) != str(ref_lineage.get(field, "")):
                    reasons.append(f"lineage_{field}_mismatch")
            for field in tensor_max:
                tensor_max[field] = max(
                    tensor_max[field],
                    _max_abs(arrays[field][index], arrays[field][ref]),
                )
            action_delta = np.asarray(arrays["settings"][index], dtype=np.float64) - np.asarray(arrays["settings"][ref], dtype=np.float64)
            if float(np.abs(action_delta).max()) > EPSILON:
                action_nonzero += 1
            else:
                reasons.append("candidate_action_equals_reference")
            context_checks += 1
            if reasons:
                context_violations.append({"group": name, "action": action, "reasons": reasons})
    return {
        "metadata_records": len(metadata_records),
        "same_prefix_verification": dict(same_prefix),
        "metadata_unit_counts": {key: dict(value) for key, value in unit_counts.items()},
        "group_reference_counts": dict(reference_counts),
        "candidate_reference_context_checks": context_checks,
        "candidate_reference_context_violations": len(context_violations),
        "candidate_reference_context_violation_examples": context_violations[:20],
        "candidate_reference_context_max_abs_tensor_difference": tensor_max,
        "candidate_action_nonzero": action_nonzero,
        "candidate_action_total": action_total,
    }


def _effect_distribution(
    cache: V60TrainCache,
    graph: GraphSchema,
) -> dict[str, Any]:
    all_names = cache.names()
    fit, holdout = deterministic_rainfall_split_v60(cache, names=all_names, holdout_fraction=0.20)
    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    source_stats = {source: _empty_effect_accumulators() for source in ("D2", "D3")}
    horizon_stats = {
        source: {label: _empty_effect_accumulators() for label, _, _ in HORIZON_BUCKETS}
        for source in ("D2", "D3")
    }
    action_rows: dict[str, list[dict[str, Any]]] = {"D2": [], "D3": []}
    type_stats: dict[str, dict[str, dict[str, _ScalarAccumulator]]] = {
        source: defaultdict(lambda: defaultdict(_ScalarAccumulator)) for source in ("D2", "D3")
    }
    identity_stats: dict[str, dict[str, dict[str, _ScalarAccumulator]]] = {
        source: defaultdict(lambda: defaultdict(_ScalarAccumulator)) for source in ("D2", "D3")
    }
    flow_sign: dict[str, _Moments] = defaultdict(_Moments)
    distance_stats: dict[str, dict[str, dict[str, _ScalarAccumulator]]] = {
        source: {label: defaultdict(_ScalarAccumulator) for _, label in STATE_CHANNELS}
        for source in ("D2", "D3")
    }
    adjacency = _adjacency(graph)
    node_distance_cache: dict[int, np.ndarray] = {}
    bucket_masks = {
        label: ((np.arange(72) + 1) * 5 > start) & ((np.arange(72) + 1) * 5 <= end)
        for label, start, end in HORIZON_BUCKETS
    }
    candidates_by_source = Counter()
    active_counts: dict[str, list[int]] = {"D2": [], "D3": []}

    for name in all_names:
        entry = cache.entry(name)
        source = str(entry.source_kind).upper()
        arrays, ref = entry.arrays, int(entry.reference_index)
        candidates = [int(index) for index in entry.indices if int(index) != ref]
        state_delta = np.asarray(arrays["target_states"][candidates], dtype=np.float64) - np.asarray(arrays["target_states"][ref], dtype=np.float64)[None]
        flow_delta = np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64) - np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)[None]
        setting_delta = np.asarray(arrays["settings"][candidates], dtype=np.float64) - np.asarray(arrays["settings"][ref], dtype=np.float64)[None]
        source_flow = np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64)
        for actuator, actuator_id in enumerate(graph.actuator_ids):
            flow_sign[_actuator_type(graph, actuator)].update(source_flow[..., actuator])
        for channel, label in STATE_CHANNELS:
            values = state_delta[..., channel]
            scale = float(scales.state_delta_scale[channel])
            source_stats[source][label].update(values, scale=scale)
            for bucket, mask in bucket_masks.items():
                horizon_stats[source][bucket][label].update(values[:, mask], scale=scale)
        source_stats[source]["managed_flow_m3s"].update(
            flow_delta,
            scale=np.asarray(scales.flow_delta_scale, dtype=np.float64)[None, None, :],
        )
        for bucket, mask in bucket_masks.items():
            horizon_stats[source][bucket]["managed_flow_m3s"].update(
                flow_delta[:, mask],
                scale=np.asarray(scales.flow_delta_scale, dtype=np.float64)[None, None, :],
            )

        # Spatial distance is summarized as candidate-level mean absolute node effect,
        # not as a dense cell count.  This preserves localisation information without
        # letting the 932-node tensor size itself dominate the comparison.
        node_mass = np.mean(np.abs(state_delta[..., [0, 2, 3]]), axis=1)
        for candidate_local, index in enumerate(candidates):
            changed = np.flatnonzero(np.any(np.abs(setting_delta[candidate_local]) > EPSILON, axis=0))
            changed_count = int(changed.size)
            active_counts[source].append(changed_count)
            candidates_by_source[source] += 1
            action_l1 = float(np.abs(setting_delta[candidate_local]).sum())
            response = {
                label: float(np.sqrt(np.mean(np.square(state_delta[candidate_local, ..., channel]))))
                for channel, label in STATE_CHANNELS
            }
            response["managed_flow_m3s"] = float(np.sqrt(np.mean(np.square(flow_delta[candidate_local]))))
            action_rows[source].append(
                {
                    "action_l1": action_l1,
                    "changed_actuator_count": changed_count,
                    "responses": response,
                }
            )
            if changed_count == 1:
                actuator = int(changed.item())
                actuator_id = str(graph.actuator_ids[actuator])
                kind = _actuator_type(graph, actuator)
                for label, value in response.items():
                    type_stats[source][kind][label].update(value)
                    identity_stats[source][actuator_id][label].update(value)
            if changed_count:
                endpoint_distances = []
                for actuator in changed.tolist():
                    for node in (int(graph.actuator_upstream[actuator]), int(graph.actuator_downstream[actuator])):
                        if node not in node_distance_cache:
                            node_distance_cache[node] = _distances(adjacency, node)
                        endpoint_distances.append(node_distance_cache[node])
                nearest = np.min(np.stack(endpoint_distances, axis=0), axis=0)
                for bucket, mask in _distance_bucket(nearest).items():
                    if not mask.any():
                        continue
                    for position, (_, label) in enumerate(STATE_CHANNELS):
                        distance_stats[source][label][bucket].update(
                            float(node_mass[candidate_local, mask, position].mean())
                        )

    action_magnitude: dict[str, Any] = {}
    for source, rows in action_rows.items():
        l1 = np.asarray([row["action_l1"] for row in rows], dtype=np.float64)
        q33, q67 = (float(np.quantile(l1, q)) for q in (1 / 3, 2 / 3))
        masks = {
            "small": l1 < q33,
            "medium": (l1 >= q33) & (l1 < q67),
            "large": l1 >= q67,
        }
        action_magnitude[source] = {
            "l1_q33": q33,
            "l1_q67": q67,
            "strata": {
                label: {
                    "candidate_count": int(mask.sum()),
                    "response_rms": {
                        channel: _ScalarAccumulator([
                            float(rows[index]["responses"][channel])
                            for index in np.flatnonzero(mask).tolist()
                        ]).report()
                        for channel in _empty_effect_accumulators()
                    },
                }
                for label, mask in masks.items()
            },
        }

    return {
        "scale_provenance": {
            "source": "deterministic TrainFit D2+D3 only; common frozen V9 diagnostic scale",
            "fit_group_count": len(fit),
            "holdout_group_count": len(holdout),
            "state_delta_scale": np.asarray(scales.state_delta_scale, dtype=float).tolist(),
            "flow_delta_scale_summary": {
                "min": float(np.min(scales.flow_delta_scale)),
                "median": float(np.median(scales.flow_delta_scale)),
                "p99": float(np.quantile(scales.flow_delta_scale, 0.99)),
                "max": float(np.max(scales.flow_delta_scale)),
            },
        },
        "candidate_counts": dict(candidates_by_source),
        "effect_distribution": {
            source: {channel: accumulator.report() for channel, accumulator in values.items()}
            for source, values in source_stats.items()
        },
        "horizon_buckets": {
            source: {
                bucket: {channel: accumulator.report() for channel, accumulator in values.items()}
                for bucket, values in buckets.items()
            }
            for source, buckets in horizon_stats.items()
        },
        "active_actuator_count": {
            source: _ScalarAccumulator([float(value) for value in values]).report()
            for source, values in active_counts.items()
        },
        "action_magnitude": action_magnitude,
        "single_actuator_type_response_rms": {
            source: {
                kind: {channel: accumulator.report() for channel, accumulator in channels.items()}
                for kind, channels in values.items()
            }
            for source, values in type_stats.items()
        },
        "single_actuator_identity_response_rms": {
            source: {
                actuator_id: {channel: accumulator.report() for channel, accumulator in channels.items()}
                for actuator_id, channels in values.items()
            }
            for source, values in identity_stats.items()
        },
        "mean_abs_node_effect_by_distance_from_changed_endpoint": {
            source: {
                channel: {bucket: accumulator.report() for bucket, accumulator in buckets.items()}
                for channel, buckets in channels.items()
            }
            for source, channels in distance_stats.items()
        },
        "authoritative_managed_flow_sign_by_actuator_type": {
            kind: accumulator.report() for kind, accumulator in flow_sign.items()
        },
    }


def _tfv_summary(cache: V60TrainCache) -> dict[str, Any]:
    result = {"D2": _Moments(), "D3": _Moments()}
    for name in cache.names():
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        candidates = [index for index in entry.indices if index != ref]
        reference = float(np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum())
        values = np.asarray(arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64).sum(axis=1) - reference
        result[entry.source_kind].update(values)
    return {source: values.report() for source, values in result.items()}


def _asset_inventory(study_root: Path) -> dict[str, Any]:
    allowed = []
    for path in sorted(study_root.iterdir()):
        lower = path.name.lower()
        if any(token in lower for token in ("validation", "final", "formal")):
            continue
        if any(token in lower for token in ("d0", "d1", "d2", "d3", "step1", "step2", "model", "cache", "shard")):
            allowed.append({
                "path": str(path),
                "kind": "directory" if path.is_dir() else "file",
            })
    return {
        "scope": "top-level Train/development-relevant asset inventory; Validation/Final/Formal excluded by path",
        "items": allowed,
    }


def audit_step2_data_chain(
    *,
    study_root: Path,
    graph_path: Path,
    cache_manifest: Path,
    run_index: Path,
    raw_groups_per_source: int,
) -> dict[str, Any]:
    graph = _load_graph(graph_path)
    cache = V60TrainCache(cache_manifest)
    cache_lineage = validate_v60_cache_lineage(cache_manifest)
    rows = _read_csv(run_index)
    if not rows:
        raise ValueError("Step2 run index is empty")
    invalid = [
        row for row in rows
        if str(row.get("scientific_split", "")).lower() != "development"
        or str(row.get("development_fold", "")).lower() != "train"
    ]
    if invalid:
        raise ValueError("data audit refuses non-development/train run-index rows")
    metadata = _metadata_index(rows)
    pairing = _scan_metadata_and_pairing(cache, metadata)
    raw = _raw_recompute(
        cache, metadata, groups_per_source=raw_groups_per_source
    )
    distribution = _effect_distribution(cache, graph)
    d2_names, d3_names = cache.names("D2"), cache.targeted_d3_names()
    group_sizes = {
        source: Counter(len(cache.entry(name).indices) for name in names)
        for source, names in (("D2", d2_names), ("D3", d3_names))
    }
    actuator_types = Counter(_actuator_type(graph, index) for index in range(len(graph.actuator_ids)))
    source_file = Path(__file__).resolve().parents[1] / "src" / "rtc"
    pfv_hits = []
    for path in source_file.glob("*.py"):
        if path.name in {"step2_control_response_v70.py", "step2_control_response_v80.py", "step2_control_response_v90.py", "step2_hydraulic_objective_v80.py", "step2_hydraulic_objective_v90.py", "step2_optimization_v90.py"}:
            if "pfv" in path.read_text(encoding="utf-8").lower():
                pfv_hits.append(str(path))
    return {
        "contract": AUDIT_CONTRACT,
        "scope": {
            "development_train_only": True,
            "swmm_run": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
        "lineage": {
            "git_head": _git_head(),
            "graph_path": str(graph_path),
            "graph_sha256": _sha256(graph_path),
            "cache_manifest": str(cache_manifest),
            "cache_manifest_sha256": _sha256(cache_manifest),
            "run_index": str(run_index),
            "run_index_sha256": _sha256(run_index),
            "v60_cache_lineage": cache_lineage,
        },
        "asset_inventory": _asset_inventory(study_root),
        "data_fact_table": {
            "events": len({cache.entry(name).event_id for name in cache.names()}),
            "rainfall_groups": len({cache.entry(name).rainfall_group for name in cache.names()}),
            "checkpoints": len({cache.entry(name).checkpoint_id for name in cache.names()}),
            "D2_groups": len(d2_names),
            "targeted_D3_groups": len(d3_names),
            "candidate_rows": {
                "D2": sum(len(cache.entry(name).indices) - 1 for name in d2_names),
                "D3": sum(len(cache.entry(name).indices) - 1 for name in d3_names),
            },
            "reference_rows": {"D2": len(d2_names), "D3": len(d3_names)},
            "branches_per_checkpoint_group_distribution": {
                source: {str(key): int(value) for key, value in values.items()}
                for source, values in group_sizes.items()
            },
            "node_count": len(graph.node_ids),
            "actuator_count": len(graph.actuator_ids),
            "actuator_type_distribution": dict(actuator_types),
            "model_horizon_frames": 72,
            "hydraulic_sampling_seconds": 300,
            "control_blocks": 36,
            "control_block_seconds": 600,
        },
        "time_contract": {
            "initial_state": "checkpoint state at t",
            "reference_and_candidate_trajectory_start": "checkpoint t; target tensor starts at t+300 s",
            "first_action_interval": "[t, t+300 s)",
            "target_index_contract": "target[0]=t+300 s; target[1]=t+600 s; target[71]=t+21600 s",
            "D3_control_expansion": "each 600 s control block is duplicated to two 300 s model steps",
            "raw_timeline_samples": raw["timeline_samples"],
        },
        "target_recomputation_from_raw_compacts": raw,
        "reference_candidate_pairing": pairing,
        "target_definition": {
            "state_effect": "raw candidate target_state minus same-prefix reference target_state",
            "managed_flow_effect": "raw candidate target_actuator_flow minus same-prefix reference target_actuator_flow",
            "state_channels": ["depth_m", "head_m", "flooding_m3s", "volume_m3", "total_inflow_m3s", "total_outflow_m3s"],
        },
        "TFV_and_PFV_semantics": {
            "authoritative_TFV": "sum of SWMM node delta_flooding_volume_m3 over complete horizon",
            "TFV_unit": "m3",
            "not_sampled_flooding_rate_integral": True,
            "V7_value_target": "direct signed candidate minus reference authoritative Delta-TFV m3",
            "PFV_in_current_V7_V8_V9_development_paths": bool(pfv_hits),
            "PFV_current_runtime_files": pfv_hits,
        },
        "units": {
            "state": "depth/head m; flooding/total inflow/total outflow m3/s; volume m3",
            "managed_actuator_flow": "m3/s signed",
            "rainfall": "mm/hr",
            "time": "s",
            "metadata_observed": pairing["metadata_unit_counts"],
        },
        "authoritative_signed_delta_TFV_distribution": _tfv_summary(cache),
        "hydraulic_effect_identifiability": distribution,
        "initial_audit_interpretation": {
            "data_alignment_bug_supported": pairing["candidate_reference_context_violations"] > 0 or raw["mismatch_comparisons_over_1e-6"] > 0,
            "targets_globally_near_zero_supported": False,
            "note": "Interpret representation sufficiency only after corrected evaluator and Train-only baseline/ladder evidence.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Project7 Step2 data-chain audit")
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-groups-per-source", type=int, default=4)
    args = parser.parse_args()
    if args.raw_groups_per_source <= 0:
        raise ValueError("raw-groups-per-source must be positive")
    report = audit_step2_data_chain(
        study_root=args.study_root,
        graph_path=args.graph,
        cache_manifest=args.cache_manifest,
        run_index=args.run_index,
        raw_groups_per_source=args.raw_groups_per_source,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_sanitize({
        "out": str(args.out),
        "groups": report["data_fact_table"]["D2_groups"] + report["data_fact_table"]["targeted_D3_groups"],
        "raw_mismatch": report["target_recomputation_from_raw_compacts"]["mismatch_comparisons_over_1e-6"],
        "pairing_violations": report["reference_candidate_pairing"]["candidate_reference_context_violations"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
