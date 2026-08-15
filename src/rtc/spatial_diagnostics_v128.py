"""Spatial-distance diagnostics for Project7 sparse sensing and action effects.

These helpers never alter the model or SWMM truth.  They quantify whether errors increase
with graph distance, which is the required P0 evidence before promoting global-attention,
edge-aware, or influence-graph experiments.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping, Sequence

import numpy as np

SPATIAL_DIAGNOSTICS_CONTRACT = "PROJECT7_SPATIAL_DISTANCE_DIAGNOSTICS_V1"
DEFAULT_DISTANCE_BINS = ((0, 0), (1, 3), (4, 6), (7, 12), (13, None))


def undirected_adjacency(edge_index: np.ndarray, node_count: int) -> tuple[tuple[int, ...], ...]:
    edge = np.asarray(edge_index, dtype=np.int64)
    if edge.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    neighbors: list[set[int]] = [set() for _ in range(int(node_count))]
    for src, dst in zip(edge[0], edge[1], strict=True):
        a, b = int(src), int(dst)
        if not (0 <= a < node_count and 0 <= b < node_count):
            raise ValueError("edge_index contains node outside node_count")
        neighbors[a].add(b)
        neighbors[b].add(a)
    return tuple(tuple(sorted(row)) for row in neighbors)


def nearest_source_hops(
    edge_index: np.ndarray,
    node_count: int,
    source_indices: Iterable[int],
) -> np.ndarray:
    adjacency = undirected_adjacency(edge_index, node_count)
    distance = np.full(int(node_count), np.iinfo(np.int32).max, dtype=np.int32)
    queue: deque[int] = deque()
    for raw in sorted({int(v) for v in source_indices}):
        if not 0 <= raw < node_count:
            raise ValueError(f"source index outside graph: {raw}")
        distance[raw] = 0
        queue.append(raw)
    if not queue:
        raise ValueError("at least one source node is required")
    while queue:
        src = queue.popleft()
        candidate = int(distance[src]) + 1
        for dst in adjacency[src]:
            if candidate < int(distance[dst]):
                distance[dst] = candidate
                queue.append(dst)
    return distance


def actuator_node_hops(
    edge_index: np.ndarray,
    node_count: int,
    actuator_upstream: Sequence[int],
    actuator_downstream: Sequence[int],
) -> np.ndarray:
    up = np.asarray(actuator_upstream, dtype=np.int64).reshape(-1)
    down = np.asarray(actuator_downstream, dtype=np.int64).reshape(-1)
    if up.shape != down.shape:
        raise ValueError("actuator upstream/downstream arrays must align")
    rows = [nearest_source_hops(edge_index, node_count, (int(a), int(b))) for a, b in zip(up, down, strict=True)]
    return np.stack(rows, axis=0) if rows else np.empty((0, int(node_count)), dtype=np.int32)


def distance_bin_name(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}")


def _bin_mask(distance: np.ndarray, lo: int, hi: int | None) -> np.ndarray:
    mask = distance >= int(lo)
    if hi is not None:
        mask &= distance <= int(hi)
    return mask


def error_by_distance(
    truth: np.ndarray,
    prediction: np.ndarray,
    distance: np.ndarray,
    *,
    bins: Sequence[tuple[int, int | None]] = DEFAULT_DISTANCE_BINS,
) -> dict[str, dict[str, float | int]]:
    true = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    dist = np.asarray(distance, dtype=np.int32).reshape(-1)
    if true.shape != pred.shape or true.shape[-1] != dist.size:
        raise ValueError("truth/prediction must align and have node axis last")
    result: dict[str, dict[str, float | int]] = {}
    for lo, hi in bins:
        mask = _bin_mask(dist, lo, hi)
        key = distance_bin_name(lo, hi)
        if not np.any(mask):
            result[key] = {"nodes": 0, "mae": float("nan"), "rmse": float("nan")}
            continue
        delta = pred[..., mask] - true[..., mask]
        result[key] = {
            "nodes": int(mask.sum()),
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
        }
    return result


def action_effect_by_distance(
    true_reference_node_volume_m3: np.ndarray,
    true_candidate_node_volume_m3: np.ndarray,
    predicted_reference_node_volume_m3: np.ndarray,
    predicted_candidate_node_volume_m3: np.ndarray,
    candidate_actuator_indices: Sequence[int],
    actuator_distance_hops: np.ndarray,
    *,
    bins: Sequence[tuple[int, int | None]] = DEFAULT_DISTANCE_BINS,
    effect_floor_m3: float = 1.0,
) -> dict[str, dict[str, float | int]]:
    true_ref = np.asarray(true_reference_node_volume_m3, dtype=np.float64).reshape(-1)
    pred_ref = np.asarray(predicted_reference_node_volume_m3, dtype=np.float64).reshape(-1)
    true_cand = np.asarray(true_candidate_node_volume_m3, dtype=np.float64)
    pred_cand = np.asarray(predicted_candidate_node_volume_m3, dtype=np.float64)
    act = np.asarray(candidate_actuator_indices, dtype=np.int64).reshape(-1)
    distances = np.asarray(actuator_distance_hops, dtype=np.int32)
    if true_cand.shape != pred_cand.shape:
        raise ValueError("candidate truth/prediction node volumes must align")
    if true_cand.ndim != 2 or true_cand.shape[1] != true_ref.size or pred_ref.size != true_ref.size:
        raise ValueError("node-volume arrays have incompatible shapes")
    if true_cand.shape[0] != act.size:
        raise ValueError("one actuator index is required per candidate")
    if distances.shape[1] != true_ref.size:
        raise ValueError("actuator-distance matrix node dimension differs")

    true_effect = true_cand - true_ref[None, :]
    pred_effect = pred_cand - pred_ref[None, :]
    result: dict[str, dict[str, float | int]] = {}
    for lo, hi in bins:
        abs_errors: list[np.ndarray] = []
        sign_hits = 0
        sign_total = 0
        true_magnitudes: list[np.ndarray] = []
        for row, actuator_index in enumerate(act):
            if not 0 <= int(actuator_index) < distances.shape[0]:
                raise ValueError("candidate actuator index outside actuator distance matrix")
            mask = _bin_mask(distances[int(actuator_index)], lo, hi)
            if not np.any(mask):
                continue
            t = true_effect[row, mask]
            p = pred_effect[row, mask]
            abs_errors.append(np.abs(p - t))
            true_magnitudes.append(np.abs(t))
            informative = np.abs(t) > float(effect_floor_m3)
            if np.any(informative):
                sign_hits += int(np.sum(np.sign(p[informative]) == np.sign(t[informative])))
                sign_total += int(np.sum(informative))
        key = distance_bin_name(lo, hi)
        if not abs_errors:
            result[key] = {
                "candidate_node_pairs": 0,
                "effect_mae_m3": float("nan"),
                "effect_relative_mae": float("nan"),
                "informative_sign_total": 0,
                "effect_sign_accuracy": float("nan"),
            }
            continue
        errors = np.concatenate(abs_errors)
        magnitudes = np.concatenate(true_magnitudes)
        result[key] = {
            "candidate_node_pairs": int(errors.size),
            "effect_mae_m3": float(np.mean(errors)),
            "effect_relative_mae": float(np.sum(errors) / max(np.sum(magnitudes), 1.0)),
            "informative_sign_total": int(sign_total),
            "effect_sign_accuracy": float(sign_hits / sign_total) if sign_total else float("nan"),
        }
    return result


def far_field_degradation_ratio(
    metrics: Mapping[str, Mapping[str, float | int]],
    *,
    near_key: str = "1-3",
    far_key: str = "13+",
) -> float:
    near = float(metrics.get(near_key, {}).get("effect_mae_m3", float("nan")))
    far = float(metrics.get(far_key, {}).get("effect_mae_m3", float("nan")))
    if not np.isfinite(near) or not np.isfinite(far) or near <= 0:
        return float("nan")
    return far / near


__all__ = [
    "DEFAULT_DISTANCE_BINS",
    "SPATIAL_DIAGNOSTICS_CONTRACT",
    "action_effect_by_distance",
    "actuator_node_hops",
    "error_by_distance",
    "far_field_degradation_ratio",
    "nearest_source_hops",
    "undirected_adjacency",
]
