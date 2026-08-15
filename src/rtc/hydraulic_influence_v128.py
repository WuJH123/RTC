"""Build a sparse Development-only hydraulic influence graph from authoritative D2 effects.

This module does not change the physical SWMM graph. It derives optional shortcut candidates
for later smoke/dev ablations using only same-prefix Development D2 counterfactuals. The
artifact is prohibited from using Validation/Final/Formal data and is never enabled in the
current full model unless separately promoted after evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .spatial_diagnostics_v128 import actuator_node_hops
from .step2_spatial_audit_v128 import _changed_actuator_per_candidate
from .step2_train_v127 import _branch_indices, _truth_node_volume

HYDRAULIC_INFLUENCE_CONTRACT = "PROJECT7_DEVELOPMENT_D2_HYDRAULIC_INFLUENCE_GRAPH_V1"


@dataclass(frozen=True)
class InfluenceEdge:
    actuator_index: int
    node_index: int
    graph_hops: int
    mean_signed_flood_volume_effect_m3: float
    mean_abs_flood_volume_effect_m3: float
    first_effect_minutes: float | None
    observations: int


def build_hydraulic_influence_edges_v128(
    cache: Any,
    names: Sequence[str],
    *,
    graph: Any,
    flood_rate_index: int = 2,
    min_graph_hops: int = 7,
    min_mean_abs_effect_m3: float = 1.0,
    flood_rate_effect_floor_m3s: float = 1.0e-4,
    top_k_per_actuator: int = 8,
) -> list[InfluenceEdge]:
    selected = [str(name) for name in names]
    if not selected or any(not name.startswith("D2::") for name in selected):
        raise ValueError("hydraulic influence graph requires non-empty Development D2 groups")
    if min_graph_hops < 1 or min_mean_abs_effect_m3 < 0 or top_k_per_actuator <= 0:
        raise ValueError("invalid hydraulic influence graph thresholds")
    node_count = len(graph.node_ids)
    distances = actuator_node_hops(
        np.asarray(graph.edge_index, dtype=np.int64),
        node_count,
        np.asarray(graph.actuator_upstream, dtype=np.int64),
        np.asarray(graph.actuator_downstream, dtype=np.int64),
    )

    signed: dict[tuple[int, int], list[float]] = {}
    first_minutes: dict[tuple[int, int], list[float]] = {}
    for name in selected:
        entry = cache.entry(name)
        order = _branch_indices(entry)
        arrays = entry.arrays
        settings = np.asarray(arrays["settings"][order], dtype=np.float32)
        actuators = _changed_actuator_per_candidate(torch.as_tensor(settings, dtype=torch.float32))
        volume = _truth_node_volume(cache, name)
        states = np.asarray(arrays["target_states"][order], dtype=np.float32)
        if states.ndim != 4 or states.shape[0] != len(order) or states.shape[2] != node_count:
            raise ValueError(f"{name}: invalid D2 state trajectory for influence audit")
        if not 0 <= int(flood_rate_index) < states.shape[-1]:
            raise ValueError("flood_rate_index outside D2 state channels")
        ref_rate = states[0, :, :, int(flood_rate_index)]
        for row, actuator in enumerate(actuators, start=1):
            effect = volume[row] - volume[0]
            rate_effect = states[row, :, :, int(flood_rate_index)] - ref_rate
            for node in range(node_count):
                if int(distances[int(actuator), node]) < int(min_graph_hops):
                    continue
                key = (int(actuator), int(node))
                signed.setdefault(key, []).append(float(effect[node]))
                hit = np.flatnonzero(np.abs(rate_effect[:, node]) > float(flood_rate_effect_floor_m3s))
                if hit.size:
                    first_minutes.setdefault(key, []).append(float((int(hit[0]) + 1) * 5))

    per_actuator: dict[int, list[InfluenceEdge]] = {}
    for (actuator, node), values in signed.items():
        sample = np.asarray(values, dtype=np.float64)
        mean_abs = float(np.mean(np.abs(sample)))
        if mean_abs < float(min_mean_abs_effect_m3):
            continue
        times = first_minutes.get((actuator, node), [])
        edge = InfluenceEdge(
            actuator_index=actuator,
            node_index=node,
            graph_hops=int(distances[actuator, node]),
            mean_signed_flood_volume_effect_m3=float(np.mean(sample)),
            mean_abs_flood_volume_effect_m3=mean_abs,
            first_effect_minutes=float(np.median(times)) if times else None,
            observations=int(sample.size),
        )
        per_actuator.setdefault(actuator, []).append(edge)

    result: list[InfluenceEdge] = []
    for actuator, edges in sorted(per_actuator.items()):
        ranked = sorted(
            edges,
            key=lambda edge: (-edge.mean_abs_flood_volume_effect_m3, edge.graph_hops, edge.node_index),
        )
        result.extend(ranked[: int(top_k_per_actuator)])
    return result


def influence_edges_to_arrays(edges: Sequence[InfluenceEdge]) -> dict[str, np.ndarray]:
    values = list(edges)
    return {
        "actuator_index": np.asarray([e.actuator_index for e in values], dtype=np.int64),
        "node_index": np.asarray([e.node_index for e in values], dtype=np.int64),
        "graph_hops": np.asarray([e.graph_hops for e in values], dtype=np.int32),
        "mean_signed_flood_volume_effect_m3": np.asarray([e.mean_signed_flood_volume_effect_m3 for e in values], dtype=np.float32),
        "mean_abs_flood_volume_effect_m3": np.asarray([e.mean_abs_flood_volume_effect_m3 for e in values], dtype=np.float32),
        "first_effect_minutes": np.asarray([np.nan if e.first_effect_minutes is None else e.first_effect_minutes for e in values], dtype=np.float32),
        "observations": np.asarray([e.observations for e in values], dtype=np.int32),
    }


__all__ = [
    "HYDRAULIC_INFLUENCE_CONTRACT",
    "InfluenceEdge",
    "build_hydraulic_influence_edges_v128",
    "influence_edges_to_arrays",
]
