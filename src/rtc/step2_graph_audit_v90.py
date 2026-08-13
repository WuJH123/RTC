"""Read-only receptive-field census for V9 direct hydraulic effects.

The helper deliberately evaluates the frozen cache rather than model output.  It
answers a structural question: how much authoritative single-actuator response
mass can an endpoint-seeded message path represent at a given graph radius?
"""
from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np


STATE_CHANNELS_V90 = {
    "delta_depth_m": 0,
    "delta_flood_m3s": 2,
    "delta_storage_m3": 3,
}


def undirected_adjacency_v90(edge_index: np.ndarray, *, node_count: int) -> list[list[int]]:
    """Build the topology reachable by the current bidirectional V9 graph."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    adjacency = [set() for _ in range(int(node_count))]
    for source, destination in edges.T:
        if source < 0 or destination < 0 or source >= node_count or destination >= node_count:
            raise ValueError("edge_index contains a node outside node_count")
        adjacency[int(source)].add(int(destination))
        adjacency[int(destination)].add(int(source))
    return [sorted(neighbours) for neighbours in adjacency]


def endpoint_distances_v90(
    adjacency: Sequence[Sequence[int]],
    actuator_upstream: np.ndarray,
    actuator_downstream: np.ndarray,
) -> np.ndarray:
    """Return shortest distance from each actuator's two endpoint sources."""
    upstream = np.asarray(actuator_upstream, dtype=np.int64).reshape(-1)
    downstream = np.asarray(actuator_downstream, dtype=np.int64).reshape(-1)
    if upstream.shape != downstream.shape:
        raise ValueError("actuator endpoint arrays differ in shape")
    nodes = len(adjacency)
    if np.any(upstream < 0) or np.any(downstream < 0) or np.any(upstream >= nodes) or np.any(downstream >= nodes):
        raise ValueError("actuator endpoint lies outside graph")
    result = np.full((upstream.size, nodes), np.inf, dtype=np.float64)
    for actuator, (first, second) in enumerate(zip(upstream, downstream, strict=True)):
        queue: deque[int] = deque()
        for source in {int(first), int(second)}:
            result[actuator, source] = 0.0
            queue.append(source)
        while queue:
            source = queue.popleft()
            for destination in adjacency[source]:
                if not np.isfinite(result[actuator, destination]):
                    result[actuator, destination] = result[actuator, source] + 1.0
                    queue.append(destination)
    return result


def receptive_field_mass_v90(
    *,
    distances_by_changed_actuator: np.ndarray,
    changed_actuators: np.ndarray,
    delta_states: np.ndarray,
    delta_flows: np.ndarray,
    actuator_upstream: np.ndarray,
    actuator_downstream: np.ndarray,
    hops: Sequence[int] = (4, 8),
) -> dict[str, dict[str, float]]:
    """Census true absolute effect mass within each endpoint graph radius.

    ``delta_states`` is [candidate, retained_time, node, six-state-channel] and
    ``delta_flows`` is [candidate, retained_time, actuator].  Every candidate
    must identify exactly one changed actuator, which is the D2 contract.
    """
    distances = np.asarray(distances_by_changed_actuator, dtype=np.float64)
    changed = np.asarray(changed_actuators, dtype=np.int64).reshape(-1)
    states = np.asarray(delta_states, dtype=np.float64)
    flows = np.asarray(delta_flows, dtype=np.float64)
    upstream = np.asarray(actuator_upstream, dtype=np.int64).reshape(-1)
    downstream = np.asarray(actuator_downstream, dtype=np.int64).reshape(-1)
    if states.ndim != 4 or states.shape[-1] < 4:
        raise ValueError("delta_states must be [candidate,time,node,>=4]")
    if flows.ndim != 3:
        raise ValueError("delta_flows must be [candidate,time,actuator]")
    candidates, _, nodes, _ = states.shape
    if changed.shape != (candidates,):
        raise ValueError("changed actuator count differs from candidate count")
    if distances.ndim != 2 or distances.shape[1] != nodes:
        raise ValueError("distance matrix does not match state node count")
    if flows.shape[0] != candidates or flows.shape[2] != upstream.size or upstream.shape != downstream.shape:
        raise ValueError("flow/actuator endpoint schema mismatch")
    if np.any(changed < 0) or np.any(changed >= distances.shape[0]):
        raise ValueError("changed actuator lies outside distance matrix")

    endpoint_distance = np.minimum(distances[:, upstream], distances[:, downstream])
    result: dict[str, dict[str, float]] = {}
    for hop in sorted({int(value) for value in hops}):
        if hop < 0:
            raise ValueError("hop must be non-negative")
        node_mask = distances[changed] <= hop  # [candidate,node]
        flow_mask = endpoint_distance[changed] <= hop  # [candidate,actuator]
        item: dict[str, float] = {
            "node_coverage_fraction": float(np.mean(distances <= hop)),
            "changed_endpoint_actuator_coverage_fraction": float(np.mean(endpoint_distance <= hop)),
        }
        for label, channel in STATE_CHANNELS_V90.items():
            absolute = np.abs(states[..., channel])
            total = float(absolute.sum())
            inside = float((absolute * node_mask[:, None, :]).sum())
            item[f"{label}_inside_mass_fraction"] = inside / total if total > 0.0 else float("nan")
            item[f"{label}_outside_mass_fraction"] = 1.0 - inside / total if total > 0.0 else float("nan")
        absolute_flow = np.abs(flows)
        total_flow = float(absolute_flow.sum())
        inside_flow = float((absolute_flow * flow_mask[:, None, :]).sum())
        item["delta_managed_flow_m3s_inside_mass_fraction"] = (
            inside_flow / total_flow if total_flow > 0.0 else float("nan")
        )
        item["delta_managed_flow_m3s_outside_mass_fraction"] = (
            1.0 - inside_flow / total_flow if total_flow > 0.0 else float("nan")
        )
        result[f"h{hop}"] = item
    return result


__all__ = [
    "STATE_CHANNELS_V90",
    "endpoint_distances_v90",
    "receptive_field_mass_v90",
    "undirected_adjacency_v90",
]
