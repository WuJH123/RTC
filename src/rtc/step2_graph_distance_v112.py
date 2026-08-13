"""Static graph-distance diagnostics for the V11.2 influence atlas.

Distances are reporting/context features only. They must never be used as a hard
reachability cutoff for hydraulic effects.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def undirected_endpoint_hops_v112(
    edge_index: np.ndarray,
    *,
    node_count: int,
    upstream: int,
    downstream: int,
) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("V112 edge_index must be [2,E]")
    if not (0 <= upstream < node_count and 0 <= downstream < node_count):
        raise ValueError("V112 actuator endpoint outside graph")
    adjacency = [[] for _ in range(node_count)]
    for u, v in edges.T.tolist():
        if 0 <= u < node_count and 0 <= v < node_count:
            adjacency[u].append(v)
            adjacency[v].append(u)
    distance = np.full(node_count, -1, dtype=np.int32)
    queue = deque()
    for source in dict.fromkeys((int(upstream), int(downstream))):
        distance[source] = 0
        queue.append(source)
    while queue:
        current = queue.popleft()
        next_distance = int(distance[current]) + 1
        for neighbor in adjacency[current]:
            if distance[neighbor] < 0:
                distance[neighbor] = next_distance
                queue.append(neighbor)
    return distance


def all_actuator_endpoint_hops_v112(graph) -> np.ndarray:
    """Return [A,N] diagnostic minimum endpoint hop distances."""
    node_count = len(graph.node_ids)
    result = np.empty((len(graph.actuator_ids), node_count), dtype=np.int32)
    for a in range(len(graph.actuator_ids)):
        result[a] = undirected_endpoint_hops_v112(
            graph.edge_index,
            node_count=node_count,
            upstream=int(graph.actuator_upstream[a]),
            downstream=int(graph.actuator_downstream[a]),
        )
    return result


__all__ = ["all_actuator_endpoint_hops_v112", "undirected_endpoint_hops_v112"]
