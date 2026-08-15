"""Compile current graph-edge physics from the frozen SWMM INP for development ablations.

The current physical graph may deduplicate parallel SWMM links between the same node pair.
This builder therefore aligns an aggregate physical descriptor to every directed edge already
present in ``GraphSchema`` without changing topology.  It reuses the audited V4.4.1 parser
semantics, but the artifact is a new Project7 Development contract and is not a rollback.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .step2_edge_physics_v441 import (
    CONDUIT_FEATURE_NAMES_V441,
    PhysicalLinkV441,
    normalize_conduit_static_features_v441,
    parse_frozen_inp_physical_links_v441,
)

EDGE_PHYSICS_CURRENT_CONTRACT = "PROJECT7_V128_GRAPH_EDGE_PHYSICS_DEVELOPMENT_V1"
EDGE_META_FEATURE_NAMES = (
    "has_conduit",
    "has_pump",
    "has_orifice",
    "has_weir",
    "has_outlet",
    "log1p_parallel_link_count",
    "mean_orientation_sign",
    "has_positive_conduit_length",
)


@dataclass(frozen=True)
class EdgePhysicsArtifactV128:
    edge_index: np.ndarray
    edge_static_features: np.ndarray
    edge_static_feature_names: tuple[str, ...]
    effective_length_m: np.ndarray
    physical_link_count: np.ndarray
    contract: str = EDGE_PHYSICS_CURRENT_CONTRACT

    def validate(self, graph: Any) -> None:
        edge = np.asarray(self.edge_index, dtype=np.int64)
        expected = np.asarray(graph.edge_index, dtype=np.int64)
        if edge.shape != expected.shape or not np.array_equal(edge, expected):
            raise ValueError("edge-physics artifact does not align exactly with current graph edge_index")
        if self.edge_static_features.ndim != 2 or self.edge_static_features.shape[0] != edge.shape[1]:
            raise ValueError("edge-physics features must be [E,F]")
        if self.edge_static_features.shape[1] != len(self.edge_static_feature_names):
            raise ValueError("edge-physics feature-name count mismatch")
        if self.effective_length_m.shape != (edge.shape[1],):
            raise ValueError("edge-physics effective length shape mismatch")
        if self.physical_link_count.shape != (edge.shape[1],):
            raise ValueError("edge-physics physical-link count shape mismatch")
        if not np.isfinite(self.edge_static_features).all():
            raise ValueError("edge-physics features contain non-finite values")
        if not np.isfinite(self.effective_length_m).all() or np.any(self.effective_length_m <= 0):
            raise ValueError("edge-physics effective lengths must be positive finite")


def _orientation(link: PhysicalLinkV441, src: str, dst: str) -> float:
    if link.from_node == src and link.to_node == dst:
        return 1.0
    if link.from_node == dst and link.to_node == src:
        return -1.0
    raise ValueError("physical link does not connect requested graph edge")


def build_edge_physics_artifact_v128(inp_path: str, graph: Any) -> EdgePhysicsArtifactV128:
    links = parse_frozen_inp_physical_links_v441(inp_path, graph.node_ids)
    by_pair: dict[tuple[str, str], list[PhysicalLinkV441]] = defaultdict(list)
    for link in links:
        by_pair[tuple(sorted((link.from_node, link.to_node)))].append(link)

    edge = np.asarray(graph.edge_index, dtype=np.int64)
    raw_conduit = np.zeros((edge.shape[1], len(CONDUIT_FEATURE_NAMES_V441)), dtype=np.float32)
    conduit_present = np.zeros(edge.shape[1], dtype=bool)
    meta = np.zeros((edge.shape[1], len(EDGE_META_FEATURE_NAMES)), dtype=np.float32)
    effective_length = np.ones(edge.shape[1], dtype=np.float32)
    physical_count = np.zeros(edge.shape[1], dtype=np.int32)

    for e, (src_i, dst_i) in enumerate(zip(edge[0], edge[1], strict=True)):
        src, dst = str(graph.node_ids[int(src_i)]), str(graph.node_ids[int(dst_i)])
        members = by_pair.get(tuple(sorted((src, dst))), [])
        if not members:
            raise ValueError(f"graph edge {src}->{dst} has no physical SWMM link")
        physical_count[e] = len(members)
        types = {member.link_type for member in members}
        meta[e, 0] = float("conduit" in types)
        meta[e, 1] = float("pump" in types)
        meta[e, 2] = float("orifice" in types)
        meta[e, 3] = float("weir" in types)
        meta[e, 4] = float("outlet" in types)
        meta[e, 5] = float(np.log1p(len(members)))
        meta[e, 6] = float(np.mean([_orientation(member, src, dst) for member in members]))
        conduits = [member for member in members if member.link_type == "conduit"]
        if conduits:
            conduit_present[e] = True
            raw_conduit[e] = np.mean(
                np.stack([np.asarray(member.static_features, dtype=np.float32) for member in conduits]),
                axis=0,
            )
            positive_lengths = [float(member.length_m) for member in conduits if member.length_m > 0]
            if positive_lengths:
                effective_length[e] = float(np.mean(positive_lengths))
                meta[e, 7] = 1.0

    if not np.any(conduit_present):
        raise ValueError("edge-physics artifact contains no conduit edges")
    normalized_conduit = np.zeros_like(raw_conduit)
    normalized_subset, _ = normalize_conduit_static_features_v441(raw_conduit[conduit_present])
    normalized_conduit[conduit_present] = normalized_subset
    features = np.concatenate((normalized_conduit, meta), axis=1).astype(np.float32)
    artifact = EdgePhysicsArtifactV128(
        edge_index=edge.copy(),
        edge_static_features=features,
        edge_static_feature_names=tuple(CONDUIT_FEATURE_NAMES_V441) + tuple(EDGE_META_FEATURE_NAMES),
        effective_length_m=effective_length,
        physical_link_count=physical_count,
    )
    artifact.validate(graph)
    return artifact


def save_edge_physics_artifact_v128(artifact: EdgePhysicsArtifactV128, path: str) -> str:
    np.savez_compressed(
        path,
        contract=np.asarray(artifact.contract),
        edge_index=artifact.edge_index,
        edge_static_features=artifact.edge_static_features,
        edge_static_feature_names=np.asarray(artifact.edge_static_feature_names),
        effective_length_m=artifact.effective_length_m,
        physical_link_count=artifact.physical_link_count,
    )
    return str(path)


def load_edge_physics_artifact_v128(path: str, graph: Any) -> EdgePhysicsArtifactV128:
    with np.load(path, allow_pickle=False) as raw:
        artifact = EdgePhysicsArtifactV128(
            edge_index=raw["edge_index"].astype(np.int64),
            edge_static_features=raw["edge_static_features"].astype(np.float32),
            edge_static_feature_names=tuple(raw["edge_static_feature_names"].astype(str).tolist()),
            effective_length_m=raw["effective_length_m"].astype(np.float32),
            physical_link_count=raw["physical_link_count"].astype(np.int32),
            contract=str(raw["contract"].item()),
        )
    if artifact.contract != EDGE_PHYSICS_CURRENT_CONTRACT:
        raise ValueError("not a current V128 edge-physics artifact")
    artifact.validate(graph)
    return artifact


__all__ = [
    "EDGE_META_FEATURE_NAMES",
    "EDGE_PHYSICS_CURRENT_CONTRACT",
    "EdgePhysicsArtifactV128",
    "build_edge_physics_artifact_v128",
    "load_edge_physics_artifact_v128",
    "save_edge_physics_artifact_v128",
]
