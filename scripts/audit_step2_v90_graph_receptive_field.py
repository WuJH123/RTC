"""Write the TrainFit-D2 V9 graph receptive-field evidence census.

This is a read-only cache/graph audit.  It performs no SWMM, never reads
Validation/Final/Formal assets, and measures authoritative target mass rather
than model predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rtc.graph import GraphSchema
from rtc.step2_graph_audit_v90 import (
    endpoint_distances_v90,
    receptive_field_mass_v90,
    undirected_adjacency_v90,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v60_contract import MultiResolutionHorizonV60


CONTRACT = "PROJECT7_STEP2_V90_GRAPH_RECEPTIVE_FIELD_AUDIT_V1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _graph(path: Path) -> GraphSchema:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    graph = _graph(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    selected = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, _holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=args.holdout_fraction
    )
    d2 = [name for name in fit if name.startswith("D2::")]
    if not d2:
        raise ValueError("no TrainFit D2 groups")
    adjacency = undirected_adjacency_v90(graph.edge_index, node_count=len(graph.node_ids))
    distances = endpoint_distances_v90(adjacency, graph.actuator_upstream, graph.actuator_downstream)
    indices = np.asarray(MultiResolutionHorizonV60().indices(), dtype=np.int64)
    changed: list[int] = []
    state_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []
    for name in d2:
        entry = cache.entry(name)
        arrays, reference = entry.arrays, entry.reference_index
        candidates = np.asarray([value for value in entry.indices if value != reference], dtype=np.int64)
        actions = np.asarray(arrays["settings"][candidates], dtype=np.float64) - np.asarray(
            arrays["settings"][reference], dtype=np.float64
        )[None]
        active = np.any(np.abs(actions) > 1e-8, axis=1)
        count = active.sum(axis=1)
        if not np.all(count == 1):
            raise ValueError(f"{name}: D2 candidate is not exactly single-actuator")
        changed.extend(np.argmax(active, axis=1).astype(int).tolist())
        state_parts.append(
            np.asarray(arrays["target_states"][candidates], dtype=np.float64)[:, indices]
            - np.asarray(arrays["target_states"][reference], dtype=np.float64)[None, indices]
        )
        flow_parts.append(
            np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64)[:, indices]
            - np.asarray(arrays["target_actuator_flows"][reference], dtype=np.float64)[None, indices]
        )
    mass = receptive_field_mass_v90(
        distances_by_changed_actuator=distances,
        changed_actuators=np.asarray(changed, dtype=np.int64),
        delta_states=np.concatenate(state_parts, axis=0),
        delta_flows=np.concatenate(flow_parts, axis=0),
        actuator_upstream=graph.actuator_upstream,
        actuator_downstream=graph.actuator_downstream,
    )
    payload = {
        "contract": CONTRACT,
        "scope": {
            "development_train_only": True,
            "swmm_run": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
        "git_head": _git_head(),
        "graph_sha256": _sha256(args.graph),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        "trainfit_d2_group_count": len(d2),
        "candidate_count": len(changed),
        "retained_horizon_steps": len(indices),
        "current_v9_graph": {
            "nodes": len(graph.node_ids),
            "directed_edges": int(graph.edge_index.shape[1]),
            "endpoint_seeded": True,
            "former_fixed_graph_blocks": 4,
            "audited_hops": [4, 8],
        },
        "authoritative_absolute_effect_mass": mass,
        "interpretation": (
            "Outside-radius mass is structurally unreachable by an endpoint-seeded "
            "zero-preserving message path at that radius. This is a representation "
            "census, not a model-performance metric."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["authoritative_absolute_effect_mass"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
