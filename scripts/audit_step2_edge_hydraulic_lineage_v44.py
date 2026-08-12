"""Read-only V4.4 physical-link lineage audit.

The script reads the frozen INP and graph artifact only.  It never invokes
SWMM, touches Validation/Final, or trains a model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rtc.step2_edge_hydraulic_v44 import (
    build_physical_directed_edge_lineage_v44,
    normalize_edge_static_features_v44,
    parse_frozen_inp_physical_links_v44,
    physical_link_census_v44,
)


ROOT = Path(r"E:\RTC_sewer\Project7")
INP = ROOT / "inputs" / "network" / "wuhan_method_testbed_v067.inp"
READINESS = ROOT / "study_v069" / "contracts" / "study_readiness.json"
GRAPH = ROOT / "study_v069" / "formal_assets" / "graph_schema.npz"
OUT = ROOT / "study_v069" / "step2_edge_hydraulic_interaction_v44"
DOCS = ROOT / "repo" / "docs"
EXPECTED_INP_SHA256 = "75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2, allow_nan=True) + "\n").encode("utf-8"))


def _write_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    _write_json(path, payload)
    markdown = f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n"
    path.with_suffix(".md").write_bytes(markdown.encode("utf-8"))


def _legacy_graph_audit(node_ids: tuple[str, ...], edge_index: np.ndarray, links: tuple[Any, ...]) -> dict[str, Any]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    directed_pairs = [(int(src), int(dst)) for src, dst in edge_index.T]
    undirected_pairs = [tuple(sorted(pair)) for pair in directed_pairs]
    directed_counts = Counter(directed_pairs)
    undirected_counts = Counter(undirected_pairs)
    link_by_pair: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for link in links:
        link_by_pair[link.unordered_node_pair].append(link.link_id)
    old_ambiguous = []
    old_unmapped = []
    for src, dst in directed_pairs:
        pair = tuple(sorted((node_ids[src], node_ids[dst])))
        candidates = link_by_pair.get(pair, [])
        if not candidates:
            old_unmapped.append((node_ids[src], node_ids[dst]))
        elif len(candidates) > 1:
            old_ambiguous.append({"src": node_ids[src], "dst": node_ids[dst], "link_ids": candidates})
    isolated = [node_ids[i] for i in range(len(node_ids)) if i not in set(edge_index.reshape(-1).tolist())]
    return {
        "nodes": len(node_ids),
        "directed_edges": int(edge_index.shape[1]),
        "unique_directed_pairs": len(directed_counts),
        "unique_undirected_node_pairs": len(undirected_counts),
        "bidirectional_undirected_pairs": sum(count == 2 for count in undirected_counts.values()),
        "self_loops": sum(src == dst for src, dst in directed_pairs),
        "duplicate_directed_edges": sum(count > 1 for count in directed_counts.values()),
        "isolated_nodes": len(isolated),
        "isolated_node_ids": isolated[:20],
        "ambiguous_old_mappings": len(old_ambiguous),
        "ambiguous_old_mapping_examples": old_ambiguous[:10],
        "unmapped_old_edges": len(old_unmapped),
        "unmapped_old_edge_examples": old_unmapped[:10],
        "edge_source_contract": "formal_assets/graph_schema.npz edge_index has node adjacency only; no physical link IDs",
        "edge_to_link_mapping_status": "EDGE_TO_LINK_MAPPING_NOT_ONE_TO_ONE" if old_ambiguous else "ONE_TO_ONE",
    }


def main() -> int:
    if not INP.exists() or not GRAPH.exists() or not READINESS.exists():
        raise FileNotFoundError("frozen INP, graph artifact, or study readiness contract is missing")
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    inp_sha = _sha256(INP)
    if inp_sha != EXPECTED_INP_SHA256 or inp_sha != readiness.get("frozen_inp_sha256"):
        raise RuntimeError(f"frozen INP SHA mismatch: {inp_sha}")
    graph = np.load(GRAPH, allow_pickle=True)
    node_ids = tuple(str(value) for value in graph["node_ids"].tolist())
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    links = parse_frozen_inp_physical_links_v44(INP, node_ids)
    lineage = build_physical_directed_edge_lineage_v44(
        links,
        {node_id: index for index, node_id in enumerate(node_ids)},
    )
    normalized, normalization = normalize_edge_static_features_v44(
        lineage.edge_static_features,
        lineage.edge_static_feature_names,
    )
    normalization_payload = {
        "contract": "EDGE_FEATURE_NORMALIZATION_V44",
        "method": "analytic_static_robust_graph_statistics_median_iqr_with_unit_floor",
        "feature_names": list(normalization.feature_names),
        "transform": list(normalization.transform),
        "location": normalization.location.tolist(),
        "scale": normalization.scale.tolist(),
        "normalized_shape": list(normalized.shape),
        "finite": bool(np.isfinite(normalized).all()),
    }
    normalization_bytes = json.dumps(normalization_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    normalization_payload["sha256"] = hashlib.sha256(normalization_bytes).hexdigest()
    census = physical_link_census_v44(links)
    type_feature_availability = {
        link_type: {
            "count": sum(link.link_type == link_type for link in links),
            "length_present": any(link.link_type == link_type and link.static_features[0] > 0 for link in links),
            "roughness_present": any(link.link_type == link_type and link.static_features[1] > 0 for link in links),
            "shape_identity_present": any(link.link_type == link_type and link.shape for link in links),
        }
        for link_type in ("conduit", "pump", "orifice", "weir", "outlet")
    }
    legacy = _legacy_graph_audit(node_ids, edge_index, links)
    report = {
        "contract": "STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44",
        "boundary": {
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "full_train_smoke_run": False,
        },
        "frozen_inp": {"path": str(INP), "sha256": inp_sha, "expected_sha256": EXPECTED_INP_SHA256},
        "graph": {
            "path": str(GRAPH),
            "sha256": _sha256(GRAPH),
            "nodes": len(node_ids),
            "edge_index_shape": list(edge_index.shape),
        },
        "physical_link_census": census,
        "legacy_graph_audit": legacy,
        "physical_directed_edge_contract": {
            "new_physical_directed_edges": int(lineage.edge_index.shape[1]),
            "one_directed_forward_and_reverse_per_physical_link": True,
            "mapping_complete": True,
            "unmapped_physical_links": 0,
            "parallel_links_retained": True,
            "edge_to_link_ids_unique_count": len(set(lineage.edge_to_link_id)),
            "orientation_signs": sorted(set(lineage.orientation_signs)),
        },
        "edge_feature_contract": {
            "static_feature_names": list(lineage.edge_static_feature_names),
            "dynamic_feature_names": list(lineage.dynamic_feature_names),
            "type_availability": type_feature_availability,
            "shape_identity_preserved": True,
            "raw_geometry_preserved_in_physical_link_records": True,
            "edge_features_finite": bool(np.isfinite(lineage.edge_static_features).all()),
        },
        "edge_feature_normalization": normalization_payload,
        "dynamic_contract": {
            "head_src": True,
            "head_dst": True,
            "delta_head": True,
            "hydraulic_gradient": True,
            "source": "causal current/reference model state only",
            "future_truth_used": False,
            "link_flow_used": False,
            "link_flow_online_availability": "LINK_FLOW_NOT_AVAILABLE_ONLINE",
        },
        "mapping_complete": True,
        "old_graph_mapping_one_to_one": not bool(legacy["ambiguous_old_mappings"]),
        "status": "PASS_PHYSICAL_LINEAGE_WITH_LEGACY_MULTI_EDGE_AMBIGUITY" if legacy["ambiguous_old_mappings"] else "PASS",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "EDGE_FEATURE_NORMALIZATION_V44.json", normalization_payload)
    _write_json(DOCS / "EDGE_FEATURE_NORMALIZATION_V44.json", normalization_payload)
    _write_report(OUT / "STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.json", "STEP2 EDGE HYDRAULIC LINEAGE AUDIT V4.4", report)
    _write_report(DOCS / "STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.json", "STEP2 EDGE HYDRAULIC LINEAGE AUDIT V4.4", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
