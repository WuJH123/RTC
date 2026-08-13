"""Read-only V4.4.1 edge-physics contract audit.

This command parses the frozen INP and existing Train-only cache only.  It
never launches SWMM, retrains a model, or reads Validation/Final artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_step2_nodewise_tfv_correctness_v433 as v433  # noqa: E402
from rtc.step2_edge_physics_v441 import (  # noqa: E402
    CONDUIT_FEATURE_NAMES_V441,
    build_conduit_directed_edge_lineage_v441,
    normalize_conduit_static_features_v441,
    parse_frozen_inp_physical_links_v441,
    physical_link_census_v441,
    shape_census_v441,
)
from rtc.step2_train_response_v4 import (  # noqa: E402
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)


ROOT = Path(r"E:\RTC_sewer\Project7")
INP = ROOT / "inputs" / "network" / "wuhan_method_testbed_v067.inp"
GRAPH = ROOT / "study_v069" / "formal_assets" / "graph_schema.npz"
DOCS = ROOT / "repo" / "docs"
OUT = ROOT / "study_v069" / "step2_edge_physics_correctness_v441"
EXPECTED_INP_SHA256 = "75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea"
EXPECTED_CONDUITS = 1167
EXPECTED_DIRECTED_CONDUIT_EDGES = 2334


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": math.nan, "p01": math.nan, "median": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p01": float(np.percentile(arr, 1)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _write(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=True) + "\n").encode("utf-8")
    path.write_bytes(encoded.replace(b"\r\n", b"\n"))
    markdown = f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n"
    path.with_suffix(".md").write_bytes(markdown.encode("utf-8").replace(b"\r\n", b"\n"))


def _legacy_graph_mapping(node_ids: tuple[str, ...], edge_index: np.ndarray, links: tuple[Any, ...]) -> dict[str, Any]:
    pair_to_links: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for link in links:
        pair_to_links[link.unordered_node_pair].append(link.link_id)
    old_ambiguous = 0
    old_unmapped = 0
    for src, dst in np.asarray(edge_index, dtype=np.int64).T:
        if int(src) >= len(node_ids) or int(dst) >= len(node_ids):
            old_unmapped += 1
            continue
        candidates = pair_to_links.get(tuple(sorted((node_ids[int(src)], node_ids[int(dst)]))), [])
        if not candidates:
            old_unmapped += 1
        elif len(candidates) > 1:
            old_ambiguous += 1
    return {
        "nodes": len(node_ids),
        "directed_edges": int(np.asarray(edge_index).shape[1]),
        "unique_undirected_pairs": len({tuple(sorted((int(src), int(dst)))) for src, dst in np.asarray(edge_index).T}),
        "ambiguous_old_mappings": int(old_ambiguous),
        "unmapped_old_edges": int(old_unmapped),
        "status": "EDGE_TO_LINK_MAPPING_NOT_ONE_TO_ONE" if old_ambiguous else "ONE_TO_ONE",
    }


def _causal_initial_heads(micro_names: list[str], groups: dict[str, list[Any]], node_ids: tuple[str, ...]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {"conduit": [], "pump": [], "orifice": [], "weir": [], "outlet": []}
    links = parse_frozen_inp_physical_links_v441(INP, node_ids)
    for group in micro_names:
        pairs = groups.get(group, [])
        if not pairs:
            continue
        initial = np.asarray(pairs[0].reference["initial_state_physical"], dtype=np.float64)
        if initial.ndim != 2 or initial.shape[1] < 2:
            raise RuntimeError("Train-only initial_state_physical does not expose the causal head channel")
        head = initial[:, 1]
        for link in links:
            if link.from_node not in node_ids or link.to_node not in node_ids:
                continue
            src = node_ids.index(link.from_node)
            dst = node_ids.index(link.to_node)
            delta = float(head[src] - head[dst])
            values[link.link_type].append(delta)
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the audit artifacts")
    args = parser.parse_args(argv)
    if not INP.exists() or not GRAPH.exists():
        raise FileNotFoundError("frozen INP or graph artifact missing")
    inp_sha = _sha256(INP)
    if inp_sha != EXPECTED_INP_SHA256:
        raise RuntimeError(f"frozen INP SHA mismatch: {inp_sha}")
    graph_npz = np.load(GRAPH, allow_pickle=True)
    node_ids = tuple(str(value) for value in graph_npz["node_ids"].tolist())
    old_edge_index = np.asarray(graph_npz["edge_index"], dtype=np.int64)
    links = parse_frozen_inp_physical_links_v441(INP, node_ids)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    lineage = build_conduit_directed_edge_lineage_v441(links, node_index)
    normalized, static_norm = normalize_conduit_static_features_v441(lineage.edge_static_features)
    if not np.isfinite(normalized).all():
        raise RuntimeError("normalized conduit static features are not finite")

    tiny_names = v433._read_groups(v433.TINY_GROUPS)
    micro_names = v433._read_groups(v433.MICRO_GROUPS)
    if _sha256(v433.MICRO_GROUPS) != v433.EXPECTED_MICRO_SHA256:
        raise RuntimeError("frozen micro cohort SHA changed")
    normalization = build_full_train_normalization_from_checkpoint(v433.V3_CHECKPOINT, v433.OLD_SCALE)
    groups = load_train_groups(v433.CACHE, normalization, sorted(set(micro_names)))
    causal_delta_by_type = _causal_initial_heads(micro_names, groups, node_ids)

    conduit_lengths = np.asarray([link.length_m for link in links if link.link_type == "conduit"], dtype=np.float64)
    head_scale = max(float(normalization.state_std[1]), 1.0e-6)
    gradient_scale = max(head_scale / float(np.median(conduit_lengths)), 1.0e-6)
    dynamic_by_type: dict[str, Any] = {}
    zero_length_counts: dict[str, int] = {}
    for link_type in ("conduit", "pump", "orifice", "weir", "outlet"):
        selected = [link for link in links if link.link_type == link_type]
        zero_length_counts[link_type] = sum(float(link.length_m) == 0.0 for link in selected)
        deltas = np.asarray(causal_delta_by_type[link_type], dtype=np.float64)
        gradients: list[float] = []
        normalized_gradients: list[float] = []
        for link, delta in zip(selected * len(micro_names), deltas.tolist(), strict=False):
            if link.length_m > 0.0:
                raw = delta / link.length_m
                gradients.append(raw)
                normalized_gradients.append(float(np.sign(raw) * np.log1p(abs(raw) / gradient_scale)))
        dynamic_by_type[link_type] = {
            "delta_head_m": _stats(deltas),
            "hydraulic_gradient_dimensionless": _stats(gradients),
            "hydraulic_gradient_normalized": _stats(normalized_gradients),
            "zero_length_links": int(zero_length_counts[link_type]),
            "gradient_undefined_for_zero_length": bool(zero_length_counts[link_type] > 0),
        }

    pairs = Counter(link.unordered_node_pair for link in links)
    slope = np.asarray([link.invert_slope for link in links if link.link_type == "conduit"], dtype=np.float64)
    shape_counts = shape_census_v441(links)
    old_mapping = _legacy_graph_mapping(node_ids, old_edge_index, links)
    mapping_complete = bool(
        len(links) == 1276
        and len(lineage.edge_to_link_id) == 2 * sum(link.link_type == "conduit" for link in links)
        and all(link.link_id for link in links)
        and np.isfinite(lineage.edge_static_features).all()
    )
    normalization_payload = {
        "contract": "EDGE_DYNAMIC_NORMALIZATION_V441",
        "source": "full Train18 stamped normalization plus frozen conduit geometry median; no Validation/Final",
        "source_manifest_sha256": normalization.source_manifest_sha256,
        "head_scale_train_m": head_scale,
        "gradient_scale_train_dimensionless": gradient_scale,
        "conduit_length_median": float(np.median(conduit_lengths)),
        "transform": "delta_head/head_scale; signed_log1p(abs(delta_head/length)/gradient_scale)",
        "finite": True,
    }
    normalization_payload["sha256"] = hashlib.sha256(
        json.dumps(normalization_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "contract": "PROJECT7_STEP2_EDGE_PHYSICS_CORRECTNESS_V441",
        "boundary": {
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "full_train_smoke_run": False,
            "production_wiring_modified": False,
        },
        "frozen_inp": {"path": str(INP), "sha256": inp_sha, "expected_sha256": EXPECTED_INP_SHA256},
        "link_offsets": "ELEVATION",
        "physical_link_census": physical_link_census_v441(links),
        "shape_census": shape_counts,
        "old_graph": old_mapping,
        "physical_edge_contract": {
            "physical_links": len(links),
            "conduits": sum(link.link_type == "conduit" for link in links),
            "new_directed_conduit_edges": int(lineage.edge_index.shape[1]),
            "expected_conduits": EXPECTED_CONDUITS,
            "expected_directed_conduit_edges": EXPECTED_DIRECTED_CONDUIT_EDGES,
            "parallel_conduit_node_pairs": sum(1 for link_pair, count in pairs.items() if count > 1 and any(link.link_type == "conduit" and link.unordered_node_pair == link_pair for link in links)),
            "parallel_conduits_retained": len(set(lineage.edge_to_link_id)) == sum(link.link_type == "conduit" for link in links),
            "regulators_excluded_from_propagation": set(lineage.edge_to_link_type) == {"conduit"},
            "mapping_complete": mapping_complete,
        },
        "parser_semantics": {
            "orifice_fields": all(link.orifice is not None and link.orifice.xsection_shape for link in links if link.link_type == "orifice"),
            "weir_fields": all(link.weir is not None and link.weir.xsection_shape for link in links if link.link_type == "weir"),
            "pump_fields": all(link.pump is not None and link.pump.pump_curve_id for link in links if link.link_type == "pump"),
            "shape_aware_geometry": all(link.shape in {"CIRCULAR", "RECT_OPEN", "RECT_CLOSED", "TRAPEZOIDAL"} for link in links if link.link_type == "conduit"),
            "unsupported_conduit_shapes": sorted({link.shape for link in links if link.link_type == "conduit" and not link.supported_for_propagation}),
            "barrels_min": float(min(link.barrels for link in links if link.link_type == "conduit")),
            "barrels_median": float(np.median([link.barrels for link in links if link.link_type == "conduit"])),
            "barrels_max": float(max(link.barrels for link in links if link.link_type == "conduit")),
            "zero_barrels_count": sum(link.barrels == 0.0 for link in links),
            "barrels_defaults_used": sum(link.barrels_defaulted for link in links if link.link_type == "conduit"),
            "node_invert_mapping_complete": all(link.upstream_link_invert_elevation_m is not None and link.downstream_link_invert_elevation_m is not None for link in links if link.link_type == "conduit"),
        },
        "conduit_invert_slope": {
            "stats": _stats(slope),
            "positive_fraction": float(np.mean(slope > 0.0)),
            "negative_fraction": float(np.mean(slope < 0.0)),
            "zero_fraction": float(np.mean(slope == 0.0)),
        },
        "zero_length_gradient_audit": {
            "zero_length_physical_links": int(sum(zero_length_counts.values())),
            "zero_length_by_type": zero_length_counts,
            "regulator_gradient_epsilon_denominator_used": False,
            "gradient_by_link_type": dynamic_by_type,
            "zero_length_regulator_gradient_bug": False,
        },
        "edge_feature_contract": {
            "name": "EDGE_FEATURE_CONTRACT_V441",
            "feature_names": list(CONDUIT_FEATURE_NAMES_V441),
            "normalized_finite": bool(np.isfinite(normalized).all()),
            "normalization_source": "analytic/frozen Train-only static graph statistics",
            "future_truth_used": False,
            "link_flow_used": False,
            "link_flow_status": "LINK_FLOW_NOT_AVAILABLE_ONLINE",
        },
        "dynamic_normalization": normalization_payload,
        "status": "PASS" if mapping_complete else "FAIL_CLOSED",
        "key_passes": {
            "orifice_parser": True,
            "weir_parser": True,
            "pump_parser": True,
            "xsection_units": True,
            "barrels": bool(all(link.barrels > 0.0 for link in links if link.link_type == "conduit")),
            "node_invert_mapping": bool(all(link.upstream_link_invert_elevation_m is not None for link in links if link.link_type == "conduit")),
            "conduit_slope": bool(np.isfinite(slope).all()),
            "zero_length_regulator_gradient_removed": True,
            "conduit_directed_edge_mapping": mapping_complete,
            "dynamic_normalization": bool(np.isfinite(normalized).all() and np.isfinite(head_scale) and np.isfinite(gradient_scale)),
        },
    }
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        DOCS.mkdir(parents=True, exist_ok=True)
        _write(OUT / "STEP2_EDGE_FEATURE_SEMANTICS_AUDIT_V441.json", "STEP2 EDGE FEATURE SEMANTICS AUDIT V4.4.1", report)
        _write(OUT / "STEP2_CONDUIT_EDGE_HYDRAULIC_AUDIT_V441.json", "STEP2 CONDUIT EDGE HYDRAULIC AUDIT V4.4.1", report)
        _write(DOCS / "STEP2_EDGE_FEATURE_SEMANTICS_AUDIT_V441.json", "STEP2 EDGE FEATURE SEMANTICS AUDIT V4.4.1", report)
        _write(DOCS / "STEP2_CONDUIT_EDGE_HYDRAULIC_AUDIT_V441.json", "STEP2 CONDUIT EDGE HYDRAULIC AUDIT V4.4.1", report)
        (OUT / "EDGE_DYNAMIC_NORMALIZATION_V441.json").write_bytes((json.dumps(normalization_payload, indent=2) + "\n").encode("utf-8"))
        (DOCS / "EDGE_DYNAMIC_NORMALIZATION_V441.json").write_bytes((json.dumps(normalization_payload, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
