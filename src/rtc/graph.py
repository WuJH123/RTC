from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .inp import ActuatorCatalog, discover_actuators, discover_nodes
from .units import length_to_m


_LINK_SECTIONS = {"CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS"}
_NODE_KIND = {"JUNCTIONS": 0, "OUTFALLS": 1, "STORAGE": 2, "DIVIDERS": 3}
_ACTUATOR_KIND = {"pump": 0, "orifice": 1, "weir": 2, "outlet": 3}
_SI_FLOW_UNITS = {"CMS", "LPS", "MLD"}
_US_FLOW_UNITS = {"CFS", "GPM", "MGD"}


@dataclass(frozen=True)
class GraphSchema:
    node_ids: tuple[str, ...]
    edge_index: np.ndarray
    static_node_features: np.ndarray
    static_node_feature_names: tuple[str, ...]
    actuator_ids: tuple[str, ...]
    actuator_upstream: np.ndarray
    actuator_downstream: np.ndarray
    actuator_physics: np.ndarray
    actuator_physics_feature_names: tuple[str, ...]
    system_units: str


def _iter_rows(path: str | Path) -> Iterable[tuple[str, list[str]]]:
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        tokens = line.split()
        if tokens:
            yield section, tokens


def _float(token: str, default: float = 0.0) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        return float(default)


def infer_system_units(path: str | Path) -> str:
    """Infer SWMM SI/US system units from [OPTIONS] FLOW_UNITS."""

    for section, tokens in _iter_rows(path):
        if section == "OPTIONS" and len(tokens) >= 2 and tokens[0].upper() == "FLOW_UNITS":
            units = tokens[1].upper()
            if units in _SI_FLOW_UNITS:
                return "SI"
            if units in _US_FLOW_UNITS:
                return "US"
            raise ValueError(f"unsupported SWMM FLOW_UNITS: {units}")
    raise ValueError("[OPTIONS] FLOW_UNITS not found in INP")


def build_graph_schema(path: str | Path, *, bidirectional: bool = True) -> GraphSchema:
    """Compile stable graph/model features directly from the frozen SWMM INP.

    Dynamic-wave systems can exhibit backwater and flow reversal, so message passing is
    bidirectional by default. Physical actuator orientation is retained separately. All
    dimensional static node features are converted to SI before entering the models.
    """

    node_ids = discover_nodes(path)
    node_index = {node: i for i, node in enumerate(node_ids)}
    catalog: ActuatorCatalog = discover_actuators(path)
    system_units = infer_system_units(path)

    # [invert_elevation_m, max_depth_m, one-hot node type x4]
    static = np.zeros((len(node_ids), 6), dtype=np.float32)
    seen_nodes: set[str] = set()
    physical_edges: list[tuple[int, int]] = []
    for section, tokens in _iter_rows(path):
        if section in _NODE_KIND and tokens[0] in node_index:
            nid = tokens[0]
            idx = node_index[nid]
            seen_nodes.add(nid)
            raw_elevation = _float(tokens[1]) if len(tokens) > 1 else 0.0
            static[idx, 0] = float(length_to_m(raw_elevation, system_units))
            if section in {"JUNCTIONS", "STORAGE"} and len(tokens) > 2:
                static[idx, 1] = float(length_to_m(_float(tokens[2]), system_units))
            static[idx, 2 + _NODE_KIND[section]] = 1.0
        if section in _LINK_SECTIONS and len(tokens) >= 3:
            upstream, downstream = tokens[1], tokens[2]
            if upstream in node_index and downstream in node_index:
                physical_edges.append((node_index[upstream], node_index[downstream]))

    if seen_nodes != set(node_ids):
        missing = sorted(set(node_ids) - seen_nodes)
        raise ValueError(f"failed to compile node static features for: {missing[:20]}")
    if not physical_edges:
        raise ValueError("no hydraulic graph edges discovered")
    edges = physical_edges + ([(v, u) for u, v in physical_edges] if bidirectional else [])
    edges = list(dict.fromkeys(edges))
    edge_index = np.asarray(edges, dtype=np.int64).T

    up = np.array([node_index[a.upstream_node] for a in catalog.actuators], dtype=np.int64)
    down = np.array([node_index[a.downstream_node] for a in catalog.actuators], dtype=np.int64)
    physics = np.zeros((len(catalog.actuators), 6), dtype=np.float32)
    for i, actuator in enumerate(catalog.actuators):
        physics[i, _ACTUATOR_KIND[actuator.kind]] = 1.0
        physics[i, 4] = actuator.min_setting
        physics[i, 5] = actuator.max_setting

    return GraphSchema(
        node_ids=tuple(node_ids),
        edge_index=edge_index,
        static_node_features=static,
        static_node_feature_names=(
            "invert_elevation_m",
            "max_depth_m",
            "is_junction",
            "is_outfall",
            "is_storage",
            "is_divider",
        ),
        actuator_ids=tuple(catalog.ids),
        actuator_upstream=up,
        actuator_downstream=down,
        actuator_physics=physics,
        actuator_physics_feature_names=(
            "is_pump",
            "is_orifice",
            "is_weir",
            "is_outlet",
            "min_setting",
            "max_setting",
        ),
        system_units=system_units,
    )


def save_graph_schema(schema: GraphSchema, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        node_ids=np.asarray(schema.node_ids),
        edge_index=schema.edge_index,
        static_node_features=schema.static_node_features,
        static_node_feature_names=np.asarray(schema.static_node_feature_names),
        actuator_ids=np.asarray(schema.actuator_ids),
        actuator_upstream=schema.actuator_upstream,
        actuator_downstream=schema.actuator_downstream,
        actuator_physics=schema.actuator_physics,
        actuator_physics_feature_names=np.asarray(schema.actuator_physics_feature_names),
        system_units=np.asarray(schema.system_units),
    )
    return out
