from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .inp import ActuatorCatalog, discover_actuators, discover_nodes
from .units import flow_rate_to_m3s, length_to_m


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
    for section, tokens in _iter_rows(path):
        if section == "OPTIONS" and len(tokens) >= 2 and tokens[0].upper() == "FLOW_UNITS":
            units = tokens[1].upper()
            if units in _SI_FLOW_UNITS:
                return "SI"
            if units in _US_FLOW_UNITS:
                return "US"
            raise ValueError(f"unsupported SWMM FLOW_UNITS: {units}")
    raise ValueError("[OPTIONS] FLOW_UNITS not found in INP")


def infer_flow_units(path: str | Path) -> str:
    for section, tokens in _iter_rows(path):
        if section == "OPTIONS" and len(tokens) >= 2 and tokens[0].upper() == "FLOW_UNITS":
            return tokens[1].upper()
    raise ValueError("[OPTIONS] FLOW_UNITS not found in INP")


def _curve_catalog(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for section, tokens in _iter_rows(path):
        if section != "CURVES" or len(tokens) < 3:
            continue
        curve_id = tokens[0]
        # First row may include curve type (e.g. PUMP2); continuation rows do not.
        offset = 2 if len(tokens) >= 4 and not _is_number(tokens[1]) else 1
        if len(tokens) <= offset + 1:
            continue
        try:
            x, y = float(tokens[offset]), float(tokens[offset + 1])
        except ValueError:
            continue
        curves.setdefault(curve_id, []).append((x, y))
    return curves


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _xsections(path: str | Path) -> dict[str, tuple[str, tuple[float, float, float, float]]]:
    result: dict[str, tuple[str, tuple[float, float, float, float]]] = {}
    for section, tokens in _iter_rows(path):
        if section != "XSECTIONS" or len(tokens) < 3:
            continue
        values = tuple(_float(tokens[i]) if i < len(tokens) else 0.0 for i in range(2, 6))
        result[tokens[0]] = (tokens[1].upper(), values)
    return result


def build_graph_schema(path: str | Path, *, bidirectional: bool = True) -> GraphSchema:
    """Compile topology and SI hydraulic features from the frozen SWMM INP.

    The feature schema intentionally includes device capacity/geometry so dozens of pumps,
    orifices and weirs are not indistinguishable merely because they share a device type.
    A learned actuator-ID embedding is added separately by the Step2 model and is locked to
    this stable actuator order.
    """

    node_ids = discover_nodes(path)
    node_index = {node: i for i, node in enumerate(node_ids)}
    catalog: ActuatorCatalog = discover_actuators(path)
    system_units = infer_system_units(path)
    flow_units = infer_flow_units(path)
    curves = _curve_catalog(path)
    xsections = _xsections(path)

    # [invert_elevation_m, max_depth_m, one-hot node type x4]
    static = np.zeros((len(node_ids), 6), dtype=np.float32)
    seen_nodes: set[str] = set()
    physical_edges: list[tuple[int, int]] = []
    raw_by_link: dict[str, tuple[str, list[str]]] = {}
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
            raw_by_link[tokens[0]] = (section, tokens)
            if upstream in node_index and downstream in node_index:
                physical_edges.append((node_index[upstream], node_index[downstream]))

    if seen_nodes != set(node_ids):
        missing = sorted(set(node_ids) - seen_nodes)
        raise ValueError(f"failed to compile node static features for: {missing[:20]}")
    if not physical_edges:
        raise ValueError("no hydraulic graph edges discovered")
    edges = physical_edges + ([(v, u) for u, v in physical_edges] if bidirectional else [])
    edge_index = np.asarray(list(dict.fromkeys(edges)), dtype=np.int64).T

    up = np.array([node_index[a.upstream_node] for a in catalog.actuators], dtype=np.int64)
    down = np.array([node_index[a.downstream_node] for a in catalog.actuators], dtype=np.int64)

    feature_names = (
        "is_pump", "is_orifice", "is_weir", "is_outlet",
        "min_setting", "max_setting",
        "pump_curve_max_flow_m3s", "pump_curve_max_x_m", "pump_curve_point_count",
        "offset_or_crest_m", "discharge_coefficient", "has_flap_gate",
        "xsection_geom1_m", "xsection_geom2_m", "xsection_geom3_m", "xsection_geom4_m",
        "xsection_is_circular", "xsection_is_rect_closed", "xsection_is_rect_open",
    )
    physics = np.zeros((len(catalog.actuators), len(feature_names)), dtype=np.float32)
    for i, actuator in enumerate(catalog.actuators):
        physics[i, _ACTUATOR_KIND[actuator.kind]] = 1.0
        physics[i, 4] = actuator.min_setting
        physics[i, 5] = actuator.max_setting
        section, tokens = raw_by_link[actuator.actuator_id]

        if section == "PUMPS" and len(tokens) >= 4:
            points = curves.get(tokens[3], [])
            if points:
                physics[i, 6] = float(flow_rate_to_m3s(max(y for _, y in points), flow_units))
                physics[i, 7] = float(length_to_m(max(x for x, _ in points), system_units))
                physics[i, 8] = float(len(points))
        elif section == "ORIFICES":
            if len(tokens) > 4:
                physics[i, 9] = float(length_to_m(_float(tokens[4]), system_units))
            if len(tokens) > 5:
                physics[i, 10] = _float(tokens[5])
            if len(tokens) > 6:
                physics[i, 11] = float(tokens[6].upper() == "YES")
        elif section == "WEIRS":
            if len(tokens) > 4:
                physics[i, 9] = float(length_to_m(_float(tokens[4]), system_units))
            if len(tokens) > 5:
                physics[i, 10] = _float(tokens[5])
            if len(tokens) > 6:
                physics[i, 11] = float(tokens[6].upper() == "YES")

        shape, geom = xsections.get(actuator.actuator_id, ("", (0.0, 0.0, 0.0, 0.0)))
        for j, raw in enumerate(geom):
            physics[i, 12 + j] = float(length_to_m(raw, system_units))
        physics[i, 16] = float(shape == "CIRCULAR")
        physics[i, 17] = float(shape == "RECT_CLOSED")
        physics[i, 18] = float(shape == "RECT_OPEN")

    return GraphSchema(
        node_ids=tuple(node_ids),
        edge_index=edge_index,
        static_node_features=static,
        static_node_feature_names=(
            "invert_elevation_m", "max_depth_m", "is_junction", "is_outfall", "is_storage", "is_divider",
        ),
        actuator_ids=tuple(catalog.ids),
        actuator_upstream=up,
        actuator_downstream=down,
        actuator_physics=physics,
        actuator_physics_feature_names=feature_names,
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
