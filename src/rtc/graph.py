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


def _trapz(points: list[tuple[float, float]], max_depth: float) -> tuple[float, float]:
    if not points or max_depth <= 0:
        return 0.0, 0.0
    pts = sorted((max(0.0, float(x)), max(0.0, float(y))) for x, y in points)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < max_depth:
        pts.append((max_depth, pts[-1][1]))
    samples: list[tuple[float, float]] = []
    for j, (x, area) in enumerate(pts):
        if x <= max_depth:
            samples.append((x, area))
            continue
        x0, a0 = pts[j - 1]
        frac = 0.0 if x == x0 else (max_depth - x0) / (x - x0)
        samples.append((max_depth, a0 + frac * (area - a0)))
        break
    if samples[-1][0] < max_depth:
        samples.append((max_depth, samples[-1][1]))
    volume = sum(
        0.5 * (a0 + a1) * max(0.0, x1 - x0)
        for (x0, a0), (x1, a1) in zip(samples[:-1], samples[1:], strict=False)
    )
    return volume, samples[-1][1]


def _node_extra_features(
    path: str | Path,
    *,
    node_ids: tuple[str, ...],
    node_index: dict[str, int],
    system_units: str,
    curves: dict[str, list[tuple[float, float]]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Expose INP hydraulic/hydrologic information that v0.6.6 discarded.

    The topology contract stays stable. Step1/Step2 automatically consume these appended
    node-static values because both models use the complete static feature matrix.
    """

    names = (
        "init_depth_m",
        "surcharge_depth_m",
        "ponded_area_m2",
        "storage_capacity_m3",
        "storage_area_full_m2",
        "conduit_in_count",
        "conduit_out_count",
        "conduit_in_length_sum_m",
        "conduit_out_length_sum_m",
        "conduit_in_roughness_mean",
        "conduit_out_roughness_mean",
        "conduit_in_geom1_mean_m",
        "conduit_out_geom1_mean_m",
        "subcatchment_count",
        "subcatchment_area_m2",
        "subcatchment_impervious_area_m2",
        "subcatchment_width_area_weighted_m",
        "subcatchment_slope_area_weighted_pct",
        "infiltration_max_rate_area_weighted_mmhr",
        "infiltration_min_rate_area_weighted_mmhr",
    )
    out = np.zeros((len(node_ids), len(names)), dtype=np.float64)
    cin_rough = [[] for _ in node_ids]
    cout_rough = [[] for _ in node_ids]
    cin_geom = [[] for _ in node_ids]
    cout_geom = [[] for _ in node_ids]
    sub_area_sum = np.zeros(len(node_ids), dtype=float)
    sub_width_weighted = np.zeros(len(node_ids), dtype=float)
    sub_slope_weighted = np.zeros(len(node_ids), dtype=float)
    inf_max_weighted = np.zeros(len(node_ids), dtype=float)
    inf_min_weighted = np.zeros(len(node_ids), dtype=float)
    xsections = _xsections(path)
    infiltration: dict[str, tuple[float, float]] = {}
    subcatchments: list[tuple[str, str, float, float, float, float]] = []

    for section, tokens in _iter_rows(path):
        if section == "INFILTRATION" and len(tokens) >= 3:
            infiltration[tokens[0]] = (_float(tokens[1]), _float(tokens[2]))
        elif section == "SUBCATCHMENTS" and len(tokens) >= 7:
            subcatchments.append(
                (
                    tokens[0],
                    tokens[2],
                    max(0.0, _float(tokens[3])),
                    min(100.0, max(0.0, _float(tokens[4]))),
                    max(0.0, _float(tokens[5])),
                    max(0.0, _float(tokens[6])),
                )
            )
        elif section == "JUNCTIONS" and tokens[0] in node_index:
            idx = node_index[tokens[0]]
            if len(tokens) > 3:
                out[idx, 0] = float(length_to_m(_float(tokens[3]), system_units))
            if len(tokens) > 4:
                out[idx, 1] = float(length_to_m(_float(tokens[4]), system_units))
            if len(tokens) > 5:
                area = max(0.0, _float(tokens[5]))
                out[idx, 2] = area if system_units == "SI" else area * 0.09290304
        elif section == "STORAGE" and tokens[0] in node_index:
            idx = node_index[tokens[0]]
            max_depth_native = _float(tokens[2]) if len(tokens) > 2 else 0.0
            if len(tokens) > 3:
                out[idx, 0] = float(length_to_m(_float(tokens[3]), system_units))
            shape = tokens[4].upper() if len(tokens) > 4 else ""
            volume_native = area_full_native = 0.0
            if shape == "TABULAR" and len(tokens) > 5:
                volume_native, area_full_native = _trapz(
                    curves.get(tokens[5], []), max_depth_native
                )
            elif shape == "FUNCTIONAL" and len(tokens) > 7:
                a1, a2, a0 = _float(tokens[5]), _float(tokens[6]), _float(tokens[7])
                depth = max(0.0, max_depth_native)
                area_full_native = max(
                    0.0, a1 * (depth ** a2 if depth > 0 or a2 != 0 else 1.0) + a0
                )
                if a2 > -1 and depth > 0:
                    volume_native = max(
                        0.0, a1 * depth ** (a2 + 1) / (a2 + 1) + a0 * depth
                    )
            if system_units == "SI":
                out[idx, 3] = volume_native
                out[idx, 4] = area_full_native
            else:
                out[idx, 3] = volume_native * 0.028316846592
                out[idx, 4] = area_full_native * 0.09290304
        elif section == "CONDUITS" and len(tokens) >= 6:
            conduit_id, upstream, downstream = tokens[0], tokens[1], tokens[2]
            if upstream not in node_index or downstream not in node_index:
                continue
            u, v = node_index[upstream], node_index[downstream]
            length_m = float(length_to_m(max(0.0, _float(tokens[3])), system_units))
            roughness = max(0.0, _float(tokens[4]))
            shape, geom = xsections.get(conduit_id, ("", (0.0, 0.0, 0.0, 0.0)))
            geom1_m = 0.0
            if shape not in {"IRREGULAR", "CUSTOM"}:
                geom1_m = float(length_to_m(max(0.0, geom[0]), system_units))
            out[v, 5] += 1
            out[u, 6] += 1
            out[v, 7] += length_m
            out[u, 8] += length_m
            cin_rough[v].append(roughness)
            cout_rough[u].append(roughness)
            cin_geom[v].append(geom1_m)
            cout_geom[u].append(geom1_m)

    for i in range(len(node_ids)):
        out[i, 9] = float(np.mean(cin_rough[i])) if cin_rough[i] else 0.0
        out[i, 10] = float(np.mean(cout_rough[i])) if cout_rough[i] else 0.0
        out[i, 11] = float(np.mean(cin_geom[i])) if cin_geom[i] else 0.0
        out[i, 12] = float(np.mean(cout_geom[i])) if cout_geom[i] else 0.0

    for sid, outlet, area_native, imperv, width_native, slope_pct in subcatchments:
        if outlet not in node_index:
            continue
        idx = node_index[outlet]
        area_m2 = area_native * (10000.0 if system_units == "SI" else 4046.8564224)
        width_m = float(length_to_m(width_native, system_units))
        out[idx, 13] += 1
        out[idx, 14] += area_m2
        out[idx, 15] += area_m2 * imperv / 100.0
        sub_area_sum[idx] += area_m2
        sub_width_weighted[idx] += width_m * area_m2
        sub_slope_weighted[idx] += slope_pct * area_m2
        max_rate, min_rate = infiltration.get(sid, (0.0, 0.0))
        rate_factor = 1.0 if system_units == "SI" else 25.4
        inf_max_weighted[idx] += max_rate * rate_factor * area_m2
        inf_min_weighted[idx] += min_rate * rate_factor * area_m2

    nonzero = sub_area_sum > 0
    out[nonzero, 16] = sub_width_weighted[nonzero] / sub_area_sum[nonzero]
    out[nonzero, 17] = sub_slope_weighted[nonzero] / sub_area_sum[nonzero]
    out[nonzero, 18] = inf_max_weighted[nonzero] / sub_area_sum[nonzero]
    out[nonzero, 19] = inf_min_weighted[nonzero] / sub_area_sum[nonzero]
    return out.astype(np.float32), names


def build_graph_schema(path: str | Path, *, bidirectional: bool = True) -> GraphSchema:
    """Compile topology and SI hydraulic features from the frozen SWMM INP."""

    node_ids = discover_nodes(path)
    node_index = {node: i for i, node in enumerate(node_ids)}
    catalog: ActuatorCatalog = discover_actuators(path)
    system_units = infer_system_units(path)
    flow_units = infer_flow_units(path)
    curves = _curve_catalog(path)
    xsections = _xsections(path)

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

    extra, extra_names = _node_extra_features(
        path,
        node_ids=tuple(node_ids),
        node_index=node_index,
        system_units=system_units,
        curves=curves,
    )
    static = np.concatenate([static, extra], axis=1).astype(np.float32)

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
            "invert_elevation_m",
            "max_depth_m",
            "is_junction",
            "is_outfall",
            "is_storage",
            "is_divider",
            *extra_names,
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
