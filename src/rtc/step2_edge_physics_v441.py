"""Correct SWMM edge-physics contracts for the bounded Step2 V4.4.1 study.

This module is deliberately parser/lineage focused.  It never runs SWMM and
does not consume future hydraulic truth.  Regulator records are retained for
lineage/audit, while the first corrected propagation contract is conduit-only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .graph import infer_system_units
from .units import length_to_m


EDGE_FEATURE_CONTRACT_V441 = "EDGE_FEATURE_CONTRACT_V441"
LINK_OFFSETS_VALUES_V441 = {"DEPTH", "ELEVATION"}
SUPPORTED_CONDUIT_SHAPES_V441 = (
    "CIRCULAR",
    "RECT_OPEN",
    "RECT_CLOSED",
    "TRAPEZOIDAL",
    "POWER",
)
SHAPE_CENSUS_NAMES_V441 = SUPPORTED_CONDUIT_SHAPES_V441 + ("OTHER",)

CONDUIT_FEATURE_NAMES_V441 = (
    "log_length_m",
    "roughness_n",
    "invert_slope",
    "inlet_relative_offset_m",
    "outlet_relative_offset_m",
    "full_depth_m",
    "width_or_base_width_m",
    "left_side_slope",
    "right_side_slope",
    "power_exponent",
    "barrels",
    "entrance_loss",
    "exit_loss",
    "average_loss",
    "flap_gate",
    "valid_length",
    "valid_roughness",
    "valid_invert_slope",
    "valid_inlet_relative_offset",
    "valid_outlet_relative_offset",
    "valid_full_depth",
    "valid_width_or_base_width",
    "valid_left_side_slope",
    "valid_right_side_slope",
    "valid_power_exponent",
    "valid_barrels",
    "link_type_conduit",
    "shape_circular",
    "shape_rect_open",
    "shape_rect_closed",
    "shape_trapezoidal",
    "shape_power",
    "shape_other",
)
DYNAMIC_FEATURE_NAMES_V441 = (
    "delta_head_norm",
    "hydraulic_gradient_norm",
    "orientation_sign",
)


def _iter_inp_rows(path: str | Path) -> Iterable[tuple[str, list[str]]]:
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        yield section, line.split()


def _rows_by_section(path: str | Path) -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = defaultdict(list)
    for section, tokens in _iter_inp_rows(path):
        rows[section].append(tokens)
    return rows


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _has_number(value: str | None) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _bool_flag(value: str | None) -> bool:
    return str(value or "").strip().upper() in {"YES", "TRUE", "1", "ON"}


def _length(value: str | None, system_units: str) -> float:
    return float(length_to_m(max(0.0, _float(value)), system_units))


def _shape_name(value: str | None) -> str:
    value = str(value or "").strip().upper().replace("-", "_")
    return value if value in SUPPORTED_CONDUIT_SHAPES_V441 else "OTHER"


@dataclass(frozen=True)
class XSectionPhysicalV441:
    link_id: str
    shape: str
    geometry: tuple[float, float, float, float]
    barrels: float
    barrels_defaulted: bool
    culvert: str = ""


@dataclass(frozen=True)
class OrificePhysicalV441:
    link_id: str
    orifice_type: str
    offset: float
    discharge_coefficient: float
    flap_gate: bool
    open_close_hours: float
    xsection_shape: str
    xsection_geometry: tuple[float, float, float, float]


@dataclass(frozen=True)
class WeirPhysicalV441:
    link_id: str
    weir_type: str
    crest_offset: float
    discharge_coefficient: float
    flap_gate: bool
    end_contractions: float
    secondary_coefficient: float
    surcharge: bool
    xsection_shape: str
    xsection_geometry: tuple[float, float, float, float]


@dataclass(frozen=True)
class PumpPhysicalV441:
    link_id: str
    pump_curve_id: str
    initial_status: str
    startup_depth: float
    shutoff_depth: float


@dataclass(frozen=True)
class PhysicalLinkV441:
    link_id: str
    link_type: str
    from_node: str
    to_node: str
    static_features: np.ndarray
    shape: str
    shape_geometry: tuple[float, float, float, float]
    barrels: float
    barrels_defaulted: bool
    length_m: float
    roughness_n: float
    inlet_offset_m: float
    outlet_offset_m: float
    link_offsets_semantics: str
    upstream_link_invert_elevation_m: float | None
    downstream_link_invert_elevation_m: float | None
    invert_slope: float
    losses: tuple[float, float, float, float]
    supported_for_propagation: bool
    original_orientation_sign: int = 1
    orifice: OrificePhysicalV441 | None = None
    weir: WeirPhysicalV441 | None = None
    pump: PumpPhysicalV441 | None = None

    @property
    def raw_geometry(self) -> tuple[float, float, float, float]:
        return self.shape_geometry

    @property
    def unordered_node_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.from_node, self.to_node)))


@dataclass(frozen=True)
class EdgeFeatureNormalizationV441:
    feature_names: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    transform: tuple[str, ...]
    source: str = "frozen Train-only static graph statistics"


@dataclass(frozen=True)
class PhysicalDirectedEdgeLineageV441:
    edge_index: np.ndarray
    edge_to_link_id: tuple[str, ...]
    edge_to_link_type: tuple[str, ...]
    orientation_signs: tuple[int, ...]
    edge_static_features: np.ndarray
    edge_static_feature_names: tuple[str, ...]
    edge_lengths_m: np.ndarray
    src_node_ids: tuple[str, ...]
    dst_node_ids: tuple[str, ...]
    dynamic_feature_names: tuple[str, ...] = DYNAMIC_FEATURE_NAMES_V441
    uses_future_truth: bool = False
    uses_online_link_flow: bool = False
    propagation_link_types: tuple[str, ...] = ("conduit",)


def _parse_options(rows: Sequence[Sequence[str]]) -> dict[str, str]:
    options: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2:
            options[str(row[0]).upper()] = str(row[1]).upper()
    return options


def _parse_node_inverts(
    rows: Mapping[str, Sequence[Sequence[str]]], system_units: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for section in ("JUNCTIONS", "STORAGE", "OUTFALLS"):
        for row in rows.get(section, ()):
            if len(row) >= 2:
                result[row[0]] = _length(row[1], system_units)
    return result


def _shape_contract(
    rows: Sequence[Sequence[str]], system_units: str
) -> dict[str, XSectionPhysicalV441]:
    result: dict[str, XSectionPhysicalV441] = {}
    for row in rows:
        if len(row) < 2:
            continue
        raw_shape = str(row[1]).upper()
        shape = _shape_name(raw_shape)
        # SWMM XSECTIONS: only geometry fields with length semantics are
        # converted.  Side slopes and power exponents stay dimensionless.
        raw = [_float(row[i]) if i < len(row) else 0.0 for i in range(2, 6)]
        if raw_shape in {"CIRCULAR", "RECT_OPEN", "RECT_CLOSED", "TRAPEZOIDAL", "POWER"}:
            g1 = _length(str(raw[0]), system_units)
            g2 = _length(str(raw[1]), system_units) if raw_shape != "CIRCULAR" else 0.0
            if raw_shape == "TRAPEZOIDAL":
                geometry = (g1, g2, float(raw[2]), float(raw[3]))
            elif raw_shape == "POWER":
                geometry = (g1, g2, float(raw[2]), float(raw[3]))
            else:
                geometry = (g1, g2, 0.0, 0.0)
        else:
            geometry = tuple(float(value) for value in raw)
        barrels_defaulted = len(row) <= 6 or not _has_number(row[6])
        barrels = _float(row[6], 1.0) if not barrels_defaulted else 1.0
        if barrels <= 0.0:
            barrels = 1.0
            barrels_defaulted = True
        result[row[0]] = XSectionPhysicalV441(
            link_id=row[0],
            shape=raw_shape,
            geometry=tuple(float(value) for value in geometry),
            barrels=float(barrels),
            barrels_defaulted=bool(barrels_defaulted),
            culvert=row[7] if len(row) > 7 else "",
        )
    return result


def _loss_contract(
    rows: Sequence[Sequence[str]],
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    for row in rows:
        if len(row) < 4:
            continue
        result[row[0]] = (
            _float(row[1]),
            _float(row[2]),
            _float(row[3]),
            1.0 if _bool_flag(row[4] if len(row) > 4 else None) else 0.0,
        )
    return result


def _shape_values(
    shape: str,
    geometry: tuple[float, float, float, float],
    barrels: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Return semantic shape values and validity flags."""

    shape = str(shape).upper()
    g1, g2, g3, g4 = geometry
    full = width = left = right = exponent = 0.0
    full_valid = width_valid = left_valid = right_valid = exponent_valid = 0.0
    if shape == "CIRCULAR":
        full, full_valid = g1, 1.0
    elif shape in {"RECT_OPEN", "RECT_CLOSED"}:
        full, width = g1, g2
        full_valid = width_valid = 1.0
    elif shape == "TRAPEZOIDAL":
        full, width, left, right = g1, g2, g3, g4
        full_valid = width_valid = left_valid = right_valid = 1.0
    elif shape == "POWER":
        full, width, exponent = g1, g2, g3
        full_valid = width_valid = exponent_valid = 1.0
    return (
        full,
        width,
        left,
        right,
        exponent,
        full_valid,
        width_valid,
        left_valid,
        right_valid,
        exponent_valid,
        1.0 if barrels > 0.0 else 0.0,
    )


def _conduit_features(link: PhysicalLinkV441) -> np.ndarray:
    values = np.zeros(len(CONDUIT_FEATURE_NAMES_V441), dtype=np.float32)
    full, width, left, right, exponent, vf, vw, vl, vr, vp, vb = _shape_values(
        link.shape, link.shape_geometry, link.barrels
    )
    values[:15] = np.asarray(
        (
            np.log1p(max(link.length_m, 0.0)),
            link.roughness_n,
            link.invert_slope,
            link.inlet_offset_m,
            link.outlet_offset_m,
            full,
            width,
            left,
            right,
            exponent,
            link.barrels,
            link.losses[0],
            link.losses[1],
            link.losses[2],
            link.losses[3],
        ),
        dtype=np.float32,
    )
    values[15:26] = np.asarray(
        (
            1.0 if link.length_m > 0.0 else 0.0,
            1.0 if link.roughness_n > 0.0 else 0.0,
            1.0 if np.isfinite(link.invert_slope) else 0.0,
            1.0 if np.isfinite(link.inlet_offset_m) else 0.0,
            1.0 if np.isfinite(link.outlet_offset_m) else 0.0,
            vf,
            vw,
            vl,
            vr,
            vp,
            vb,
        ),
        dtype=np.float32,
    )
    values[26] = 1.0
    shape_offset = 27
    shape_key = link.shape if link.shape in SUPPORTED_CONDUIT_SHAPES_V441 else "OTHER"
    values[shape_offset + SHAPE_CENSUS_NAMES_V441.index(shape_key)] = 1.0
    return values


def parse_frozen_inp_physical_links_v441(
    inp_path: str | Path,
    node_ids: Sequence[str],
) -> tuple[PhysicalLinkV441, ...]:
    """Parse physical links with explicit SWMM field semantics."""

    node_set = {str(value) for value in node_ids}
    if len(node_set) != len(tuple(node_ids)):
        raise ValueError("graph node IDs must be unique")
    rows = _rows_by_section(inp_path)
    system_units = infer_system_units(inp_path)
    options = _parse_options(rows.get("OPTIONS", ()))
    link_offsets = options.get("LINK_OFFSETS", "DEPTH").upper()
    if link_offsets not in LINK_OFFSETS_VALUES_V441:
        raise ValueError(f"unsupported LINK_OFFSETS semantics: {link_offsets}")
    node_inverts = _parse_node_inverts(rows, system_units)
    xsections = _shape_contract(rows.get("XSECTIONS", ()), system_units)
    losses = _loss_contract(rows.get("LOSSES", ()))
    links: list[PhysicalLinkV441] = []

    def xsec(link_id: str) -> XSectionPhysicalV441:
        return xsections.get(
            link_id,
            XSectionPhysicalV441(link_id, "", (0.0, 0.0, 0.0, 0.0), 1.0, True),
        )

    def check_nodes(link_id: str, from_node: str, to_node: str) -> None:
        if from_node not in node_set or to_node not in node_set:
            raise ValueError(f"physical link {link_id} references node outside graph: {from_node}->{to_node}")

    for tokens in rows.get("CONDUITS", ()):
        if len(tokens) < 3:
            continue
        link_id, from_node, to_node = tokens[:3]
        check_nodes(link_id, from_node, to_node)
        section = xsec(link_id)
        length_m = _length(tokens[3] if len(tokens) > 3 else None, system_units)
        roughness = _float(tokens[4] if len(tokens) > 4 else None)
        inlet_offset = _length(tokens[5] if len(tokens) > 5 else None, system_units)
        outlet_offset = _length(tokens[6] if len(tokens) > 6 else None, system_units)
        if link_offsets == "ELEVATION":
            up_link_invert = inlet_offset
            down_link_invert = outlet_offset
        else:
            up_link_invert = node_inverts.get(from_node, 0.0) + inlet_offset
            down_link_invert = node_inverts.get(to_node, 0.0) + outlet_offset
        inlet_relative_offset = up_link_invert - node_inverts.get(from_node, 0.0)
        outlet_relative_offset = down_link_invert - node_inverts.get(to_node, 0.0)
        slope = (up_link_invert - down_link_invert) / length_m if length_m > 0.0 else float("nan")
        section_shape = section.shape
        supported = length_m > 0.0 and section_shape in SUPPORTED_CONDUIT_SHAPES_V441
        link = PhysicalLinkV441(
            link_id=link_id,
            link_type="conduit",
            from_node=from_node,
            to_node=to_node,
            static_features=np.empty((0,), dtype=np.float32),
            shape=section_shape,
            shape_geometry=section.geometry,
            barrels=section.barrels,
            barrels_defaulted=section.barrels_defaulted,
            length_m=length_m,
            roughness_n=roughness,
            inlet_offset_m=float(inlet_relative_offset),
            outlet_offset_m=float(outlet_relative_offset),
            link_offsets_semantics=link_offsets,
            upstream_link_invert_elevation_m=up_link_invert,
            downstream_link_invert_elevation_m=down_link_invert,
            invert_slope=float(slope),
            losses=losses.get(link_id, (0.0, 0.0, 0.0, 0.0)),
            supported_for_propagation=supported,
        )
        object.__setattr__(link, "static_features", _conduit_features(link))
        links.append(link)

    for tokens in rows.get("PUMPS", ()):
        if len(tokens) < 3:
            continue
        link_id, from_node, to_node = tokens[:3]
        check_nodes(link_id, from_node, to_node)
        pump = PumpPhysicalV441(
            link_id=link_id,
            pump_curve_id=tokens[3] if len(tokens) > 3 else "",
            initial_status=tokens[4].upper() if len(tokens) > 4 else "",
            startup_depth=_float(tokens[5] if len(tokens) > 5 else None),
            shutoff_depth=_float(tokens[6] if len(tokens) > 6 else None),
        )
        links.append(
            PhysicalLinkV441(
                link_id, "pump", from_node, to_node, np.empty((0,), np.float32), "", (0.0,) * 4,
                1.0, True, 0.0, 0.0, 0.0, 0.0, link_offsets, None, None, 0.0,
                (0.0, 0.0, 0.0, 0.0), False, pump=pump,
            )
        )

    for tokens in rows.get("ORIFICES", ()):
        if len(tokens) < 3:
            continue
        link_id, from_node, to_node = tokens[:3]
        check_nodes(link_id, from_node, to_node)
        section = xsec(link_id)
        orifice = OrificePhysicalV441(
            link_id=link_id,
            orifice_type=tokens[3].upper() if len(tokens) > 3 else "",
            offset=_length(tokens[4] if len(tokens) > 4 else None, system_units),
            discharge_coefficient=_float(tokens[5] if len(tokens) > 5 else None),
            flap_gate=_bool_flag(tokens[6] if len(tokens) > 6 else None),
            open_close_hours=_float(tokens[7] if len(tokens) > 7 else None),
            xsection_shape=section.shape,
            xsection_geometry=section.geometry,
        )
        links.append(
            PhysicalLinkV441(
                link_id, "orifice", from_node, to_node, np.empty((0,), np.float32), "", section.geometry,
                section.barrels, section.barrels_defaulted, 0.0, 0.0, orifice.offset, 0.0, link_offsets,
                None, None, 0.0, (0.0, 0.0, 0.0, 1.0 if orifice.flap_gate else 0.0), False, orifice=orifice,
            )
        )

    for tokens in rows.get("WEIRS", ()):
        if len(tokens) < 3:
            continue
        link_id, from_node, to_node = tokens[:3]
        check_nodes(link_id, from_node, to_node)
        section = xsec(link_id)
        weir = WeirPhysicalV441(
            link_id=link_id,
            weir_type=tokens[3].upper() if len(tokens) > 3 else "",
            crest_offset=_length(tokens[4] if len(tokens) > 4 else None, system_units),
            discharge_coefficient=_float(tokens[5] if len(tokens) > 5 else None),
            flap_gate=_bool_flag(tokens[6] if len(tokens) > 6 else None),
            end_contractions=_float(tokens[7] if len(tokens) > 7 else None),
            secondary_coefficient=_float(tokens[8] if len(tokens) > 8 else None),
            surcharge=_bool_flag(tokens[9] if len(tokens) > 9 else None),
            xsection_shape=section.shape,
            xsection_geometry=section.geometry,
        )
        links.append(
            PhysicalLinkV441(
                link_id, "weir", from_node, to_node, np.empty((0,), np.float32), "", section.geometry,
                section.barrels, section.barrels_defaulted, 0.0, 0.0, weir.crest_offset, 0.0, link_offsets,
                None, None, 0.0, (0.0, 0.0, 0.0, 1.0 if weir.flap_gate else 0.0), False, weir=weir,
            )
        )

    for tokens in rows.get("OUTLETS", ()):
        if len(tokens) < 3:
            continue
        link_id, from_node, to_node = tokens[:3]
        check_nodes(link_id, from_node, to_node)
        links.append(
            PhysicalLinkV441(
                link_id, "outlet", from_node, to_node, np.empty((0,), np.float32), "", (0.0,) * 4,
                1.0, True, 0.0, 0.0, _length(tokens[3] if len(tokens) > 3 else None, system_units), 0.0,
                link_offsets, None, None, 0.0, (0.0, 0.0, 0.0, 0.0), False,
            )
        )
    if not links:
        raise ValueError(f"no physical hydraulic links found in {inp_path}")
    return tuple(links)


def build_conduit_directed_edge_lineage_v441(
    links: Sequence[PhysicalLinkV441], node_index: Mapping[str, int]
) -> PhysicalDirectedEdgeLineageV441:
    rows: list[tuple[int, int]] = []
    edge_ids: list[str] = []
    edge_types: list[str] = []
    signs: list[int] = []
    features: list[np.ndarray] = []
    lengths: list[float] = []
    src_ids: list[str] = []
    dst_ids: list[str] = []
    for link in links:
        if link.link_type != "conduit":
            continue
        if not link.supported_for_propagation:
            raise ValueError(f"conduit {link.link_id} is unsupported for V4.4.1 propagation")
        if link.length_m <= 0.0 or not np.isfinite(link.length_m):
            raise ValueError(f"conduit {link.link_id} must have positive finite length")
        if link.from_node not in node_index or link.to_node not in node_index:
            raise ValueError(f"conduit {link.link_id} cannot be indexed in graph")
        u, v = int(node_index[link.from_node]), int(node_index[link.to_node])
        for src, dst, sign, src_name, dst_name in (
            (u, v, 1, link.from_node, link.to_node),
            (v, u, -1, link.to_node, link.from_node),
        ):
            rows.append((src, dst))
            edge_ids.append(link.link_id)
            edge_types.append("conduit")
            signs.append(sign)
            features.append(link.static_features)
            lengths.append(link.length_m)
            src_ids.append(src_name)
            dst_ids.append(dst_name)
    if not rows:
        raise ValueError("no supported conduits available for V4.4.1 propagation")
    return PhysicalDirectedEdgeLineageV441(
        edge_index=np.asarray(rows, dtype=np.int64).T,
        edge_to_link_id=tuple(edge_ids),
        edge_to_link_type=tuple(edge_types),
        orientation_signs=tuple(signs),
        edge_static_features=np.asarray(features, dtype=np.float32),
        edge_static_feature_names=tuple(CONDUIT_FEATURE_NAMES_V441),
        edge_lengths_m=np.asarray(lengths, dtype=np.float32),
        src_node_ids=tuple(src_ids),
        dst_node_ids=tuple(dst_ids),
    )


def normalize_conduit_static_features_v441(
    features: np.ndarray,
    feature_names: Sequence[str] = CONDUIT_FEATURE_NAMES_V441,
) -> tuple[np.ndarray, EdgeFeatureNormalizationV441]:
    values = np.asarray(features, dtype=np.float32)
    names = tuple(feature_names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("conduit static features must be [E,F] with matching names")
    if not np.isfinite(values).all():
        raise ValueError("conduit static features must be finite")
    transformed = values.astype(np.float64, copy=True)
    location = np.median(transformed, axis=0)
    q25, q75 = np.percentile(transformed, (25.0, 75.0), axis=0)
    scale = np.maximum(q75 - q25, 1.0)
    transforms = ["identity"] * len(names)
    categorical = np.asarray(
        [name.startswith(("shape_", "link_type_", "valid_")) or name == "flap_gate" for name in names],
        dtype=bool,
    )
    location[categorical] = 0.0
    scale[categorical] = 1.0
    normalized = ((transformed - location) / scale).astype(np.float32)
    return normalized, EdgeFeatureNormalizationV441(
        feature_names=names,
        location=location.astype(np.float32),
        scale=scale.astype(np.float32),
        transform=tuple(transforms),
    )


def causal_edge_dynamic_features_v441(
    node_context: np.ndarray,
    lineage: PhysicalDirectedEdgeLineageV441,
    *,
    current_head: np.ndarray,
    current_depth: np.ndarray,
    head_scale_train: float,
    gradient_scale_train: float,
) -> np.ndarray:
    """Return dimensionless causal dynamic edge features.

    ``current_head``/``current_depth`` must be current or model-predicted
    reference state.  No future target state or link flow is accepted.
    """

    context = np.asarray(node_context, dtype=np.float32)
    head = np.asarray(current_head, dtype=np.float32)
    depth = np.asarray(current_depth, dtype=np.float32)
    if context.ndim not in (3, 4):
        raise ValueError("node_context must be [B,N,D] or [B,H,N,D]")
    expected = context.shape[:2] if context.ndim == 3 else context.shape[:3]
    if head.shape != expected or depth.shape != expected:
        raise ValueError("current_head/current_depth shapes must match causal context")
    if not np.isfinite(context).all() or not np.isfinite(head).all() or not np.isfinite(depth).all():
        raise ValueError("causal node context and hydraulic values must be finite")
    head_scale = float(head_scale_train)
    gradient_scale = float(gradient_scale_train)
    if not np.isfinite(head_scale) or head_scale <= 0.0:
        raise ValueError("head_scale_train must be positive finite")
    if not np.isfinite(gradient_scale) or gradient_scale <= 0.0:
        raise ValueError("gradient_scale_train must be positive finite")
    src = np.asarray(lineage.edge_index[0], dtype=np.int64)
    dst = np.asarray(lineage.edge_index[1], dtype=np.int64)
    lengths = np.asarray(lineage.edge_lengths_m, dtype=np.float32)
    if lengths.shape != (src.size,) or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("V4.4.1 dynamic features require positive conduit edge lengths")
    if context.ndim == 3:
        delta = head[:, src] - head[:, dst]
        raw_gradient = delta / lengths[None, :]
        delta_norm = delta / head_scale
        gradient_norm = np.sign(raw_gradient) * np.log1p(np.abs(raw_gradient) / gradient_scale)
        orientation = np.asarray(lineage.orientation_signs, dtype=np.float32)[None].repeat(context.shape[0], axis=0)
    else:
        delta = head[:, :, src] - head[:, :, dst]
        raw_gradient = delta / lengths[None, None, :]
        delta_norm = delta / head_scale
        gradient_norm = np.sign(raw_gradient) * np.log1p(np.abs(raw_gradient) / gradient_scale)
        orientation = np.asarray(lineage.orientation_signs, dtype=np.float32)[None, None].repeat(context.shape[0], axis=0)
        orientation = orientation.repeat(context.shape[1], axis=1)
    return np.stack((delta_norm, gradient_norm, orientation), axis=-1).astype(np.float32, copy=False)


def physical_link_census_v441(links: Sequence[PhysicalLinkV441]) -> dict[str, int | bool]:
    pairs = Counter(link.unordered_node_pair for link in links)
    return {
        "physical_links": len(links),
        **{f"{link_type}s": sum(link.link_type == link_type for link in links) for link_type in ("conduit", "pump", "orifice", "weir", "outlet")},
        "single_link_node_pairs": sum(count == 1 for count in pairs.values()),
        "multi_link_node_pairs": sum(count > 1 for count in pairs.values()),
        "maximum_links_per_pair": max(pairs.values(), default=0),
    }


def shape_census_v441(links: Sequence[PhysicalLinkV441]) -> dict[str, int]:
    return dict(sorted(Counter(link.shape or "OTHER" for link in links if link.link_type == "conduit").items()))


__all__ = [
    "CONDUIT_FEATURE_NAMES_V441",
    "DYNAMIC_FEATURE_NAMES_V441",
    "EDGE_FEATURE_CONTRACT_V441",
    "EdgeFeatureNormalizationV441",
    "OrificePhysicalV441",
    "PhysicalDirectedEdgeLineageV441",
    "PhysicalLinkV441",
    "PumpPhysicalV441",
    "WeirPhysicalV441",
    "build_conduit_directed_edge_lineage_v441",
    "causal_edge_dynamic_features_v441",
    "normalize_conduit_static_features_v441",
    "parse_frozen_inp_physical_links_v441",
    "physical_link_census_v441",
    "shape_census_v441",
]
