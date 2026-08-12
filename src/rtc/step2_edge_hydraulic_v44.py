"""Frozen-INP physical-link lineage and causal edge features for Step2 V4.4.

This module is intentionally independent of SWMM execution.  It parses only
static INP definitions, retains one identity per physical hydraulic link, and
creates forward/reverse directed edges without collapsing parallel links.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .graph import infer_system_units
from .units import length_to_m


LINK_SECTIONS_V44 = ("CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS")
LINK_TYPE_NAMES_V44 = ("conduit", "pump", "orifice", "weir", "outlet")
SHAPE_NAMES_V44 = (
    "circular",
    "rect_open",
    "rect_closed",
    "elliptical",
    "arch",
    "trapezoidal",
    "custom",
    "irregular",
    "other",
)
STATIC_FEATURE_NAMES_V44 = (
    "length_m",
    "roughness_n",
    "inlet_offset_m",
    "outlet_offset_m",
    "geom1_m",
    "geom2_m",
    "geom3_m",
    "geom4_m",
    "barrels",
    "entrance_loss",
    "exit_loss",
    "average_loss",
    "flap_gate",
    *(f"link_type_{name}" for name in LINK_TYPE_NAMES_V44),
    *(f"shape_{name}" for name in SHAPE_NAMES_V44),
)
DYNAMIC_FEATURE_NAMES_V44 = (
    "head_src",
    "head_dst",
    "delta_head",
    "hydraulic_gradient",
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


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _bool_flag(value: str | None) -> float:
    return 1.0 if str(value or "").upper() in {"YES", "TRUE", "1"} else 0.0


def _shape_key(shape: str) -> str:
    key = str(shape or "").strip().lower().replace("-", "_")
    return key if key in SHAPE_NAMES_V44[:-1] else "other"


def _length(value: str | None, system_units: str) -> float:
    return float(length_to_m(max(0.0, _float(value)), system_units))


def _shape_contract(rows: Sequence[Sequence[str]]) -> dict[str, tuple[str, tuple[float, float, float, float], float]]:
    result: dict[str, tuple[str, tuple[float, float, float, float], float]] = {}
    for row in rows:
        if len(row) < 3:
            continue
        values = tuple(_float(row[i]) if i < len(row) else 0.0 for i in range(2, 6))
        barrels = _float(row[6]) if len(row) > 6 else 0.0
        result[row[0]] = (row[1].upper(), values, barrels)
    return result


def _loss_contract(rows: Sequence[Sequence[str]]) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    for row in rows:
        if len(row) < 4:
            continue
        result[row[0]] = (
            _float(row[1]),
            _float(row[2]),
            _float(row[3]),
            _bool_flag(row[4]) if len(row) > 4 else 0.0,
        )
    return result


@dataclass(frozen=True)
class PhysicalLinkV44:
    link_id: str
    link_type: str
    from_node: str
    to_node: str
    static_features: np.ndarray
    shape: str
    raw_geometry: tuple[float, float, float, float]
    barrels: float
    original_orientation_sign: int = 1

    @property
    def unordered_node_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.from_node, self.to_node)))


@dataclass(frozen=True)
class EdgeFeatureNormalizationV44:
    feature_names: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    transform: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalDirectedEdgeLineageV44:
    edge_index: np.ndarray
    edge_to_link_id: tuple[str, ...]
    edge_to_link_type: tuple[str, ...]
    orientation_signs: tuple[int, ...]
    edge_static_features: np.ndarray
    edge_static_feature_names: tuple[str, ...]
    src_node_ids: tuple[str, ...]
    dst_node_ids: tuple[str, ...]
    dynamic_feature_names: tuple[str, ...] = DYNAMIC_FEATURE_NAMES_V44
    uses_future_truth: bool = False
    uses_online_link_flow: bool = False


def _build_static_features(
    *,
    link_type: str,
    length_m: float = 0.0,
    roughness_n: float = 0.0,
    inlet_offset_m: float = 0.0,
    outlet_offset_m: float = 0.0,
    geometry: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    barrels: float = 0.0,
    losses: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    shape: str = "",
) -> np.ndarray:
    values = np.zeros(len(STATIC_FEATURE_NAMES_V44), dtype=np.float32)
    values[0:9] = np.asarray(
        (
            length_m,
            roughness_n,
            inlet_offset_m,
            outlet_offset_m,
            *geometry,
            barrels,
        ),
        dtype=np.float32,
    )
    values[9:13] = np.asarray(losses, dtype=np.float32)
    type_offset = 13 + LINK_TYPE_NAMES_V44.index(link_type)
    values[type_offset] = 1.0
    shape_offset = 13 + len(LINK_TYPE_NAMES_V44)
    values[shape_offset + SHAPE_NAMES_V44.index(_shape_key(shape))] = 1.0
    return values


def parse_frozen_inp_physical_links_v44(
    inp_path: str | Path,
    node_ids: Sequence[str],
) -> tuple[PhysicalLinkV44, ...]:
    """Parse static physical links while retaining parallel-link identity."""

    node_set = set(str(value) for value in node_ids)
    if len(node_set) != len(tuple(node_ids)):
        raise ValueError("graph node IDs must be unique")
    rows: dict[str, list[list[str]]] = defaultdict(list)
    for section, tokens in _iter_inp_rows(inp_path):
        if section in {*LINK_SECTIONS_V44, "XSECTIONS", "LOSSES"}:
            rows[section].append(tokens)
    system_units = infer_system_units(inp_path)
    xsections = _shape_contract(rows["XSECTIONS"])
    losses = _loss_contract(rows["LOSSES"])
    links: list[PhysicalLinkV44] = []

    for section in LINK_SECTIONS_V44:
        link_type = section[:-1].lower() if section != "OUTLETS" else "outlet"
        for tokens in rows[section]:
            if len(tokens) < 3:
                continue
            link_id, from_node, to_node = tokens[:3]
            if from_node not in node_set or to_node not in node_set:
                raise ValueError(f"physical link {link_id} references node outside graph: {from_node}->{to_node}")
            shape, geometry, barrels = xsections.get(link_id, ("", (0.0, 0.0, 0.0, 0.0), 0.0))
            link_losses = losses.get(link_id, (0.0, 0.0, 0.0, 0.0))
            if link_type == "conduit":
                static = _build_static_features(
                    link_type=link_type,
                    length_m=_length(tokens[3] if len(tokens) > 3 else None, system_units),
                    roughness_n=_float(tokens[4] if len(tokens) > 4 else None),
                    inlet_offset_m=_length(tokens[5] if len(tokens) > 5 else None, system_units),
                    outlet_offset_m=_length(tokens[6] if len(tokens) > 6 else None, system_units),
                    geometry=tuple(_length(str(value), system_units) for value in geometry),
                    barrels=barrels,
                    losses=link_losses,
                    shape=shape,
                )
            elif link_type == "orifice":
                static = _build_static_features(
                    link_type=link_type,
                    inlet_offset_m=_length(tokens[4] if len(tokens) > 4 else None, system_units),
                    geometry=(_length(tokens[5] if len(tokens) > 5 else None, system_units), _float(tokens[6] if len(tokens) > 6 else None), 0.0, 0.0),
                    losses=(0.0, 0.0, 0.0, _bool_flag(tokens[7] if len(tokens) > 7 else None)),
                )
            elif link_type == "weir":
                static = _build_static_features(
                    link_type=link_type,
                    inlet_offset_m=_length(tokens[4] if len(tokens) > 4 else None, system_units),
                    geometry=(_float(tokens[5] if len(tokens) > 5 else None), _float(tokens[6] if len(tokens) > 6 else None), 0.0, 0.0),
                    losses=(0.0, 0.0, 0.0, _bool_flag(tokens[7] if len(tokens) > 7 else None)),
                )
            else:
                static = _build_static_features(
                    link_type=link_type,
                    shape=tokens[3] if len(tokens) > 3 else "",
                )
            links.append(
                PhysicalLinkV44(
                    link_id=link_id,
                    link_type=link_type,
                    from_node=from_node,
                    to_node=to_node,
                    static_features=static,
                    shape=shape if link_type == "conduit" else "",
                    raw_geometry=tuple(float(value) for value in geometry),
                    barrels=float(barrels),
                )
            )
    if not links:
        raise ValueError(f"no physical hydraulic links found in {inp_path}")
    return tuple(links)


def build_physical_directed_edge_lineage_v44(
    links: Sequence[PhysicalLinkV44],
    node_index: dict[str, int],
) -> PhysicalDirectedEdgeLineageV44:
    """Create forward/reverse directed edges for every physical link."""

    edge_rows: list[tuple[int, int]] = []
    edge_ids: list[str] = []
    edge_types: list[str] = []
    orientations: list[int] = []
    edge_features: list[np.ndarray] = []
    src_ids: list[str] = []
    dst_ids: list[str] = []
    for link in links:
        if link.from_node not in node_index or link.to_node not in node_index:
            raise ValueError(f"link {link.link_id} cannot be indexed in graph")
        u, v = int(node_index[link.from_node]), int(node_index[link.to_node])
        for src, dst, sign in ((u, v, 1), (v, u, -1)):
            edge_rows.append((src, dst))
            edge_ids.append(link.link_id)
            edge_types.append(link.link_type)
            orientations.append(sign)
            edge_features.append(link.static_features)
            src_ids.append(link.from_node if sign == 1 else link.to_node)
            dst_ids.append(link.to_node if sign == 1 else link.from_node)
    return PhysicalDirectedEdgeLineageV44(
        edge_index=np.asarray(edge_rows, dtype=np.int64).T,
        edge_to_link_id=tuple(edge_ids),
        edge_to_link_type=tuple(edge_types),
        orientation_signs=tuple(orientations),
        edge_static_features=np.asarray(edge_features, dtype=np.float32),
        edge_static_feature_names=tuple(STATIC_FEATURE_NAMES_V44),
        src_node_ids=tuple(src_ids),
        dst_node_ids=tuple(dst_ids),
    )


def normalize_edge_static_features_v44(
    features: np.ndarray,
    feature_names: Sequence[str] = STATIC_FEATURE_NAMES_V44,
) -> tuple[np.ndarray, EdgeFeatureNormalizationV44]:
    """Apply deterministic Train-only static transforms and robust scaling."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(tuple(feature_names)):
        raise ValueError("edge static features must be [E,F] with matching feature names")
    if not np.isfinite(values).all():
        raise ValueError("edge static features must be finite")
    transformed = values.astype(np.float64, copy=True)
    transforms: list[str] = []
    for column, name in enumerate(feature_names):
        if name in {"length_m", "geom1_m", "geom2_m", "geom3_m", "geom4_m", "roughness_n"}:
            transformed[:, column] = np.log1p(np.maximum(transformed[:, column], 0.0))
            transforms.append("log1p")
        else:
            transforms.append("identity")
    location = np.median(transformed, axis=0)
    q25, q75 = np.percentile(transformed, (25.0, 75.0), axis=0)
    scale = np.maximum(q75 - q25, 1.0)
    categorical = np.asarray(
        [name.startswith("link_type_") or name.startswith("shape_") or name == "flap_gate" for name in feature_names],
        dtype=bool,
    )
    location[categorical] = 0.0
    scale[categorical] = 1.0
    normalized = ((transformed - location) / scale).astype(np.float32)
    return normalized, EdgeFeatureNormalizationV44(
        feature_names=tuple(feature_names),
        location=location.astype(np.float32),
        scale=scale.astype(np.float32),
        transform=tuple(transforms),
    )


def causal_edge_dynamic_features_v44(
    node_context: np.ndarray,
    lineage: PhysicalDirectedEdgeLineageV44,
    *,
    current_head: np.ndarray,
    current_depth: np.ndarray,
) -> np.ndarray:
    """Build only current/reference causal head-gradient edge features.

    ``node_context`` may be ``[B,N,D]`` or ``[B,H,N,D]``.  The depth argument
    is validated as a causal context contract; link flow is deliberately not
    accepted because it is unavailable in the current online Step2 input.
    """

    context = np.asarray(node_context, dtype=np.float32)
    head = np.asarray(current_head, dtype=np.float32)
    depth = np.asarray(current_depth, dtype=np.float32)
    if context.ndim not in (3, 4):
        raise ValueError("node_context must be [B,N,D] or [B,H,N,D]")
    if context.ndim == 3 and head.shape != context.shape[:2]:
        raise ValueError("current_head must be [B,N]")
    if context.ndim == 3 and depth.shape != context.shape[:2]:
        raise ValueError("current_depth must be [B,N]")
    if context.ndim == 4 and head.shape != context.shape[:3]:
        raise ValueError("current_head must be [B,H,N]")
    if context.ndim == 4 and depth.shape != context.shape[:3]:
        raise ValueError("current_depth must be [B,H,N]")
    if not np.isfinite(context).all() or not np.isfinite(head).all() or not np.isfinite(depth).all():
        raise ValueError("causal node context must be finite")
    src = np.asarray([int(value) for value in lineage.edge_index[0]], dtype=np.int64)
    dst = np.asarray([int(value) for value in lineage.edge_index[1]], dtype=np.int64)
    lengths = np.asarray(lineage.edge_static_features[:, 0], dtype=np.float32)
    safe_length = np.maximum(lengths, 1.0e-6)
    if context.ndim == 3:
        src_head, dst_head = head[:, src], head[:, dst]
        output = np.stack((src_head, dst_head, src_head - dst_head, (src_head - dst_head) / safe_length[None]), axis=-1)
    else:
        src_head, dst_head = head[:, :, src], head[:, :, dst]
        output = np.stack((src_head, dst_head, src_head - dst_head, (src_head - dst_head) / safe_length[None, None]), axis=-1)
    return output.astype(np.float32, copy=False)


def physical_link_census_v44(links: Sequence[PhysicalLinkV44]) -> dict[str, int | bool]:
    pairs = Counter(link.unordered_node_pair for link in links)
    return {
        "physical_links": len(links),
        **{f"{link_type}s": sum(link.link_type == link_type for link in links) for link_type in LINK_TYPE_NAMES_V44},
        "single_link_node_pairs": sum(count == 1 for count in pairs.values()),
        "multi_link_node_pairs": sum(count > 1 for count in pairs.values()),
        "maximum_links_per_pair": max(pairs.values(), default=0),
    }


__all__ = [
    "DYNAMIC_FEATURE_NAMES_V44",
    "STATIC_FEATURE_NAMES_V44",
    "EdgeFeatureNormalizationV44",
    "PhysicalDirectedEdgeLineageV44",
    "PhysicalLinkV44",
    "build_physical_directed_edge_lineage_v44",
    "causal_edge_dynamic_features_v44",
    "normalize_edge_static_features_v44",
    "parse_frozen_inp_physical_links_v44",
    "physical_link_census_v44",
]
