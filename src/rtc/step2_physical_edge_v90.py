"""Frozen-INP conduit-only physical edge assets for V9 diagnostics.

This additive utility deliberately does not alter any V9 model, trainer, or
runner.  It turns the corrected V4.4.1 SWMM parser lineage into CPU tensors
that retain physical conduit identity, including parallel conduits.  Pumps,
orifices, weirs, and outlets remain available in parser lineage but are
excluded from propagation by construction.

The dynamic helper accepts only caller-supplied current or model-predicted
reference head/depth tensors.  It has no target, future-state, or link-flow
argument and is differentiable with respect to those supplied tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .step2_edge_physics_v441 import (
    CONDUIT_FEATURE_NAMES_V441,
    build_conduit_directed_edge_lineage_v441,
    normalize_conduit_static_features_v441,
    parse_frozen_inp_physical_links_v441,
)


PHYSICAL_EDGE_CONTRACT_V90 = "PROJECT7_STEP2_V90_CONDUIT_PHYSICAL_EDGE_ASSETS_V1"
STATIC_EDGE_FEATURE_NAMES_V90 = tuple(CONDUIT_FEATURE_NAMES_V441)
DYNAMIC_EDGE_FEATURE_NAMES_V90 = (
    "source_reference_head_norm",
    "destination_reference_head_norm",
    "source_reference_depth_norm",
    "destination_reference_depth_norm",
    "delta_head_norm",
    "hydraulic_gradient_norm",
    "orientation_sign",
)
_REGULATOR_LINK_TYPES = frozenset({"pump", "orifice", "weir"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_static_sha256(
    *,
    names: Sequence[str],
    location: np.ndarray,
    scale: np.ndarray,
    normalized: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(str(name) for name in names).encode("utf-8"))
    for values in (location, scale, normalized):
        contiguous = np.ascontiguousarray(values, dtype=np.float32)
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ConduitPhysicalEdgeAssetsV90:
    """Torch-ready, frozen physical conduit multiedges on CPU.

    ``edge_index`` deliberately retains repeated ``src,dst`` pairs.  Degrees
    count physical directed multiedges, not deduplicated topology neighbours.
    """

    contract: str
    inp_path: str
    inp_sha256: str
    node_count: int
    physical_link_count: int
    conduit_physical_link_count: int
    edge_index: torch.Tensor
    edge_to_link_id: tuple[str, ...]
    edge_to_link_type: tuple[str, ...]
    orientation_sign: torch.Tensor
    edge_length_m: torch.Tensor
    static_feature_names: tuple[str, ...]
    static_features_raw: torch.Tensor
    static_features_normalized: torch.Tensor
    static_normalization_location: torch.Tensor
    static_normalization_scale: torch.Tensor
    static_normalization_sha256: str
    in_degree: torch.Tensor
    out_degree: torch.Tensor
    excluded_regulator_link_ids: tuple[str, ...]
    excluded_nonconduit_link_ids: tuple[str, ...]
    regulator_propagation_edge_count: int
    uses_future_truth: bool = False
    uses_online_link_flow: bool = False

    @property
    def directed_edge_count(self) -> int:
        return int(self.edge_index.shape[1])


def build_conduit_physical_edge_assets_v90(
    *,
    inp_path: str | Path,
    expected_inp_sha256: str,
    node_ids: Sequence[str],
    expected_conduit_count: int | None = None,
    expected_directed_edge_count: int | None = None,
) -> ConduitPhysicalEdgeAssetsV90:
    """Build a fail-closed, conduit-only physical directed-edge asset.

    The caller must provide the expected SHA256 of the exact frozen INP.  The
    optional count assertions make a one-off frozen study census explicit
    without hard-coding its 1167 conduits / 2334 directed-edge totals here.
    """

    path = Path(inp_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"frozen INP path is not a file: {path}")
    expected_sha = str(expected_inp_sha256).strip().lower()
    if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
        raise ValueError("expected INP SHA256 must be a 64-character hexadecimal digest")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"frozen INP SHA256 mismatch: expected {expected_sha}, observed {actual_sha}"
        )

    ids = tuple(str(node_id) for node_id in node_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("node_ids must be nonempty and unique")
    node_index = {node_id: index for index, node_id in enumerate(ids)}
    links = parse_frozen_inp_physical_links_v441(path, ids)
    conduit_links = tuple(link for link in links if link.link_type == "conduit")
    if expected_conduit_count is not None and len(conduit_links) != int(expected_conduit_count):
        raise ValueError(
            f"conduit census mismatch: expected {int(expected_conduit_count)}, observed {len(conduit_links)}"
        )
    lineage = build_conduit_directed_edge_lineage_v441(links, node_index)
    if expected_directed_edge_count is not None and lineage.edge_index.shape[1] != int(expected_directed_edge_count):
        raise ValueError(
            "directed conduit census mismatch: "
            f"expected {int(expected_directed_edge_count)}, observed {lineage.edge_index.shape[1]}"
        )
    if len(STATIC_EDGE_FEATURE_NAMES_V90) != 33:
        raise RuntimeError("V9 physical edge contract requires exactly 33 conduit static features")
    if tuple(lineage.edge_static_feature_names) != STATIC_EDGE_FEATURE_NAMES_V90:
        raise RuntimeError("V4.4.1 conduit feature contract differs from V9 physical edge contract")
    if any(link_type != "conduit" for link_type in lineage.edge_to_link_type):
        raise RuntimeError("non-conduit regulator entered V9 physical propagation assets")
    if lineage.edge_index.shape[1] != 2 * len(conduit_links):
        raise RuntimeError("each physical conduit must produce exactly two directed physical edges")

    normalized, normalization = normalize_conduit_static_features_v441(
        lineage.edge_static_features,
        lineage.edge_static_feature_names,
    )
    edge_index_np = np.ascontiguousarray(lineage.edge_index, dtype=np.int64)
    lengths_np = np.ascontiguousarray(lineage.edge_lengths_m, dtype=np.float32)
    if not np.isfinite(lengths_np).all() or np.any(lengths_np <= 0.0):
        raise RuntimeError("conduit propagation assets require positive finite physical lengths")
    source = edge_index_np[0]
    destination = edge_index_np[1]
    out_degree_np = np.bincount(source, minlength=len(ids)).astype(np.int64, copy=False)
    in_degree_np = np.bincount(destination, minlength=len(ids)).astype(np.int64, copy=False)
    excluded_regulators = tuple(
        link.link_id for link in links if link.link_type in _REGULATOR_LINK_TYPES
    )
    excluded_nonconduits = tuple(link.link_id for link in links if link.link_type != "conduit")

    return ConduitPhysicalEdgeAssetsV90(
        contract=PHYSICAL_EDGE_CONTRACT_V90,
        inp_path=str(path),
        inp_sha256=actual_sha,
        node_count=len(ids),
        physical_link_count=len(links),
        conduit_physical_link_count=len(conduit_links),
        edge_index=torch.from_numpy(edge_index_np.copy()),
        edge_to_link_id=tuple(lineage.edge_to_link_id),
        edge_to_link_type=tuple(lineage.edge_to_link_type),
        orientation_sign=torch.tensor(lineage.orientation_signs, dtype=torch.float32),
        edge_length_m=torch.from_numpy(lengths_np.copy()),
        static_feature_names=STATIC_EDGE_FEATURE_NAMES_V90,
        static_features_raw=torch.from_numpy(
            np.ascontiguousarray(lineage.edge_static_features, dtype=np.float32).copy()
        ),
        static_features_normalized=torch.from_numpy(np.ascontiguousarray(normalized).copy()),
        static_normalization_location=torch.from_numpy(
            np.ascontiguousarray(normalization.location, dtype=np.float32).copy()
        ),
        static_normalization_scale=torch.from_numpy(
            np.ascontiguousarray(normalization.scale, dtype=np.float32).copy()
        ),
        static_normalization_sha256=_normalized_static_sha256(
            names=normalization.feature_names,
            location=normalization.location,
            scale=normalization.scale,
            normalized=normalized,
        ),
        in_degree=torch.from_numpy(in_degree_np.copy()),
        out_degree=torch.from_numpy(out_degree_np.copy()),
        excluded_regulator_link_ids=excluded_regulators,
        excluded_nonconduit_link_ids=excluded_nonconduits,
        regulator_propagation_edge_count=0,
    )


def _positive_finite_scale(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive finite")
    return result


def causal_reference_dynamic_edge_features_v90(
    assets: ConduitPhysicalEdgeAssetsV90,
    *,
    reference_head_m: torch.Tensor,
    reference_depth_m: torch.Tensor,
    head_scale_m: float,
    depth_scale_m: float,
    gradient_scale: float,
) -> torch.Tensor:
    """Return directed dimensionless features from causal reference hydraulics.

    Inputs are `[B,N]` or `[B,T,N]` tensors holding only current or
    model-predicted *reference* state.  The function intentionally has no
    future-target or SWMM-link-flow input.  It does not detach the supplied
    tensors, so action/state gradients remain available to a future caller.
    """

    if not isinstance(reference_head_m, torch.Tensor) or not isinstance(reference_depth_m, torch.Tensor):
        raise TypeError("reference_head_m and reference_depth_m must be torch tensors")
    if reference_head_m.shape != reference_depth_m.shape:
        raise ValueError("reference head/depth tensors must have identical shapes")
    if reference_head_m.ndim not in (2, 3) or reference_head_m.shape[-1] != assets.node_count:
        raise ValueError("reference head/depth must be [B,N] or [B,T,N] matching physical edge nodes")
    if not torch.is_floating_point(reference_head_m) or not torch.is_floating_point(reference_depth_m):
        raise TypeError("reference head/depth tensors must be floating point")
    if reference_head_m.device != reference_depth_m.device:
        raise ValueError("reference head/depth tensors must share a device")
    if not bool(torch.isfinite(reference_head_m).all()) or not bool(torch.isfinite(reference_depth_m).all()):
        raise ValueError("reference head/depth tensors must be finite")
    head_scale = _positive_finite_scale("head_scale_m", head_scale_m)
    depth_scale = _positive_finite_scale("depth_scale_m", depth_scale_m)
    gradient = _positive_finite_scale("gradient_scale", gradient_scale)

    device = reference_head_m.device
    dtype = reference_head_m.dtype
    edge_index = assets.edge_index.to(device=device)
    source, destination = edge_index[0], edge_index[1]
    length = assets.edge_length_m.to(device=device, dtype=dtype)
    orientation = assets.orientation_sign.to(device=device, dtype=dtype)
    source_head = reference_head_m[..., source]
    destination_head = reference_head_m[..., destination]
    source_depth = reference_depth_m[..., source]
    destination_depth = reference_depth_m[..., destination]
    delta_head = source_head - destination_head
    raw_gradient = delta_head / length
    gradient_norm = torch.sign(raw_gradient) * torch.log1p(torch.abs(raw_gradient) / gradient)
    orientation_view = orientation.expand_as(delta_head)
    return torch.stack(
        (
            source_head / head_scale,
            destination_head / head_scale,
            source_depth / depth_scale,
            destination_depth / depth_scale,
            delta_head / head_scale,
            gradient_norm,
            orientation_view,
        ),
        dim=-1,
    )


__all__ = [
    "ConduitPhysicalEdgeAssetsV90",
    "DYNAMIC_EDGE_FEATURE_NAMES_V90",
    "PHYSICAL_EDGE_CONTRACT_V90",
    "STATIC_EDGE_FEATURE_NAMES_V90",
    "build_conduit_physical_edge_assets_v90",
    "causal_reference_dynamic_edge_features_v90",
]
