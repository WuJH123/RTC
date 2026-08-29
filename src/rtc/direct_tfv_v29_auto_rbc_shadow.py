"""Hydraulic-utility sparsification of the causal Auto-RBC shadow for Project7 V29.

The Auto-RBC rule itself is unchanged. V29 changes only how a broad rule
proposal is compressed to the frozen changed-facility ceiling: facilities are
retained by hydraulic release utility rather than absolute SWMM SETTING delta.
The result remains a candidate only and is still subject to the common q95
sequence support before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .actuator_release_semantics import graph_release_setting_signs
from .direct_tfv_policy_return_portfolio import (
    _actuator_bounds,
    _node_feature,
    _validated_supervisory_mask,
)

V29_AUTO_RBC_UTILITY_SOURCE = "AUTO_RBC_SHADOW_HYDRAULIC_UTILITY_TOPK"
V29_AUTO_RBC_UTILITY_CONTRACT = "PROJECT7_V29_AUTO_RBC_SHADOW_HYDRAULIC_UTILITY_TOPK_V1"


@dataclass(frozen=True)
class V29AutoRBCUtilityProposal:
    target: torch.Tensor
    source: str
    raw_changed_facility_count: int
    retained_changed_facility_count: int
    total_hydraulic_utility: float
    retained_hydraulic_utility: float
    retained_utility_fraction: float
    mean_upstream_fill: float
    max_downstream_fill: float


def build_auto_rbc_hydraulic_utility_proposal(
    *,
    graph: Any,
    current_state: torch.Tensor,
    active_target: torch.Tensor,
    supervisory_mask: np.ndarray | torch.Tensor,
    first_radius: np.ndarray | torch.Tensor,
    max_changed_facilities: int,
    max_delta_per_update: float,
    low_fill: float = 0.25,
    high_fill: float = 0.75,
    downstream_congestion_fill: float = 0.90,
    response: float = 0.60,
) -> V29AutoRBCUtilityProposal:
    """Return one sparse Auto-RBC candidate ranked by local hydraulic utility."""
    state = current_state
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 2 or int(state.shape[0]) != len(graph.node_ids):
        raise ValueError("V29 Auto-RBC utility shadow requires current_state [node,feature]")
    if tuple(active_target.shape) != (109,):
        raise ValueError("V29 Auto-RBC utility shadow active_target must be [109]")
    mask = _validated_supervisory_mask(
        np.asarray(torch.as_tensor(supervisory_mask, dtype=torch.bool).cpu(), dtype=bool)
    )
    radius = np.asarray(torch.as_tensor(first_radius).detach().cpu(), dtype=np.float64).reshape(-1)
    if radius.shape != (109,):
        raise ValueError("V29 Auto-RBC utility shadow first_radius must be [109]")
    if not 1 <= int(max_changed_facilities) <= int(mask.sum()):
        raise ValueError("V29 Auto-RBC utility shadow changed-facility ceiling is invalid")
    if not 0.0 < float(max_delta_per_update) <= 1.0:
        raise ValueError("V29 Auto-RBC utility shadow max delta is invalid")
    if not 0.0 <= float(low_fill) < float(high_fill) <= 1.0:
        raise ValueError("V29 Auto-RBC utility shadow fill thresholds are invalid")
    if not 0.0 < float(downstream_congestion_fill) < 1.0:
        raise ValueError("V29 Auto-RBC utility shadow downstream threshold is invalid")
    if not 0.0 < float(response) <= 1.0:
        raise ValueError("V29 Auto-RBC utility shadow response is invalid")

    device = active_target.device
    dtype = active_target.dtype
    max_depth = torch.as_tensor(
        np.maximum(_node_feature(graph, "max_depth_m", 1.0), 1.0e-6),
        dtype=dtype,
        device=device,
    )
    upstream = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    downstream = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    depth = torch.clamp(state[:, 0].to(dtype=dtype, device=device), min=0.0)
    fill = torch.clamp(depth / max_depth, 0.0, 1.5)
    up = fill[upstream]
    down = fill[downstream]

    release_drive = torch.clamp(
        (up - float(low_fill)) / float(high_fill - low_fill), 0.0, 1.0
    )
    downstream_penalty = torch.clamp(
        (down - float(downstream_congestion_fill))
        / float(1.0 - downstream_congestion_fill),
        0.0,
        1.0,
    )
    desired_release = release_drive * (1.0 - downstream_penalty)

    sign = torch.as_tensor(
        graph_release_setting_signs(graph), dtype=dtype, device=device
    )
    current_release = torch.where(sign > 0.0, active_target, 1.0 - active_target)
    release_delta = float(response) * (desired_release - current_release)
    setting_delta = release_delta * sign

    allowed = np.minimum(np.maximum(radius, 0.0), float(max_delta_per_update))
    allowed = np.where(mask, allowed, 0.0)
    delta = setting_delta.detach().cpu().numpy().astype(np.float64)
    delta = np.where(mask, np.clip(delta, -allowed, allowed), 0.0)
    raw_changed = np.flatnonzero(np.abs(delta) > 1.0e-7)

    release_mismatch = np.abs(release_delta.detach().cpu().numpy().astype(np.float64))
    upstream_fill = up.detach().cpu().numpy().astype(np.float64)
    downstream_factor = (1.0 - downstream_penalty).detach().cpu().numpy().astype(np.float64)
    utility = release_mismatch * (0.25 + np.clip(upstream_fill, 0.0, 1.5)) * (
        0.25 + np.clip(downstream_factor, 0.0, 1.0)
    )
    utility = np.where(mask & (np.abs(delta) > 1.0e-7), utility, 0.0)
    total_utility = float(np.sum(utility))

    if raw_changed.size > int(max_changed_facilities):
        ranked = sorted(
            raw_changed.tolist(),
            key=lambda idx: (
                -float(utility[idx]),
                -float(release_mismatch[idx]),
                int(idx),
            ),
        )
        keep = np.zeros(109, dtype=bool)
        keep[np.asarray(ranked[: int(max_changed_facilities)], dtype=np.int64)] = True
        delta = np.where(keep, delta, 0.0)

    lower, upper = _actuator_bounds(graph)
    active = active_target.detach().cpu().numpy().astype(np.float64)
    target = np.clip(active + delta, lower, upper)
    target = active + np.clip(target - active, -allowed, allowed)
    target = np.where(mask, target, active)
    target = np.clip(target, lower, upper).astype(np.float32)

    retained = np.flatnonzero(np.abs(target.astype(np.float64) - active) > 1.0e-7)
    retained_utility = float(np.sum(utility[retained])) if retained.size else 0.0
    fraction = retained_utility / total_utility if total_utility > 1.0e-12 else 1.0
    return V29AutoRBCUtilityProposal(
        target=torch.as_tensor(target, dtype=dtype, device=device).detach(),
        source=V29_AUTO_RBC_UTILITY_SOURCE,
        raw_changed_facility_count=int(raw_changed.size),
        retained_changed_facility_count=int(retained.size),
        total_hydraulic_utility=total_utility,
        retained_hydraulic_utility=retained_utility,
        retained_utility_fraction=float(np.clip(fraction, 0.0, 1.0)),
        mean_upstream_fill=float(up.mean().detach().cpu()) if int(up.numel()) else 0.0,
        max_downstream_fill=float(down.max().detach().cpu()) if int(down.numel()) else 0.0,
    )


__all__ = [
    "V29_AUTO_RBC_UTILITY_CONTRACT",
    "V29_AUTO_RBC_UTILITY_SOURCE",
    "V29AutoRBCUtilityProposal",
    "build_auto_rbc_hydraulic_utility_proposal",
]
