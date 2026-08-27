"""Auto-RBC shadow proposal for Project7 Step3 V27.

The successful Auto-RBC baseline is used as a *candidate proposer*, never as an unconditional ACTION
rule.  The proposal reproduces its upstream-filling/downstream-congestion/type-aware release logic on
the causal reconstructed state, preserves passive channels, retains only the largest engineering-
relevant supervisory moves, and is then scored by the same learned V27 exact-return value model as
all other candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .actuator_release_semantics import release_fraction_to_setting
from .direct_tfv_policy_return_portfolio import _node_feature
from .rule_baselines import AUTO_RBC_CONTRACT


V27_AUTO_RBC_SHADOW_SOURCE = "AUTO_RBC_SHADOW_TOPK"
V27_AUTO_RBC_SHADOW_CONTRACT = "PROJECT7_V27_AUTO_RBC_CAUSAL_SHADOW_PROPOSAL_TOPK_V1"


@dataclass(frozen=True)
class AutoRBCShadowProposal:
    target: torch.Tensor
    source: str
    raw_changed_facility_count: int
    retained_changed_facility_count: int
    mean_upstream_fill: float
    max_downstream_fill: float


def _actuator_kinds(graph: Any) -> tuple[str, ...]:
    names = tuple(str(value) for value in getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(getattr(graph, "actuator_physics", ()), dtype=np.float64)
    ids = tuple(str(value) for value in getattr(graph, "actuator_ids", ()))
    if physics.ndim != 2 or physics.shape[0] != len(ids):
        raise ValueError("V27 Auto-RBC shadow graph actuator physics is not aligned")
    columns = {kind: names.index(f"is_{kind}") for kind in ("pump", "orifice", "weir", "outlet") if f"is_{kind}" in names}
    if len(columns) != 4:
        raise ValueError("V27 Auto-RBC shadow requires complete actuator type indicators")
    kinds: list[str] = []
    for row, actuator_id in enumerate(ids):
        matched = [kind for kind, column in columns.items() if physics[row, column] > 0.5]
        if len(matched) != 1:
            raise ValueError(f"actuator {actuator_id!r} has ambiguous type indicators")
        kinds.append(matched[0])
    return tuple(kinds)


def build_auto_rbc_shadow_proposal(
    *,
    graph: Any,
    current_state: torch.Tensor,
    active_target: torch.Tensor,
    supervisory_mask: np.ndarray | torch.Tensor,
    max_changed_facilities: int,
    max_delta_per_update: float,
    low_fill: float = 0.25,
    high_fill: float = 0.75,
    downstream_congestion_fill: float = 0.90,
    response: float = 0.60,
) -> AutoRBCShadowProposal:
    """Build a causal type-aware Auto-RBC proposal on the same 109-D action representation."""
    state = current_state
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 2 or int(state.shape[0]) != len(graph.node_ids):
        raise ValueError("V27 Auto-RBC shadow requires current_state [node,feature]")
    if tuple(active_target.shape) != (109,):
        raise ValueError("V27 Auto-RBC shadow active_target must be [109]")
    mask = torch.as_tensor(supervisory_mask, dtype=torch.bool, device=active_target.device).reshape(-1)
    if tuple(mask.shape) != (109,):
        raise ValueError("V27 Auto-RBC shadow supervisory mask must be [109]")
    if int(max_changed_facilities) <= 0:
        raise ValueError("V27 Auto-RBC shadow max_changed_facilities must be positive")
    if not 0.0 < float(max_delta_per_update) <= 1.0:
        raise ValueError("V27 Auto-RBC shadow max_delta_per_update must lie in (0,1]")
    if not 0.0 <= low_fill < high_fill <= 1.0:
        raise ValueError("V27 Auto-RBC shadow fill thresholds are invalid")
    if not 0.0 < downstream_congestion_fill < 1.0:
        raise ValueError("V27 Auto-RBC shadow downstream threshold is invalid")
    if not 0.0 < response <= 1.0:
        raise ValueError("V27 Auto-RBC shadow response is invalid")

    dtype = active_target.dtype
    device = active_target.device
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
    release_drive = torch.clamp((up - float(low_fill)) / float(high_fill - low_fill), 0.0, 1.0)
    downstream_penalty = torch.clamp(
        (down - float(downstream_congestion_fill)) / float(1.0 - downstream_congestion_fill),
        0.0,
        1.0,
    )
    release_intent = release_drive * (1.0 - downstream_penalty)
    kinds = _actuator_kinds(graph)
    desired = torch.as_tensor(
        [release_fraction_to_setting(kind, float(value)) for kind, value in zip(kinds, release_intent.detach().cpu(), strict=True)],
        dtype=dtype,
        device=device,
    )
    raw = active_target + float(response) * (desired - active_target)
    lower = active_target - float(max_delta_per_update)
    upper = active_target + float(max_delta_per_update)
    raw = torch.minimum(torch.maximum(raw, lower), upper).clamp(0.0, 1.0)
    raw = torch.where(mask, raw, active_target)
    raw_delta = torch.abs(raw - active_target)
    raw_changed = int(torch.count_nonzero(mask & (raw_delta > 1.0e-7)).item())

    target = active_target.clone()
    supervisory_indices = torch.nonzero(mask & (raw_delta > 1.0e-7)).reshape(-1)
    if int(supervisory_indices.numel()) > 0:
        keep = min(int(max_changed_facilities), int(supervisory_indices.numel()))
        local_delta = raw_delta[supervisory_indices]
        chosen_local = torch.topk(local_delta, k=keep, largest=True).indices
        chosen = supervisory_indices[chosen_local]
        target[chosen] = raw[chosen]
    retained = int(torch.count_nonzero(torch.abs(target - active_target) > 1.0e-7).item())
    return AutoRBCShadowProposal(
        target=target.detach(),
        source=V27_AUTO_RBC_SHADOW_SOURCE,
        raw_changed_facility_count=raw_changed,
        retained_changed_facility_count=retained,
        mean_upstream_fill=float(up.mean().detach().cpu()),
        max_downstream_fill=float(down.max().detach().cpu()) if int(down.numel()) else 0.0,
    )


__all__ = [
    "AUTO_RBC_CONTRACT",
    "AutoRBCShadowProposal",
    "V27_AUTO_RBC_SHADOW_CONTRACT",
    "V27_AUTO_RBC_SHADOW_SOURCE",
    "build_auto_rbc_shadow_proposal",
]
