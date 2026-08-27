"""Local hydraulic action representation for Project7 Step3 V27.

V26 added physically meaningful aggregate Auto-RBC-style signals.  V27 keeps them and preserves more
of the spatial decision structure by appending top-k actuator-local pressure/headroom/action terms and
per-actuator-type summaries.  This remains a compact deterministic representation; no hard hydraulic
threshold authorizes ACTION.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .actuator_release_semantics import graph_release_setting_signs
from .direct_tfv_policy_return_portfolio import _node_feature
from .direct_tfv_v26_hydraulic_features import V26HydraulicFeature, build_v26_hydraulic_feature


V27_HYDRAULIC_FEATURE_CONTRACT = (
    "PROJECT7_STEP3_V27_V26_PLUS_LOCAL_TOPK_TYPE_AWARE_HYDRAULIC_ACTION_FEATURES_V1"
)
V27_LOCAL_FEATURE_WIDTH = 41


@dataclass(frozen=True)
class V27HydraulicFeature:
    feature: torch.Tensor
    base: V26HydraulicFeature
    local_feature_width: int
    top_upstream_pressure: tuple[float, ...]
    top_downstream_congestion: tuple[float, ...]
    top_action_magnitude: tuple[float, ...]


def _state2(current_state: torch.Tensor, graph: Any) -> torch.Tensor:
    state = current_state
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 2 or int(state.shape[0]) != len(graph.node_ids) or int(state.shape[1]) < 4:
        raise ValueError("V27 requires current_state [node,>=4]")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("V27 current_state contains non-finite values")
    return state


def _topk(values: torch.Tensor, *, k: int = 3, largest: bool = True) -> torch.Tensor:
    flat = values.reshape(-1)
    if int(flat.numel()) == 0:
        return torch.zeros(k, dtype=values.dtype, device=values.device)
    take = min(k, int(flat.numel()))
    selected = torch.topk(flat, k=take, largest=largest).values
    if take < k:
        selected = torch.cat((selected, torch.zeros(k - take, dtype=values.dtype, device=values.device)))
    return selected


def _actuator_kind_masks(graph: Any, *, device: torch.device) -> dict[str, torch.Tensor]:
    names = tuple(str(value) for value in getattr(graph, "actuator_physics_feature_names", ()))
    physics = torch.as_tensor(np.asarray(graph.actuator_physics), dtype=torch.float32, device=device)
    result: dict[str, torch.Tensor] = {}
    for kind in ("pump", "orifice", "weir", "outlet"):
        name = f"is_{kind}"
        if name not in names:
            raise ValueError(f"V27 graph lacks actuator physics column {name}")
        result[kind] = physics[:, names.index(name)] > 0.5
    if any(int(mask.sum()) == 0 for mask in result.values()):
        # Zero-count kinds are allowed in principle, but Wuhan's frozen graph is expected to expose
        # all four semantics.  Do not use this as a performance gate; it is only a schema check.
        pass
    combined = sum((mask.to(torch.int64) for mask in result.values()), torch.zeros(len(graph.actuator_ids), dtype=torch.int64, device=device))
    if bool(torch.any(combined != 1)):
        raise ValueError("V27 actuator kind indicators are not one-hot")
    return result


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(weights)
    if float(denom.detach().cpu()) <= 1.0e-12:
        return values.new_zeros(())
    return torch.sum(values * weights) / denom


def build_v27_hydraulic_feature(
    *,
    step2_model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    candidate_target: torch.Tensor,
    candidate_source: str,
    supervisory_mask: np.ndarray | torch.Tensor,
    target_scale_m3: float,
) -> V27HydraulicFeature:
    base = build_v26_hydraulic_feature(
        step2_model=step2_model,
        normalization=normalization,
        graph=graph,
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        candidate_target=candidate_target,
        candidate_source=candidate_source,
        supervisory_mask=supervisory_mask,
        target_scale_m3=float(target_scale_m3),
    )
    state = _state2(current_state, graph)
    dtype = state.dtype
    device = state.device
    mask = torch.as_tensor(supervisory_mask, dtype=torch.bool, device=device).reshape(-1)
    if tuple(mask.shape) != (109,):
        raise ValueError("V27 supervisory mask must be [109]")
    upstream = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    downstream = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    max_depth = torch.as_tensor(
        np.maximum(_node_feature(graph, "max_depth_m", 1.0), 1.0e-6),
        dtype=dtype,
        device=device,
    )
    storage_capacity = torch.as_tensor(
        np.maximum(_node_feature(graph, "storage_capacity_m3", 1.0), 1.0e-6),
        dtype=dtype,
        device=device,
    )
    depth_fill = torch.clamp(torch.clamp(state[:, 0], min=0.0) / max_depth, 0.0, 1.5)
    volume_fill = torch.clamp(torch.clamp(state[:, 3], min=0.0) / storage_capacity, 0.0, 1.5)
    pressure = 0.65 * depth_fill + 0.35 * volume_fill
    up_pressure = pressure[upstream]
    down_pressure = pressure[downstream]
    headroom = torch.clamp(1.0 - down_pressure, min=-0.5, max=1.0)
    gradient = up_pressure - down_pressure
    delta = candidate_target - active_target
    abs_delta = torch.abs(delta)
    changed = mask & (abs_delta > 1.0e-7)
    release_sign = torch.as_tensor(graph_release_setting_signs(graph), dtype=dtype, device=device)
    release_delta = delta * release_sign
    alignment = release_delta * gradient

    if int(changed.sum()) == 0:
        local = torch.zeros(V27_LOCAL_FEATURE_WIDTH, dtype=dtype, device=device)
        feature = torch.cat((base.feature, local), dim=0)
        return V27HydraulicFeature(
            feature=feature,
            base=base,
            local_feature_width=V27_LOCAL_FEATURE_WIDTH,
            top_upstream_pressure=(0.0, 0.0, 0.0),
            top_downstream_congestion=(0.0, 0.0, 0.0),
            top_action_magnitude=(0.0, 0.0, 0.0),
        )

    local_parts = [
        _topk(up_pressure[changed]),
        _topk(down_pressure[changed]),
        _topk(gradient[changed]),
        _topk(headroom[changed]),
        _topk(abs_delta[changed]),
        _topk(alignment[changed], largest=True),
        _topk(alignment[changed], largest=False),
    ]
    # 21 top-k coordinates above.  Add five action-conditioned summaries for each SWMM actuator kind.
    kind_masks = _actuator_kind_masks(graph, device=device)
    changed_count = float(int(changed.sum()))
    for kind in ("pump", "orifice", "weir", "outlet"):
        local_mask = changed & kind_masks[kind]
        weights = torch.where(local_mask, abs_delta, torch.zeros_like(abs_delta))
        local_parts.append(
            torch.stack(
                (
                    state.new_tensor(float(int(local_mask.sum())) / max(1.0, changed_count)),
                    _weighted_mean(release_delta, weights),
                    _weighted_mean(up_pressure, weights),
                    _weighted_mean(down_pressure, weights),
                    _weighted_mean(headroom, weights),
                )
            )
        )
    local = torch.cat(local_parts, dim=0)
    if int(local.numel()) != V27_LOCAL_FEATURE_WIDTH:
        raise RuntimeError(f"V27 local feature width drifted: {int(local.numel())}")
    feature = torch.cat((base.feature, local), dim=0)
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("V27 feature contains non-finite values")
    return V27HydraulicFeature(
        feature=feature,
        base=base,
        local_feature_width=V27_LOCAL_FEATURE_WIDTH,
        top_upstream_pressure=tuple(float(v) for v in _topk(up_pressure[changed]).detach().cpu()),
        top_downstream_congestion=tuple(float(v) for v in _topk(down_pressure[changed]).detach().cpu()),
        top_action_magnitude=tuple(float(v) for v in _topk(abs_delta[changed]).detach().cpu()),
    )


__all__ = [
    "V27_HYDRAULIC_FEATURE_CONTRACT",
    "V27_LOCAL_FEATURE_WIDTH",
    "V27HydraulicFeature",
    "build_v27_hydraulic_feature",
]
