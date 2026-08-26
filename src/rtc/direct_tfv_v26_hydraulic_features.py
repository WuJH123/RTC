"""Action-conditioned hydraulic features for Project7 Step3 V26.

V26 keeps the proven V20 facility-resolved representation and augments it with a compact set of
explicit hydraulic quantities distilled from the successful Auto-RBC logic: upstream filling,
downstream congestion/headroom, actuator release direction, flooding, rainfall loading and action
magnitude.  These quantities are *features*, not hard control rules.  The learned exact-return model
therefore remains free to contradict the rule baseline when the data support doing so.

Historical exact-return actions are reusable even when their old candidate-family label is no longer
one of the current online proposal names.  For those rows V26 keeps every state/action hydraulic
feature and simply zeros V20's three family-indicator coordinates instead of inventing a new family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .actuator_release_semantics import graph_release_setting_signs
from .direct_tfv_policy_return_facility_boundary_v20 import build_facility_boundary_parts_v20
from .direct_tfv_policy_return_portfolio import _node_feature, _rain_level
from .direct_tfv_policy_return_portfolio_admission import CURRENT_THREE_FAMILY_SOURCES


V26_HYDRAULIC_FEATURE_CONTRACT = (
    "PROJECT7_STEP3_V26_V20_PLUS_AUTO_RBC_HYDRAULIC_ACTION_CONDITIONED_EXACT_RETURN_V2"
)


@dataclass(frozen=True)
class V26HydraulicFeature:
    feature: torch.Tensor
    network_stress_q75: float
    rain_level: float
    changed_facility_count: int
    mean_upstream_pressure: float
    mean_downstream_pressure: float
    mean_release_delta: float


def _state2(current_state: torch.Tensor, graph: Any) -> torch.Tensor:
    state = current_state
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 2 or int(state.shape[0]) != len(graph.node_ids) or int(state.shape[1]) < 4:
        raise ValueError("V26 requires current_state [node,>=4]")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("V26 current_state contains non-finite values")
    return state


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(weights).clamp_min(1.0e-9)
    return torch.sum(values * weights) / denom


def _v20_base_feature(
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
) -> torch.Tensor:
    """Reuse V20 mechanics while making obsolete family names source-agnostic.

    V20 encodes the current three family names only in scalar coordinates 18:21. Historical actions
    with other labels still have valid 109-D action/state supervision, so V26 evaluates the same
    mechanics with a placeholder current family and then clears those three indicator coordinates.
    No historical label is reclassified as a current online family.
    """
    source = str(candidate_source)
    allowed = tuple(CURRENT_THREE_FAMILY_SOURCES)
    known = source in allowed
    base = build_facility_boundary_parts_v20(
        step2_model=step2_model,
        normalization=normalization,
        graph=graph,
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        candidate_target=candidate_target,
        candidate_source=source if known else allowed[0],
        supervisory_mask=supervisory_mask,
        target_scale_m3=float(target_scale_m3),
    ).feature
    if not known:
        base = base.clone()
        base[18 : 18 + len(allowed)] = 0.0
    return base


def build_v26_hydraulic_feature(
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
) -> V26HydraulicFeature:
    """Build one causal candidate feature for exact system-wide TFV return prediction."""

    state = _state2(current_state, graph)
    if tuple(active_target.shape) != (109,) or tuple(candidate_target.shape) != (109,):
        raise ValueError("V26 active/candidate target must be [109]")
    mask = torch.as_tensor(supervisory_mask, dtype=torch.bool, device=active_target.device).reshape(-1)
    if tuple(mask.shape) != (109,) or int(mask.sum()) != 82:
        raise ValueError("V26 requires the frozen 82/109 supervisory mask")
    passive = ~mask
    if bool(torch.any(torch.abs(candidate_target[passive] - active_target[passive]) > 1.0e-7)):
        raise ValueError("V26 candidate changed a passive/reference-only channel")

    base = _v20_base_feature(
        step2_model=step2_model,
        normalization=normalization,
        graph=graph,
        current_state=state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow.reshape(-1),
        active_target=active_target,
        candidate_target=candidate_target,
        candidate_source=str(candidate_source),
        supervisory_mask=mask,
        target_scale_m3=float(target_scale_m3),
    )

    dtype = state.dtype
    device = state.device
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
    flooding = torch.clamp(state[:, 2], min=0.0)
    pressure = 0.65 * depth_fill + 0.35 * volume_fill
    up_pressure = pressure[upstream]
    down_pressure = pressure[downstream]
    up_depth = depth_fill[upstream]
    down_depth = depth_fill[downstream]
    up_volume = volume_fill[upstream]
    down_volume = volume_fill[downstream]
    up_flood = flooding[upstream]
    down_flood = flooding[downstream]

    delta = candidate_target - active_target
    abs_delta = torch.abs(delta)
    weights = torch.where(mask, abs_delta, torch.zeros_like(abs_delta))
    action_mass = torch.sum(weights)
    changed = int(torch.count_nonzero(weights > 1.0e-7).item())
    if float(action_mass.detach().cpu()) <= 1.0e-9:
        hydraulic = torch.zeros(24, dtype=dtype, device=device)
        feature = torch.cat((base, hydraulic), dim=0)
        return V26HydraulicFeature(
            feature=feature,
            network_stress_q75=0.0,
            rain_level=float(_rain_level(rainfall_scenarios.detach().cpu().numpy())),
            changed_facility_count=0,
            mean_upstream_pressure=0.0,
            mean_downstream_pressure=0.0,
            mean_release_delta=0.0,
        )

    release_sign = torch.as_tensor(graph_release_setting_signs(graph), dtype=dtype, device=device)
    release_delta = delta * release_sign
    masked_weights = weights
    mean_up = _weighted_mean(up_pressure, masked_weights)
    mean_down = _weighted_mean(down_pressure, masked_weights)
    mean_up_depth = _weighted_mean(up_depth, masked_weights)
    mean_down_depth = _weighted_mean(down_depth, masked_weights)
    mean_up_volume = _weighted_mean(up_volume, masked_weights)
    mean_down_volume = _weighted_mean(down_volume, masked_weights)
    mean_up_flood = _weighted_mean(torch.tanh(up_flood / 0.01), masked_weights)
    mean_down_flood = _weighted_mean(torch.tanh(down_flood / 0.01), masked_weights)
    mean_release = _weighted_mean(release_delta, masked_weights)
    mean_abs_release = _weighted_mean(torch.abs(release_delta), masked_weights)
    release_more = _weighted_mean((release_delta > 1.0e-7).to(dtype), masked_weights)
    release_less = _weighted_mean((release_delta < -1.0e-7).to(dtype), masked_weights)
    pressure_gradient = mean_up - mean_down
    release_alignment = _weighted_mean(release_delta * (up_pressure - down_pressure), masked_weights)
    downstream_headroom = _weighted_mean(torch.clamp(1.0 - down_pressure, min=-0.5, max=1.0), masked_weights)
    upstream_headroom = _weighted_mean(torch.clamp(1.0 - up_pressure, min=-0.5, max=1.0), masked_weights)
    stress_values = up_pressure[mask]
    stress = torch.quantile(stress_values, 0.75) if int(stress_values.numel()) else state.new_zeros(())
    max_up = torch.max(up_pressure[mask]) if int(mask.sum()) else state.new_zeros(())
    max_down = torch.max(down_pressure[mask]) if int(mask.sum()) else state.new_zeros(())
    rain_level = float(_rain_level(rainfall_scenarios.detach().cpu().numpy()))
    rain_scaled = state.new_tensor(rain_level / 20.0)
    changed_fraction = state.new_tensor(float(changed) / 82.0)
    rms_delta = torch.sqrt(torch.mean(torch.square(delta[mask])).clamp_min(0.0))
    mean_abs_delta = torch.mean(abs_delta[mask])

    hydraulic = torch.stack(
        (
            mean_up,
            mean_down,
            pressure_gradient,
            mean_up_depth,
            mean_down_depth,
            mean_up_volume,
            mean_down_volume,
            mean_up_flood,
            mean_down_flood,
            mean_release,
            mean_abs_release,
            release_more,
            release_less,
            release_alignment,
            downstream_headroom,
            upstream_headroom,
            stress,
            max_up,
            max_down,
            rain_scaled,
            changed_fraction,
            rms_delta,
            mean_abs_delta,
            mean_release * downstream_headroom * mean_up,
        )
    )
    feature = torch.cat((base, hydraulic), dim=0)
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("V26 hydraulic feature contains non-finite values")
    return V26HydraulicFeature(
        feature=feature,
        network_stress_q75=float(stress.detach().cpu()),
        rain_level=rain_level,
        changed_facility_count=changed,
        mean_upstream_pressure=float(mean_up.detach().cpu()),
        mean_downstream_pressure=float(mean_down.detach().cpu()),
        mean_release_delta=float(mean_release.detach().cpu()),
    )


__all__ = [
    "V26_HYDRAULIC_FEATURE_CONTRACT",
    "V26HydraulicFeature",
    "build_v26_hydraulic_feature",
]
