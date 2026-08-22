"""Development-only stress-adaptive refinement of the existing hydraulic candidate family.

The current three-family portfolio is preserved.  Only the target generated for the existing
``TYPE_AWARE_HYDRAULIC_PRESSURE`` family is refined when the causal hydraulic state indicates
widespread high filling.  Low/moderate stress stays close to the frozen V21 generator; strong-storm
stress smoothly shifts toward an absolute upstream-filling release target with downstream-congestion
protection and actuator-type-aware SWMM SETTING semantics.

This module deliberately does not add a fourth family, alter q95 support, change the 0.5 target slew,
use future rainfall/SWMM truth, or enable gradient/L-BFGS-B search.  Because the candidate target
distribution differs from the V21 training generator, this lane is Development-only and must never be
presented as matched Policy-Lock evidence without fresh matched candidate truth/calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .actuator_release_semantics import graph_release_setting_signs
from .direct_tfv_policy_return_hybrid_portfolio import (
    HybridPolicyReturnPortfolio,
    build_hybrid_policy_return_portfolio,
)
from .direct_tfv_policy_return_portfolio import (
    PolicyReturnPortfolioCandidate,
    _bounded_supported_target,
    _node_feature,
    _rain_level,
    _validated_supervisory_mask,
    hydraulic_pressure_setting_delta,
)


V23_HYDRAULIC_CANDIDATE_CONTRACT = (
    "PROJECT7_OPERATIONAL_V23_STRESS_ADAPTIVE_ABSOLUTE_FILL_RELEASE_TARGET_V1"
)
V23_PORTFOLIO_CONTRACT = (
    "PROJECT7_OPERATIONAL_V23_THREE_FAMILY_STRONG_STORM_HYDRAULIC_REFINEMENT_V1"
)
HYDRAULIC_SOURCE = "TYPE_AWARE_HYDRAULIC_PRESSURE"


@dataclass(frozen=True)
class StrongStormHydraulicDiagnostics:
    network_stress_q75: float
    strong_storm_blend: float
    rain_level: float
    mean_release_fraction: float
    maximum_release_fraction: float
    mean_abs_legacy_delta: float
    mean_abs_strong_delta: float
    changed_facility_count: int
    contract: str = V23_HYDRAULIC_CANDIDATE_CONTRACT


@dataclass(frozen=True)
class OperationalV23Portfolio:
    candidates: tuple[PolicyReturnPortfolioCandidate, ...]
    learned_probe: Any
    projected_gradient: Any
    hydraulic_diagnostics: StrongStormHydraulicDiagnostics
    portfolio_contract: str = V23_PORTFOLIO_CONTRACT
    projected_gradient_online: bool = False


def _state2(current_state: torch.Tensor, graph: Any) -> np.ndarray:
    state = current_state.detach().cpu().numpy().astype(np.float64)
    if state.ndim == 3 and state.shape[0] == 1:
        state = state[0]
    if state.ndim != 2 or state.shape[0] != len(graph.node_ids) or state.shape[1] < 4:
        raise ValueError("V23 hydraulic refinement requires current_state [node,>=4]")
    if not np.isfinite(state).all():
        raise ValueError("V23 hydraulic refinement received non-finite state")
    return state


def stress_adaptive_hydraulic_target_v23(
    *,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    active_target: torch.Tensor,
    graph: Any,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    max_delta_per_update: float = 0.5,
    supervisory_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, StrongStormHydraulicDiagnostics]:
    """Return one supported target for the existing hydraulic family.

    The legacy pressure-difference heuristic remains the low-stress anchor.  A smooth stress gate
    increases the contribution of an absolute upstream-filling release target only when the network
    is broadly loaded.  This addresses the strong-storm failure mode where both ends of an actuator
    can be highly filled and their *difference* is therefore misleadingly small.
    """
    state = _state2(current_state, graph)
    rain = rainfall_scenarios.detach().cpu().numpy().astype(np.float64)
    active = active_target.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if active.shape != (109,):
        raise ValueError("V23 hydraulic refinement requires active_target [109]")
    mask = _validated_supervisory_mask(supervisory_mask)
    upstream = np.asarray(graph.actuator_upstream, dtype=np.int64)
    downstream = np.asarray(graph.actuator_downstream, dtype=np.int64)
    if upstream.shape != (109,) or downstream.shape != (109,):
        raise ValueError("V23 hydraulic refinement requires 109 actuator endpoints")

    max_depth = np.maximum(_node_feature(graph, "max_depth_m", 1.0), 1.0e-6)
    storage_capacity = np.maximum(_node_feature(graph, "storage_capacity_m3", 1.0), 1.0e-6)
    depth_fill = np.clip(np.maximum(state[:, 0], 0.0) / max_depth, 0.0, 1.5)
    volume_fill = np.clip(np.maximum(state[:, 3], 0.0) / storage_capacity, 0.0, 1.5)
    flooding = np.maximum(state[:, 2], 0.0)
    node_pressure = 0.65 * depth_fill + 0.35 * volume_fill
    up_pressure = node_pressure[upstream]
    down_pressure = node_pressure[downstream]

    eligible_up = up_pressure[mask]
    network_stress = float(np.quantile(eligible_up, 0.75)) if eligible_up.size else 0.0
    # Smoothly preserve legacy behaviour below broadly-high filling, while avoiding a hard mode switch.
    blend = float(np.clip((network_stress - 0.65) / 0.25, 0.0, 1.0))

    legacy_delta = hydraulic_pressure_setting_delta(
        current_state=state,
        rainfall_scenarios=rain,
        graph=graph,
        max_delta_per_update=max_delta_per_update,
    )

    # Absolute upstream filling is the release driver.  Downstream pressure/flooding suppresses,
    # rather than reverses, the release target.  This retains local hydraulic protection without
    # collapsing the candidate merely because both sides are highly loaded in an extreme event.
    release_drive = np.clip((up_pressure - 0.20) / 0.55, 0.0, 1.0)
    downstream_congestion = np.clip((down_pressure - 0.90) / 0.25, 0.0, 1.0)
    downstream_flood = np.tanh(flooding[downstream] / 0.01)
    downstream_penalty = np.maximum(downstream_congestion, downstream_flood)
    rain_level = _rain_level(rain)
    rain_gain = 1.0 + 0.15 * np.tanh(rain_level / 20.0)
    release_fraction = np.clip(
        release_drive * rain_gain * (1.0 - 0.85 * downstream_penalty),
        0.0,
        1.0,
    )

    setting_sign = graph_release_setting_signs(graph)
    desired_setting = np.where(setting_sign > 0.0, release_fraction, 1.0 - release_fraction)
    strong_delta = 0.60 * (desired_setting - active)
    raw_delta = (1.0 - blend) * legacy_delta + blend * strong_delta
    raw_delta = np.where(mask, raw_delta, 0.0)

    target_np = _bounded_supported_target(
        active_target=active,
        raw_delta=raw_delta,
        graph=graph,
        first_radius=first_radius,
        max_changed_facilities=int(max_changed_facilities),
        max_delta_per_update=float(max_delta_per_update),
        supervisory_mask=mask,
    )
    changed = int(np.count_nonzero(np.abs(target_np.astype(np.float64) - active) > 1.0e-7))
    diagnostics = StrongStormHydraulicDiagnostics(
        network_stress_q75=network_stress,
        strong_storm_blend=blend,
        rain_level=float(rain_level),
        mean_release_fraction=float(np.mean(release_fraction[mask])) if np.any(mask) else 0.0,
        maximum_release_fraction=float(np.max(release_fraction[mask], initial=0.0)),
        mean_abs_legacy_delta=float(np.mean(np.abs(legacy_delta[mask]))) if np.any(mask) else 0.0,
        mean_abs_strong_delta=float(np.mean(np.abs(strong_delta[mask]))) if np.any(mask) else 0.0,
        changed_facility_count=changed,
    )
    return (
        torch.as_tensor(target_np, dtype=active_target.dtype, device=active_target.device),
        diagnostics,
    )


def build_hybrid_policy_return_portfolio_v23(
    *,
    model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    max_delta_per_update: float = 0.5,
    probe_chunk_size: int = 24,
    supervisory_mask: np.ndarray | None = None,
) -> OperationalV23Portfolio:
    """Keep the frozen three-family set and replace only the hydraulic target semantics."""
    base: HybridPolicyReturnPortfolio = build_hybrid_policy_return_portfolio(
        model=model,
        normalization=normalization,
        graph=graph,
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        first_radius=first_radius,
        max_changed_facilities=int(max_changed_facilities),
        max_delta_per_update=float(max_delta_per_update),
        probe_chunk_size=int(probe_chunk_size),
        supervisory_mask=supervisory_mask,
        include_projected_gradient_ablation=False,
    )
    hydraulic_target, diagnostics = stress_adaptive_hydraulic_target_v23(
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        active_target=active_target,
        graph=graph,
        first_radius=first_radius,
        max_changed_facilities=int(max_changed_facilities),
        max_delta_per_update=float(max_delta_per_update),
        supervisory_mask=supervisory_mask,
    )

    active = active_target.detach().cpu().to(torch.float32).contiguous().numpy()
    replacement_changed = int(
        torch.count_nonzero(torch.abs(hydraulic_target - active_target) > 1.0e-7).item()
    )
    rows: list[PolicyReturnPortfolioCandidate] = []
    hydraulic_seen = False
    for row in base.candidates:
        if row.source == HYDRAULIC_SOURCE:
            hydraulic_seen = True
            if replacement_changed > 0:
                rows.append(
                    PolicyReturnPortfolioCandidate(
                        source=HYDRAULIC_SOURCE,
                        target=hydraulic_target.detach(),
                        changed_facility_count=replacement_changed,
                    )
                )
        else:
            rows.append(row)
    if not hydraulic_seen and replacement_changed > 0:
        rows.append(
            PolicyReturnPortfolioCandidate(
                source=HYDRAULIC_SOURCE,
                target=hydraulic_target.detach(),
                changed_facility_count=replacement_changed,
            )
        )

    deduped: list[PolicyReturnPortfolioCandidate] = []
    seen: set[bytes] = set()
    active_bytes = np.ascontiguousarray(active, dtype=np.float32).tobytes()
    for row in rows:
        key = row.target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
        if key == active_bytes or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return OperationalV23Portfolio(
        candidates=tuple(deduped[:3]),
        learned_probe=base.learned_probe,
        projected_gradient=base.projected_gradient,
        hydraulic_diagnostics=diagnostics,
        projected_gradient_online=False,
    )


__all__ = [
    "HYDRAULIC_SOURCE",
    "OperationalV23Portfolio",
    "StrongStormHydraulicDiagnostics",
    "V23_HYDRAULIC_CANDIDATE_CONTRACT",
    "V23_PORTFOLIO_CONTRACT",
    "build_hybrid_policy_return_portfolio_v23",
    "stress_adaptive_hydraulic_target_v23",
]
