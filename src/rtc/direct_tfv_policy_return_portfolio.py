"""Causal candidate portfolio for policy-return-aligned Direct-TFV control.

The historical policy-return layer only judged the single target supplied by the V12 open-loop
direction generator.  That can reject a bad action but cannot discover a better executable action
when the V12 direction is wrong.  This module constructs a small, deterministic portfolio that
preserves the Proposed method's information budget:

* several magnitudes of the learned V12 target;
* a topology/volume/headroom-aware hydraulic-pressure target; and
* one blend of the learned and hydraulic directions.

No baseline action, future realised rainfall, online SWMM call, PFV/peak objective or action penalty
is used.  Every target is clipped to the frozen per-facility first-move support, the engineering
0.5 target slew, actuator bounds and the caller-supplied q95 changed-facility ceiling.  Final joint
sequence support remains the responsibility of the Step3 controller, which has the authoritative
D3-HOLD geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from .actuator_release_semantics import graph_release_setting_signs


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_CAUSAL_CANDIDATE_PORTFOLIO_V1"
)
DEFAULT_V12_SHRINK_SCALES = (0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class PolicyReturnPortfolioCandidate:
    source: str
    target: torch.Tensor
    changed_facility_count: int


def _node_feature(graph: Any, name: str, default: float) -> np.ndarray:
    names = tuple(str(value) for value in getattr(graph, "static_node_feature_names", ()))
    features = np.asarray(getattr(graph, "static_node_features", ()), dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(graph.node_ids):
        raise ValueError("portfolio graph node features are misaligned")
    if name not in names:
        return np.full(len(graph.node_ids), float(default), dtype=np.float64)
    return features[:, names.index(name)]


def _actuator_bounds(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(str(value) for value in getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(getattr(graph, "actuator_physics", ()), dtype=np.float64)
    if physics.ndim != 2 or physics.shape[0] != 109:
        raise ValueError("portfolio requires aligned 109-actuator physics")
    if "min_setting" not in names or "max_setting" not in names:
        raise ValueError("portfolio requires explicit actuator setting bounds")
    lower = physics[:, names.index("min_setting")]
    upper = physics[:, names.index("max_setting")]
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("portfolio actuator setting bounds are invalid")
    return lower, upper


def _rain_level(rainfall_scenarios: np.ndarray) -> float:
    rain = np.asarray(rainfall_scenarios, dtype=np.float64)
    if rain.ndim == 4:  # [scenario,H,node,feature]
        pass
    elif rain.ndim == 3:  # [H,node,feature]
        rain = rain[None]
    else:
        raise ValueError("portfolio rainfall must be [scenario,H,node,feature] or [H,node,feature]")
    if not np.isfinite(rain).all():
        raise ValueError("portfolio rainfall contains non-finite values")
    return float(np.max(np.maximum(rain, 0.0), initial=0.0))


def hydraulic_pressure_setting_delta(
    *,
    current_state: np.ndarray,
    rainfall_scenarios: np.ndarray,
    graph: Any,
    max_delta_per_update: float = 0.5,
) -> np.ndarray:
    """Return a type-aware local hydraulic release direction from causal state only.

    State columns follow the frozen Project7 six-state contract used by the existing knowledge seed:
    depth is column 0, flooding rate/state is column 2 and storage volume is column 3.  Storage
    capacity and maximum depth are static graph features, so the pressure signal represents current
    hydraulic headroom rather than a generic 0..1 actuator preference.
    """

    state = np.asarray(current_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] != len(graph.node_ids) or state.shape[1] < 4:
        raise ValueError("portfolio hydraulic state must be [node,>=4]")
    if not np.isfinite(state).all():
        raise ValueError("portfolio hydraulic state contains non-finite values")
    upstream = np.asarray(graph.actuator_upstream, dtype=np.int64)
    downstream = np.asarray(graph.actuator_downstream, dtype=np.int64)
    if upstream.shape != (109,) or downstream.shape != (109,):
        raise ValueError("portfolio requires 109 actuator endpoints")

    max_depth = np.maximum(_node_feature(graph, "max_depth_m", 1.0), 1.0e-6)
    storage_capacity = np.maximum(_node_feature(graph, "storage_capacity_m3", 1.0), 1.0e-6)
    depth_fill = np.clip(np.maximum(state[:, 0], 0.0) / max_depth, 0.0, 1.5)
    volume_fill = np.clip(np.maximum(state[:, 3], 0.0) / storage_capacity, 0.0, 1.5)
    flooding = np.maximum(state[:, 2], 0.0)

    up_pressure = 0.65 * depth_fill[upstream] + 0.35 * volume_fill[upstream]
    down_pressure = 0.65 * depth_fill[downstream] + 0.35 * volume_fill[downstream]
    downstream_flood_penalty = 0.5 * np.tanh(flooding[downstream] / 0.01)
    rain_level = _rain_level(rainfall_scenarios)
    rain_gain = 1.0 + 0.10 * np.tanh(rain_level / 20.0)
    release_intent = np.tanh((up_pressure - down_pressure - downstream_flood_penalty) * rain_gain)

    setting_sign = graph_release_setting_signs(graph)
    delta = float(max_delta_per_update) * release_intent * setting_sign
    if not np.isfinite(delta).all():
        raise ValueError("portfolio hydraulic pressure produced non-finite target delta")
    return np.clip(delta, -float(max_delta_per_update), float(max_delta_per_update))


def _bounded_supported_target(
    *,
    active_target: np.ndarray,
    raw_delta: np.ndarray,
    graph: Any,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    max_delta_per_update: float,
) -> np.ndarray:
    active = np.asarray(active_target, dtype=np.float64).reshape(-1)
    delta = np.asarray(raw_delta, dtype=np.float64).reshape(-1)
    radius = np.asarray(first_radius, dtype=np.float64).reshape(-1)
    if active.shape != (109,) or delta.shape != (109,) or radius.shape != (109,):
        raise ValueError("portfolio target/support vectors must contain 109 actuators")
    if not 1 <= int(max_changed_facilities) <= 109:
        raise ValueError("portfolio changed-facility ceiling must lie in [1,109]")
    allowed = np.minimum(np.maximum(radius, 0.0), float(max_delta_per_update))
    delta = np.clip(delta, -allowed, allowed)

    nonzero = np.flatnonzero(np.abs(delta) > 1.0e-7)
    if nonzero.size > int(max_changed_facilities):
        order = nonzero[np.argsort(-np.abs(delta[nonzero]), kind="mergesort")]
        keep = order[: int(max_changed_facilities)]
        mask = np.zeros(109, dtype=bool)
        mask[keep] = True
        delta = np.where(mask, delta, 0.0)

    lower, upper = _actuator_bounds(graph)
    target = np.clip(active + delta, lower, upper)
    target = active + np.clip(target - active, -allowed, allowed)
    target = np.clip(target, lower, upper)
    return target.astype(np.float32)


def build_policy_return_candidate_portfolio(
    *,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    active_target: torch.Tensor,
    v12_target: torch.Tensor | None,
    graph: Any,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    max_delta_per_update: float = 0.5,
    shrink_scales: Iterable[float] = DEFAULT_V12_SHRINK_SCALES,
) -> tuple[PolicyReturnPortfolioCandidate, ...]:
    """Build deterministic supported first-target alternatives for the policy-return critic."""

    active = active_target.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if active.shape != (109,):
        raise ValueError("portfolio requires a 109-dimensional active target")
    state = current_state.detach().cpu().numpy()
    if state.ndim == 3 and state.shape[0] == 1:
        state = state[0]
    rain = rainfall_scenarios.detach().cpu().numpy()

    raw: list[tuple[str, np.ndarray]] = []
    learned_delta: np.ndarray | None = None
    if isinstance(v12_target, torch.Tensor):
        learned = v12_target.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if learned.shape != (109,):
            raise ValueError("portfolio V12 target must contain 109 settings")
        learned_delta = learned - active
        for value in shrink_scales:
            scale = float(value)
            if not 0.0 < scale <= 1.0:
                raise ValueError("portfolio shrink scales must lie in (0,1]")
            raw.append((f"V12_DIRECTION_SCALE_{scale:.2f}", scale * learned_delta))

    pressure_delta = hydraulic_pressure_setting_delta(
        current_state=state,
        rainfall_scenarios=rain,
        graph=graph,
        max_delta_per_update=max_delta_per_update,
    )
    raw.append(("TYPE_AWARE_HYDRAULIC_PRESSURE", pressure_delta))
    if learned_delta is not None:
        raw.append(("V12_HYDRAULIC_BLEND_50", 0.5 * learned_delta + 0.5 * pressure_delta))

    candidates: list[PolicyReturnPortfolioCandidate] = []
    seen: set[bytes] = set()
    for source, delta in raw:
        target_np = _bounded_supported_target(
            active_target=active,
            raw_delta=delta,
            graph=graph,
            first_radius=first_radius,
            max_changed_facilities=max_changed_facilities,
            max_delta_per_update=max_delta_per_update,
        )
        changed = int(np.count_nonzero(np.abs(target_np.astype(np.float64) - active) > 1.0e-7))
        if changed <= 0:
            continue
        key = np.ascontiguousarray(target_np, dtype=np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            PolicyReturnPortfolioCandidate(
                source=source,
                target=torch.as_tensor(target_np, dtype=active_target.dtype, device=active_target.device),
                changed_facility_count=changed,
            )
        )
    return tuple(candidates)


__all__ = [
    "DEFAULT_V12_SHRINK_SCALES",
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT",
    "PolicyReturnPortfolioCandidate",
    "build_policy_return_candidate_portfolio",
    "hydraulic_pressure_setting_delta",
]
