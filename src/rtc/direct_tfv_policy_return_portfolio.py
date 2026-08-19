"""Causal first-action proposal portfolio for practical Project7 RTC.

The pretrained Step2 keeps its frozen 109-channel hydraulic action representation. Current online
proposal generation may additionally receive a native supervisory mask; channels outside that mask
remain present in the model but are forced to candidate == reference. This lets Project7 reduce the
actual control dimension without retraining Step1 or the base Step2.

The proposal layer uses the frozen base Step2 as a directional probe model: for every *eligible*
actuator it batch-scores positive/negative H10 pulses followed by HOLD for H350, selects individually
beneficial directions, and combines at most the TrainFit support ceiling. The learned direction is
offered at full and half magnitude alongside one hydraulically interpretable, actuator-type-aware
pressure direction. The policy-return critic, not the base Step2 score, ranks/admits final execution.

No future realised rainfall, online SWMM, PFV/peak objective, rule baseline warm start, or action
penalty is used. All targets stay inside first-move support, the 0.5 target slew and actuator bounds
before the runtime applies its H10 joint-sequence trust region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from .actuator_release_semantics import graph_release_setting_signs
from .direct_tfv_policy_return import encode_policy_return_action_token


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_CAUSAL_CANDIDATE_PORTFOLIO_V3_H10_PROBE_THREE_FAMILY"
)
DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT = (
    "PROJECT7_DIRECT_TFV_SUPPORTED_H10_PROBE_DIRECTION_GENERATOR_V1"
)
DEFAULT_LEARNED_SHRINK_SCALES = (0.50, 1.00)
DEFAULT_V12_SHRINK_SCALES = DEFAULT_LEARNED_SHRINK_SCALES


@dataclass(frozen=True)
class PolicyReturnPortfolioCandidate:
    source: str
    target: torch.Tensor
    changed_facility_count: int


@dataclass(frozen=True)
class LearnedH10ProbeProposal:
    target: torch.Tensor | None
    facility_best_scores_m3: tuple[float, ...]
    predicted_beneficial_facility_count: int
    selected_facility_indices: tuple[int, ...]
    probe_count: int
    generator_contract: str = DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT


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


def _validated_supervisory_mask(supervisory_mask: np.ndarray | None) -> np.ndarray:
    if supervisory_mask is None:
        return np.ones(109, dtype=bool)
    mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
    if mask.shape != (109,) or int(mask.sum()) <= 0:
        raise ValueError("portfolio supervisory mask must contain 109 entries with >=1 enabled")
    return mask


def _rain_level(rainfall_scenarios: np.ndarray) -> float:
    rain = np.asarray(rainfall_scenarios, dtype=np.float64)
    if rain.ndim == 3:
        rain = rain[None]
    if rain.ndim != 4:
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
    """Return a type-aware local hydraulic release direction from causal state only."""
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
    supervisory_mask: np.ndarray | None = None,
) -> np.ndarray:
    active = np.asarray(active_target, dtype=np.float64).reshape(-1)
    delta = np.asarray(raw_delta, dtype=np.float64).reshape(-1)
    radius = np.asarray(first_radius, dtype=np.float64).reshape(-1)
    if active.shape != (109,) or delta.shape != (109,) or radius.shape != (109,):
        raise ValueError("portfolio target/support vectors must contain 109 actuators")
    mask = _validated_supervisory_mask(supervisory_mask)
    if not 1 <= int(max_changed_facilities) <= int(mask.sum()):
        raise ValueError("portfolio changed-facility ceiling exceeds the supervisory-control dimension")
    allowed = np.minimum(np.maximum(radius, 0.0), float(max_delta_per_update))
    allowed = np.where(mask, allowed, 0.0)
    delta = np.where(mask, np.clip(delta, -allowed, allowed), 0.0)

    nonzero = np.flatnonzero(np.abs(delta) > 1.0e-7)
    if nonzero.size > int(max_changed_facilities):
        order = nonzero[np.argsort(-np.abs(delta[nonzero]), kind="mergesort")]
        keep = order[: int(max_changed_facilities)]
        keep_mask = np.zeros(109, dtype=bool)
        keep_mask[keep] = True
        delta = np.where(keep_mask, delta, 0.0)

    lower, upper = _actuator_bounds(graph)
    target = np.clip(active + delta, lower, upper)
    target = active + np.clip(target - active, -allowed, allowed)
    target = np.where(mask, target, active)
    target = np.clip(target, lower, upper)
    return target.astype(np.float32)


def _normalization_tensors(
    normalization: Any, *, dtype: torch.dtype, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.as_tensor(normalization.state_mean, dtype=dtype, device=device),
        "state_std": torch.as_tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1.0e-6),
        "rain_mean": torch.as_tensor(normalization.rainfall_mean, dtype=dtype, device=device),
        "rain_std": torch.as_tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1.0e-6),
        "flow_mean": torch.as_tensor(normalization.flow_mean, dtype=dtype, device=device),
        "flow_std": torch.as_tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1.0e-6),
    }


def score_h10_first_action_targets(
    *,
    model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    candidate_targets: torch.Tensor,
    probe_chunk_size: int = 24,
) -> torch.Tensor:
    """Batch-score H10 candidate -> H350 HOLD probes with the frozen base Step2."""
    if current_state.ndim != 3 or int(current_state.shape[0]) != 1:
        raise ValueError("H10 probe scorer expects current_state [1,node,state]")
    if rainfall_scenarios.ndim != 4 or int(rainfall_scenarios.shape[0]) < 1:
        raise ValueError("H10 probe scorer expects rainfall [scenario,H,node,feature]")
    if previous_actuator_flow.shape != (1, 109):
        raise ValueError("H10 probe scorer expects previous flow [1,109]")
    if active_target.shape != (109,):
        raise ValueError("H10 probe scorer expects active target [109]")
    if candidate_targets.ndim != 2 or int(candidate_targets.shape[1]) != 109:
        raise ValueError("H10 probe candidates must be [candidate,109]")
    if int(probe_chunk_size) <= 0:
        raise ValueError("probe_chunk_size must be positive")
    if int(candidate_targets.shape[0]) == 0:
        return candidate_targets.new_empty((0,))

    device = current_state.device
    dtype = current_state.dtype
    scenarios, horizon, nodes, rain_features = rainfall_scenarios.shape
    norm = _normalization_tensors(normalization, dtype=dtype, device=device)
    state0 = (current_state - norm["state_mean"]) / norm["state_std"]
    rain0 = (rainfall_scenarios - norm["rain_mean"]) / norm["rain_std"]
    flow0 = (previous_actuator_flow - norm["flow_mean"]) / norm["flow_std"]
    up = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    down = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    physics = torch.as_tensor(graph.actuator_physics, dtype=dtype, device=device)

    scores: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(candidate_targets.shape[0]), int(probe_chunk_size)):
            targets = candidate_targets[start : start + int(probe_chunk_size)]
            count = int(targets.shape[0])
            batch = count * int(scenarios)
            state = state0.expand(batch, -1, -1)
            rain = rain0.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(
                batch, horizon, nodes, rain_features
            )
            flow = flow0.expand(batch, -1)
            active = active_target.reshape(1, 1, 109).expand(count, scenarios, -1).reshape(batch, 109)
            target = targets[:, None, :].expand(-1, scenarios, -1).reshape(batch, 109)
            reference, candidate = encode_policy_return_action_token(
                active,
                target,
                horizon_steps=int(horizon),
                first_action_steps=2,
            )
            output = model(
                current_state=state,
                rainfall=rain,
                reference_settings=reference,
                candidate_settings=candidate,
                previous_actuator_flow=flow,
                actuator_upstream=up,
                actuator_downstream=down,
                actuator_physics=physics,
            )
            value = output.total_delta_tfv_m3.reshape(count, scenarios).mean(dim=1)
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError("base Step2 H10 probe scorer produced non-finite values")
            scores.append(value.detach())
    return torch.cat(scores, dim=0)


def build_learned_h10_probe_proposal(
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
) -> LearnedH10ProbeProposal:
    """Generate a coordinated direction from supported single-actuator H10 probes."""
    active = active_target.detach().cpu().numpy().astype(np.float64).reshape(-1)
    radius = np.asarray(first_radius, dtype=np.float64).reshape(-1)
    if active.shape != (109,) or radius.shape != (109,):
        raise ValueError("learned H10 proposal requires 109-dimensional targets/support")
    mask = _validated_supervisory_mask(supervisory_mask)
    lower, upper = _actuator_bounds(graph)
    allowed = np.minimum(np.maximum(radius, 0.0), float(max_delta_per_update))
    allowed = np.where(mask, allowed, 0.0)

    probe_targets: list[np.ndarray] = []
    probe_meta: list[tuple[int, float]] = []
    for index in np.flatnonzero(mask):
        plus = min(float(allowed[index]), float(upper[index] - active[index]))
        minus = min(float(allowed[index]), float(active[index] - lower[index]))
        if plus > 1.0e-7:
            target = active.copy()
            target[index] += plus
            probe_targets.append(target)
            probe_meta.append((int(index), plus))
        if minus > 1.0e-7:
            target = active.copy()
            target[index] -= minus
            probe_targets.append(target)
            probe_meta.append((int(index), -minus))

    best_score = np.full(109, np.inf, dtype=np.float64)
    best_delta = np.zeros(109, dtype=np.float64)
    if probe_targets:
        tensor = torch.as_tensor(
            np.stack(probe_targets).astype(np.float32),
            dtype=active_target.dtype,
            device=active_target.device,
        )
        scores = score_h10_first_action_targets(
            model=model,
            normalization=normalization,
            graph=graph,
            current_state=current_state,
            rainfall_scenarios=rainfall_scenarios,
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
            candidate_targets=tensor,
            probe_chunk_size=probe_chunk_size,
        ).detach().cpu().numpy().astype(np.float64)
        for score, (index, delta) in zip(scores.tolist(), probe_meta, strict=True):
            if score < best_score[index]:
                best_score[index] = float(score)
                best_delta[index] = float(delta)

    beneficial = np.flatnonzero(
        mask & np.isfinite(best_score) & (best_score < 0.0) & (np.abs(best_delta) > 1.0e-7)
    )
    order = beneficial[np.argsort(best_score[beneficial], kind="mergesort")]
    selected = order[: int(max_changed_facilities)]
    if selected.size == 0:
        return LearnedH10ProbeProposal(
            target=None,
            facility_best_scores_m3=tuple(float(x) for x in best_score),
            predicted_beneficial_facility_count=int(beneficial.size),
            selected_facility_indices=(),
            probe_count=len(probe_meta),
        )
    raw_delta = np.zeros(109, dtype=np.float64)
    raw_delta[selected] = best_delta[selected]
    target_np = _bounded_supported_target(
        active_target=active,
        raw_delta=raw_delta,
        graph=graph,
        first_radius=radius,
        max_changed_facilities=int(max_changed_facilities),
        max_delta_per_update=float(max_delta_per_update),
        supervisory_mask=mask,
    )
    return LearnedH10ProbeProposal(
        target=torch.as_tensor(target_np, dtype=active_target.dtype, device=active_target.device),
        facility_best_scores_m3=tuple(float(x) for x in best_score),
        predicted_beneficial_facility_count=int(beneficial.size),
        selected_facility_indices=tuple(int(x) for x in selected.tolist()),
        probe_count=len(probe_meta),
    )


def build_policy_return_candidate_portfolio(
    *,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    active_target: torch.Tensor,
    graph: Any,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    learned_target: torch.Tensor | None = None,
    v12_target: torch.Tensor | None = None,
    max_delta_per_update: float = 0.5,
    shrink_scales: Iterable[float] = DEFAULT_LEARNED_SHRINK_SCALES,
    supervisory_mask: np.ndarray | None = None,
) -> tuple[PolicyReturnPortfolioCandidate, ...]:
    """Build at most three deterministic, engineering-supported first targets."""
    active = active_target.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if active.shape != (109,):
        raise ValueError("portfolio requires a 109-dimensional active target")
    mask = _validated_supervisory_mask(supervisory_mask)
    state = current_state.detach().cpu().numpy()
    if state.ndim == 3 and state.shape[0] == 1:
        state = state[0]
    rain = rainfall_scenarios.detach().cpu().numpy()

    learned_tensor = learned_target if isinstance(learned_target, torch.Tensor) else v12_target
    raw: list[tuple[str, np.ndarray]] = []
    if isinstance(learned_tensor, torch.Tensor):
        learned = learned_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if learned.shape != (109,):
            raise ValueError("portfolio learned target must contain 109 settings")
        learned_delta = np.where(mask, learned - active, 0.0)
        for value in shrink_scales:
            scale = float(value)
            if not 0.0 < scale <= 1.0:
                raise ValueError("portfolio shrink scales must lie in (0,1]")
            raw.append((f"STEP2_H10_PROBE_SCALE_{scale:.2f}", scale * learned_delta))

    raw.append(
        (
            "TYPE_AWARE_HYDRAULIC_PRESSURE",
            hydraulic_pressure_setting_delta(
                current_state=state,
                rainfall_scenarios=rain,
                graph=graph,
                max_delta_per_update=max_delta_per_update,
            ),
        )
    )

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
            supervisory_mask=mask,
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
    return tuple(candidates[:3])


__all__ = [
    "DEFAULT_LEARNED_SHRINK_SCALES",
    "DEFAULT_V12_SHRINK_SCALES",
    "DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT",
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT",
    "LearnedH10ProbeProposal",
    "PolicyReturnPortfolioCandidate",
    "_bounded_supported_target",
    "_normalization_tensors",
    "_validated_supervisory_mask",
    "build_learned_h10_probe_proposal",
    "build_policy_return_candidate_portfolio",
    "hydraulic_pressure_setting_delta",
    "score_h10_first_action_targets",
]
