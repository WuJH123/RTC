"""Causal, engineering-bounded Auto-RBC-inspired warm-start seeds for V12.3."""
from __future__ import annotations

import numpy as np


def _feature(graph, name: str, default: float | None = None) -> np.ndarray:
    names = tuple(getattr(graph, "static_node_feature_names", ()))
    if name not in names:
        if default is None:
            raise ValueError(f"knowledge seed requires graph feature {name!r}")
        return np.full(len(graph.node_ids), float(default), dtype=np.float64)
    return np.asarray(graph.static_node_features, dtype=np.float64)[:, names.index(name)]


def knowledge_guided_seed_delta_v123(
    current_state: np.ndarray,
    causal_rainfall: np.ndarray,
    graph,
    *,
    max_delta_per_update: float = 0.5,
) -> np.ndarray:
    """Return signed first-move setting deltas from causal state/topology only.

    Positive means increase the frozen actuator setting (opening/discharge direction);
    negative means retain/reduce.  This is a warm start, not a candidate ceiling and
    never reads future SWMM state, target flow, or realized future rainfall.
    """
    state = np.asarray(current_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError("knowledge seed state must be [node,6]")
    upstream = np.asarray(graph.actuator_upstream, dtype=np.int64)
    downstream = np.asarray(graph.actuator_downstream, dtype=np.int64)
    if upstream.shape != downstream.shape or np.any(upstream < 0) or np.any(downstream < 0):
        raise ValueError("knowledge seed actuator endpoints are invalid")
    if np.max(np.r_[upstream, downstream], initial=-1) >= state.shape[0]:
        raise ValueError("knowledge seed endpoint exceeds state node count")
    max_depth = np.maximum(_feature(graph, "max_depth_m", 1.0), 1e-6)
    storage_capacity = np.maximum(_feature(graph, "storage_capacity_m3", 1.0), 1e-6)
    depth_fill = np.clip(state[:, 0] / max_depth, 0.0, 1.0)
    volume_fill = np.clip(state[:, 3] / storage_capacity, 0.0, 1.0)
    flood = np.maximum(state[:, 2], 0.0)
    rainfall = np.asarray(causal_rainfall, dtype=np.float64)
    if rainfall.ndim == 4:
        rainfall = rainfall[0]
    if rainfall.ndim == 3:
        rainfall = rainfall[-1]
    if rainfall.ndim not in (2, 3):
        raise ValueError("knowledge seed rainfall must be [time,node,1] or [node,1]")
    rain_level = float(np.nanmax(np.maximum(rainfall, 0.0))) if rainfall.size else 0.0
    rain_factor = 1.0 + 0.10 * np.tanh(rain_level / 20.0)
    upstream_fill = 0.7 * depth_fill[upstream] + 0.3 * volume_fill[upstream]
    downstream_fill = 0.7 * depth_fill[downstream] + 0.3 * volume_fill[downstream]
    downstream_surcharge = np.tanh(flood[downstream] / 0.01)
    pressure = (upstream_fill - downstream_fill - 0.5 * downstream_surcharge) * rain_factor
    delta = float(max_delta_per_update) * np.tanh(pressure)
    if not np.isfinite(delta).all():
        raise ValueError("knowledge seed produced non-finite action")
    return np.clip(delta, -float(max_delta_per_update), float(max_delta_per_update)).astype(np.float32)


def build_knowledge_guided_seed_settings_v123(
    current_state: np.ndarray,
    causal_rainfall: np.ndarray,
    reference_settings: np.ndarray,
    graph,
    *,
    control_block_steps: int = 2,
    max_delta_per_update: float = 0.5,
) -> np.ndarray:
    """Build a bounded H-step warm-start sequence, preserving current setting continuity."""
    reference = np.asarray(reference_settings, dtype=np.float64)
    if reference.ndim != 2:
        raise ValueError("knowledge seed reference settings must be [H,actuator]")
    if control_block_steps <= 0 or reference.shape[0] % int(control_block_steps):
        raise ValueError("knowledge seed horizon must be divisible by control block")
    delta = knowledge_guided_seed_delta_v123(
        current_state, causal_rainfall, graph, max_delta_per_update=max_delta_per_update
    ).astype(np.float64)
    if delta.shape[0] != reference.shape[1]:
        raise ValueError("knowledge seed actuator count does not match reference settings")
    actuator_names = tuple(getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(getattr(graph, "actuator_physics", np.zeros((delta.size, 0))), dtype=np.float64)
    if "min_setting" in actuator_names and "max_setting" in actuator_names:
        lower = physics[:, actuator_names.index("min_setting")]
        upper = physics[:, actuator_names.index("max_setting")]
    else:
        lower = np.zeros(delta.size, dtype=np.float64)
        upper = np.ones(delta.size, dtype=np.float64)
    result = reference.copy()
    previous = reference[0].copy()
    for block_start in range(0, reference.shape[0], int(control_block_steps)):
        block = block_start // int(control_block_steps)
        proposed = previous + delta * (0.92 ** block)
        current = np.clip(proposed, lower, upper)
        current = previous + np.clip(current - previous, -float(max_delta_per_update), float(max_delta_per_update))
        result[block_start : block_start + int(control_block_steps)] = current
        previous = current
    return result.astype(np.float32)


__all__ = ["build_knowledge_guided_seed_settings_v123", "knowledge_guided_seed_delta_v123"]
