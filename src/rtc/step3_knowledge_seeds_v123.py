"""Causal engineering-knowledge seeds for Project7 V12.3.

The important distinction is between a *future policy simulation* and a receding-horizon
warm start.  Online we only know the current reconstructed hydraulic state.  Therefore
an Auto-RBC-inspired anchor computes the executable target for the *next* 10-minute
control block from the current Step1 state and then holds that target over the scoring
horizon.  Ten minutes later the controller reconstructs state and computes a new anchor.

This avoids pretending that today's reconstructed state is also the hydraulic state at
future control blocks.
"""
from __future__ import annotations

import numpy as np

V123_SPARSE_RBC_ANCHOR_CONTRACT = "PROJECT7_V123_SPARSE_STATE_AUTO_RBC_ANCHOR_V1"


def _feature(graph, name: str, default: float | None = None) -> np.ndarray:
    names = tuple(getattr(graph, "static_node_feature_names", ()))
    if name not in names:
        if default is None:
            raise ValueError(f"knowledge seed requires graph feature {name!r}")
        return np.full(len(graph.node_ids), float(default), dtype=np.float64)
    return np.asarray(graph.static_node_features, dtype=np.float64)[:, names.index(name)]


def _actuator_setting_bounds(graph, actuator_count: int) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(
        getattr(graph, "actuator_physics", np.zeros((actuator_count, 0))),
        dtype=np.float64,
    )
    if physics.ndim != 2 or physics.shape[0] != actuator_count:
        raise ValueError("knowledge seed actuator physics is misaligned")
    if "min_setting" in names and "max_setting" in names:
        lower = physics[:, names.index("min_setting")]
        upper = physics[:, names.index("max_setting")]
    else:
        lower = np.zeros(actuator_count, dtype=np.float64)
        upper = np.ones(actuator_count, dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("knowledge seed actuator setting bounds are invalid")
    return lower, upper


def sparse_state_auto_rbc_target_v123(
    current_state: np.ndarray,
    current_settings: np.ndarray,
    graph,
    *,
    low_fill: float = 0.25,
    high_fill: float = 0.75,
    downstream_congestion_fill: float = 0.90,
    response: float = 0.60,
    max_delta_per_update: float = 0.5,
) -> np.ndarray:
    """Compute the Auto-RBC local feedback target from Step1 reconstructed depths.

    This mirrors the fixed Auto-RBC comparator's hydraulic logic, but its observations
    come from the Proposed method's reconstructed whole-network state rather than true
    actuator-adjacent SWMM depths.  It is therefore a low-sensor engineering anchor, not
    the comparator itself.
    """

    if not (0.0 <= float(low_fill) < float(high_fill) <= 1.0):
        raise ValueError("sparse-state RBC filling thresholds are invalid")
    if not 0.0 < float(downstream_congestion_fill) < 1.0:
        raise ValueError("sparse-state RBC downstream threshold must lie in (0,1)")
    if not 0.0 < float(response) <= 1.0:
        raise ValueError("sparse-state RBC response must lie in (0,1]")
    if not 0.0 <= float(max_delta_per_update) <= 1.0:
        raise ValueError("sparse-state RBC max delta must lie in [0,1]")

    state = np.asarray(current_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 1 or not np.isfinite(state).all():
        raise ValueError("sparse-state RBC state must be finite [node,state]")
    current = np.asarray(current_settings, dtype=np.float64).reshape(-1)
    upstream = np.asarray(graph.actuator_upstream, dtype=np.int64)
    downstream = np.asarray(graph.actuator_downstream, dtype=np.int64)
    if current.size != upstream.size or upstream.shape != downstream.shape:
        raise ValueError("sparse-state RBC actuator dimensions are misaligned")
    if np.any(upstream < 0) or np.any(downstream < 0):
        raise ValueError("sparse-state RBC actuator endpoints are invalid")
    if np.max(np.r_[upstream, downstream], initial=-1) >= state.shape[0]:
        raise ValueError("sparse-state RBC endpoint exceeds reconstructed node count")

    max_depth = np.maximum(_feature(graph, "max_depth_m", 1.0), 1.0e-6)
    depth = np.maximum(state[:, 0], 0.0)
    up_fill = np.clip(depth[upstream] / max_depth[upstream], 0.0, 1.5)
    down_fill = np.clip(depth[downstream] / max_depth[downstream], 0.0, 1.5)

    open_drive = np.clip(
        (up_fill - float(low_fill)) / (float(high_fill) - float(low_fill)),
        0.0,
        1.0,
    )
    downstream_penalty = np.clip(
        (down_fill - float(downstream_congestion_fill))
        / (1.0 - float(downstream_congestion_fill)),
        0.0,
        1.0,
    )
    raw = open_drive * (1.0 - downstream_penalty)
    desired = current + float(response) * (raw - current)

    lower, upper = _actuator_setting_bounds(graph, current.size)
    desired = np.clip(desired, lower, upper)
    desired = current + np.clip(
        desired - current,
        -float(max_delta_per_update),
        float(max_delta_per_update),
    )
    desired = np.clip(desired, lower, upper)
    if not np.isfinite(desired).all():
        raise ValueError("sparse-state RBC produced non-finite target")
    return desired.astype(np.float32)


def build_sparse_state_auto_rbc_anchor_v123(
    current_state: np.ndarray,
    current_settings: np.ndarray,
    reference_settings: np.ndarray,
    graph,
    *,
    control_block_steps: int = 2,
    max_delta_per_update: float = 0.5,
) -> np.ndarray:
    """Build one executable RBC first move followed by terminal target hold.

    Future RBC actions are *not* hallucinated from the current state.  The returned plan
    is a receding-horizon anchor: score the current hydraulic feedback action and hold its
    target; recompute the anchor from fresh observations at the next decision epoch.
    """

    reference = np.asarray(reference_settings, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] <= 0:
        raise ValueError("sparse-state RBC reference settings must be [H,actuator]")
    if control_block_steps <= 0 or reference.shape[0] % int(control_block_steps):
        raise ValueError("sparse-state RBC horizon/control block mismatch")
    current = np.asarray(current_settings, dtype=np.float64).reshape(-1)
    if current.size != reference.shape[1]:
        raise ValueError("sparse-state RBC current/reference actuator mismatch")
    target = sparse_state_auto_rbc_target_v123(
        current_state,
        current,
        graph,
        max_delta_per_update=max_delta_per_update,
    ).astype(np.float64)
    result = np.repeat(target[None, :], reference.shape[0], axis=0)
    return result.astype(np.float32)


def knowledge_guided_seed_delta_v123(
    current_state: np.ndarray,
    causal_rainfall: np.ndarray,
    graph,
    *,
    max_delta_per_update: float = 0.5,
) -> np.ndarray:
    """Return a generic hydraulic-pressure first-move delta from causal information.

    Retained as a complementary diagnostic/warm-start family.  The sparse-state RBC
    anchor above is the primary engineering anchor because it matches the comparator's
    local feedback mechanism while respecting the Proposed information budget.
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
    """Build the legacy generic pressure seed sequence for diagnostics/ablation."""
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
    lower, upper = _actuator_setting_bounds(graph, delta.size)
    result = reference.copy()
    previous = reference[0].copy()
    for block_start in range(0, reference.shape[0], int(control_block_steps)):
        block = block_start // int(control_block_steps)
        proposed = previous + delta * (0.92 ** block)
        current = np.clip(proposed, lower, upper)
        current = previous + np.clip(
            current - previous,
            -float(max_delta_per_update),
            float(max_delta_per_update),
        )
        result[block_start : block_start + int(control_block_steps)] = current
        previous = current
    return result.astype(np.float32)


__all__ = [
    "V123_SPARSE_RBC_ANCHOR_CONTRACT",
    "build_knowledge_guided_seed_settings_v123",
    "build_sparse_state_auto_rbc_anchor_v123",
    "knowledge_guided_seed_delta_v123",
    "sparse_state_auto_rbc_target_v123",
]
