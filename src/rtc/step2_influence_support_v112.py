"""Sparse, signed D2 hydraulic-support labels for Step2 V11.2.

D2 labels are TrainFit-only supervision. They answer whether a setting change
created a real flow response and where/when signed hydraulic effects occurred.
No clipping is applied to counterfactual deltas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

V112_INFLUENCE_SUPPORT_CONTRACT = "PROJECT7_STEP2_V112_INFLUENCE_SUPPORT_V1"
STATE_EFFECT_CHANNELS_V112 = (0, 2, 3, 4, 5)
STATE_EFFECT_NAMES_V112 = (
    "depth_m", "flooding_rate_m3s", "storage_volume_m3", "inflow_m3s", "outflow_m3s"
)


@dataclass(frozen=True)
class InfluenceSupportTargetsV112:
    retained_indices: np.ndarray
    delta_state: np.ndarray
    state_active: np.ndarray
    delta_flow: np.ndarray
    flow_active: np.ndarray


def infer_single_changed_actuator_v112(
    reference_settings: np.ndarray,
    candidate_settings: np.ndarray,
    *,
    epsilon: float = 1.0e-7,
) -> int:
    ref = np.asarray(reference_settings, dtype=np.float64)
    cand = np.asarray(candidate_settings, dtype=np.float64)
    if ref.ndim != 2 or cand.shape != ref.shape:
        raise ValueError("V112 D2 settings must be aligned [H,A]")
    changed = (np.abs(cand - ref) > float(epsilon)).any(axis=0)
    indices = np.flatnonzero(changed)
    if indices.size != 1:
        raise ValueError(f"V112 independent D2 probe requires one changed actuator; got {indices.size}")
    return int(indices[0])


def build_influence_support_targets_v112(
    *,
    reference_states: np.ndarray,
    candidate_states: np.ndarray,
    reference_flows: np.ndarray,
    candidate_flows: np.ndarray,
    state_active_threshold: np.ndarray,
    flow_active_threshold: np.ndarray,
    retained_indices: Sequence[int],
) -> InfluenceSupportTargetsV112:
    ref_s = np.asarray(reference_states, dtype=np.float64)
    cand_s = np.asarray(candidate_states, dtype=np.float64)
    ref_f = np.asarray(reference_flows, dtype=np.float64)
    cand_f = np.asarray(candidate_flows, dtype=np.float64)
    if ref_s.ndim != 3 or ref_s.shape[-1] != 6 or cand_s.shape != ref_s.shape:
        raise ValueError("V112 states must be aligned [H,N,6]")
    if ref_f.ndim != 2 or cand_f.shape != ref_f.shape or ref_f.shape[0] != ref_s.shape[0]:
        raise ValueError("V112 flows must be aligned [H,A] with the state horizon")
    idx = np.asarray([int(i) for i in retained_indices], dtype=np.int64)
    if idx.size == 0 or (idx < 0).any() or (idx >= ref_s.shape[0]).any():
        raise ValueError("V112 retained indices outside source horizon")
    st = np.asarray(state_active_threshold, dtype=np.float64)
    ft = np.asarray(flow_active_threshold, dtype=np.float64).reshape(-1)
    if st.shape != (ref_s.shape[1], 5) or ft.shape != (ref_f.shape[1],):
        raise ValueError("V112 active-threshold dimensions do not match state/flow domains")
    if (st <= 0).any() or (ft <= 0).any() or not np.isfinite(st).all() or not np.isfinite(ft).all():
        raise ValueError("V112 active thresholds must be finite and positive")
    delta_s = cand_s[idx][..., list(STATE_EFFECT_CHANNELS_V112)] - ref_s[idx][..., list(STATE_EFFECT_CHANNELS_V112)]
    delta_f = cand_f[idx] - ref_f[idx]
    return InfluenceSupportTargetsV112(
        retained_indices=idx,
        delta_state=delta_s,
        state_active=np.abs(delta_s) >= st[None],
        delta_flow=delta_f,
        flow_active=np.abs(delta_f) >= ft[None],
    )


def source_flow_effective_v112(targets: InfluenceSupportTargetsV112, source_actuator: int) -> bool:
    a = int(source_actuator)
    if a < 0 or a >= targets.flow_active.shape[1]:
        raise IndexError("V112 source actuator outside flow domain")
    return bool(targets.flow_active[:, a].any())


def summarize_support_v112(
    targets: InfluenceSupportTargetsV112,
    *,
    response_minutes: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    minutes = np.asarray(response_minutes, dtype=np.float64)
    if minutes.shape != (targets.state_active.shape[0],):
        raise ValueError("V112 response minutes do not match retained horizon")
    out: dict[str, dict[str, float | int]] = {}
    for k, name in enumerate(STATE_EFFECT_NAMES_V112):
        active = targets.state_active[..., k]
        delta = targets.delta_state[..., k]
        signed = delta[active]
        active_times = active.any(axis=1)
        active_nodes = active.any(axis=0)
        out[name] = {
            "active_node_count": int(active_nodes.sum()),
            "active_node_fraction": float(active_nodes.mean()),
            "active_cell_fraction": float(active.mean()),
            "positive_active_fraction": float(np.mean(signed > 0)) if signed.size else 0.0,
            "negative_active_fraction": float(np.mean(signed < 0)) if signed.size else 0.0,
            "max_abs_effect": float(np.max(np.abs(signed))) if signed.size else 0.0,
            "onset_min": float(minutes[np.flatnonzero(active_times)[0]]) if active_times.any() else 0.0,
            "peak_response_min": float(minutes[int(np.argmax(np.abs(delta).sum(axis=1)))]) if active_times.any() else 0.0,
        }
    return out


__all__ = [
    "InfluenceSupportTargetsV112", "STATE_EFFECT_CHANNELS_V112", "STATE_EFFECT_NAMES_V112",
    "V112_INFLUENCE_SUPPORT_CONTRACT", "build_influence_support_targets_v112",
    "infer_single_changed_actuator_v112", "source_flow_effective_v112", "summarize_support_v112",
]
