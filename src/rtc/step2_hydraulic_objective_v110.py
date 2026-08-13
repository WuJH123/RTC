"""Three-part Hydraulic Effect supervision for Project7 Step2 V11.

Each response cell learns:
1) whether the counterfactual action has a locally meaningful effect (active);
2) the signed direction of that effect (positive/negative);
3) the magnitude conditional on an active effect.

"Active" is no longer defined with one global channel scale.  Thresholds are
frozen from TrainFit-only node/actuator P90 counterfactual magnitudes and
physical floors (node depth/storage capacity and SI floors).  Holdout data never
selects thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import HydraulicEffectOutputV110, STATE_EFFECT_CHANNELS_V110
from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache
from .step2_v110_contract import (
    ActiveEffectContractV110,
    ActuatorSetHydraulicContractV110,
    HydraulicHorizonV110,
    V110_ACTIVE_EFFECT_CONTRACT,
)


@dataclass(frozen=True)
class EffectScalesV110:
    state_magnitude_scale: np.ndarray       # [N,5]
    state_active_threshold: np.ndarray      # [N,5]
    flow_magnitude_scale: np.ndarray        # [A]
    flow_active_threshold: np.ndarray       # [A]
    state_active_pos_weight: np.ndarray     # [5]
    flow_active_pos_weight: np.ndarray      # [A]
    source_group_count: int
    contract: str = V110_ACTIVE_EFFECT_CONTRACT


def _sqrt_pos_weight(active: np.ndarray, axes: tuple[int, ...], cap: float = 20.0) -> np.ndarray:
    positive = active.sum(axis=axes, dtype=np.float64)
    total = np.prod([active.shape[axis] for axis in axes], dtype=np.int64)
    negative = total - positive
    weight = np.sqrt((negative + 1.0) / (positive + 1.0))
    return np.clip(weight, 1.0, cap).astype(np.float32)


def derive_effect_scales_v110(
    cache: V60TrainCache,
    fit_d2_names: Sequence[str],
    prepared: PreparedStaticV60,
    *,
    horizon: HydraulicHorizonV110 = HydraulicHorizonV110(),
    contract: ActiveEffectContractV110 = ActiveEffectContractV110(),
) -> EffectScalesV110:
    """Freeze local meaningful-effect thresholds using TrainFit D2 only."""
    horizon.validate()
    contract.validate()
    if not fit_d2_names or any(not name.startswith("D2::") for name in fit_d2_names):
        raise ValueError("V11 effect scales require non-empty TrainFit D2 names only")
    indices = np.asarray(horizon.indices(), dtype=np.int64)
    state_values: list[np.ndarray] = []
    flow_values: list[np.ndarray] = []
    for name in fit_d2_names:
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        candidates = [i for i in entry.indices if i != ref]
        if not candidates:
            continue
        ref_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)[indices]
        cand_state = np.asarray(arrays["target_states"][candidates], dtype=np.float64)[:, indices]
        delta_state = cand_state - ref_state[None]
        state_values.append(np.abs(delta_state[..., list(STATE_EFFECT_CHANNELS_V110)]))
        ref_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)[indices]
        cand_flow = np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64)[:, indices]
        flow_values.append(np.abs(cand_flow - ref_flow[None]))
    if not state_values or not flow_values:
        raise ValueError("V11 could not derive effect scales from empty D2 candidates")

    state_abs = np.concatenate(state_values, axis=0)  # [samples,T,N,5]
    flow_abs = np.concatenate(flow_values, axis=0)    # [samples,T,A]
    state_p90 = np.quantile(state_abs, contract.quantile, axis=(0, 1)).astype(np.float32)
    flow_p90 = np.quantile(flow_abs, contract.quantile, axis=(0, 1)).astype(np.float32)

    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float32)
    storage_capacity = prepared.storage_capacity_m3.detach().cpu().numpy().astype(np.float32)
    depth_floor = np.maximum(
        contract.depth_floor_m,
        contract.depth_capacity_fraction * np.maximum(max_depth, contract.depth_floor_m),
    )
    volume_floor = np.maximum(
        contract.volume_floor_m3,
        contract.storage_capacity_fraction * storage_capacity,
    )
    physical_floor = np.stack(
        (
            depth_floor,
            np.full_like(depth_floor, contract.flood_floor_m3s),
            volume_floor,
            np.full_like(depth_floor, contract.node_flow_floor_m3s),
            np.full_like(depth_floor, contract.node_flow_floor_m3s),
        ),
        axis=-1,
    ).astype(np.float32)
    state_magnitude = np.maximum(state_p90, physical_floor).astype(np.float32)
    state_threshold = np.maximum(
        contract.quantile_fraction * state_p90, physical_floor
    ).astype(np.float32)
    flow_floor = np.full_like(flow_p90, contract.actuator_flow_floor_m3s)
    flow_magnitude = np.maximum(flow_p90, flow_floor).astype(np.float32)
    flow_threshold = np.maximum(
        contract.quantile_fraction * flow_p90, flow_floor
    ).astype(np.float32)

    state_active = state_abs >= state_threshold[None, None]
    flow_active = flow_abs >= flow_threshold[None, None]
    state_pos_weight = _sqrt_pos_weight(state_active, axes=(0, 1, 2))
    flow_pos_weight = _sqrt_pos_weight(flow_active, axes=(0, 1))
    return EffectScalesV110(
        state_magnitude_scale=state_magnitude,
        state_active_threshold=state_threshold,
        flow_magnitude_scale=flow_magnitude,
        flow_active_threshold=flow_threshold,
        state_active_pos_weight=state_pos_weight,
        flow_active_pos_weight=flow_pos_weight,
        source_group_count=len(fit_d2_names),
    )


def _focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    pt = target * probability + (1.0 - target) * (1.0 - probability)
    raw = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    return (1.0 - pt).clamp_min(1.0e-6).pow(float(gamma)) * raw


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.new_zeros(())


def hydraulic_effect_loss_v110(
    output: HydraulicEffectOutputV110,
    batch: V60GroupBatch,
    scales: EffectScalesV110,
    *,
    contract: ActuatorSetHydraulicContractV110 = ActuatorSetHydraulicContractV110(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct short-horizon signed response with active/sign/magnitude decomposition."""
    contract.validate()
    idx = output.horizon_indices
    true_ref_state = batch.true_reference_states.index_select(1, idx)[:, None]
    true_cand_state = batch.true_candidate_states.index_select(2, idx)
    true_delta_full = true_cand_state - true_ref_state
    true_state = true_delta_full[..., list(STATE_EFFECT_CHANNELS_V110)]

    true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None]
    true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
    true_flow = true_cand_flow - true_ref_flow

    state_scale = torch.as_tensor(
        scales.state_magnitude_scale, dtype=true_state.dtype, device=true_state.device
    ).reshape(1, 1, 1, *scales.state_magnitude_scale.shape)
    state_threshold = torch.as_tensor(
        scales.state_active_threshold, dtype=true_state.dtype, device=true_state.device
    ).reshape(1, 1, 1, *scales.state_active_threshold.shape)
    flow_scale = torch.as_tensor(
        scales.flow_magnitude_scale, dtype=true_flow.dtype, device=true_flow.device
    ).reshape(1, 1, 1, -1)
    flow_threshold = torch.as_tensor(
        scales.flow_active_threshold, dtype=true_flow.dtype, device=true_flow.device
    ).reshape(1, 1, 1, -1)

    state_active = true_state.abs() >= state_threshold
    flow_active = true_flow.abs() >= flow_threshold
    state_sign_target = (true_state > 0).to(true_state.dtype)
    flow_sign_target = (true_flow > 0).to(true_flow.dtype)
    state_mag_target = true_state.abs() / state_scale
    flow_mag_target = true_flow.abs() / flow_scale

    state_pos_weight = torch.as_tensor(
        scales.state_active_pos_weight,
        dtype=true_state.dtype,
        device=true_state.device,
    ).reshape(1, 1, 1, 1, -1)
    flow_pos_weight = torch.as_tensor(
        scales.flow_active_pos_weight,
        dtype=true_flow.dtype,
        device=true_flow.device,
    ).reshape(1, 1, 1, -1)

    active_state = _focal_bce(
        output.active_state_logits,
        state_active.to(true_state.dtype),
        pos_weight=state_pos_weight,
        gamma=contract.active_focal_gamma,
    ).mean()
    active_flow = _focal_bce(
        output.active_flow_logits,
        flow_active.to(true_flow.dtype),
        pos_weight=flow_pos_weight,
        gamma=contract.active_focal_gamma,
    ).mean()

    state_sign_raw = F.binary_cross_entropy_with_logits(
        output.sign_state_logits, state_sign_target, reduction="none"
    )
    flow_sign_raw = F.binary_cross_entropy_with_logits(
        output.sign_flow_logits, flow_sign_target, reduction="none"
    )
    sign_state = _masked_mean(state_sign_raw, state_active)
    sign_flow = _masked_mean(flow_sign_raw, flow_active)

    state_mag_raw = F.smooth_l1_loss(
        torch.log1p(output.magnitude_state_normalized),
        torch.log1p(state_mag_target),
        reduction="none",
        beta=0.5,
    )
    flow_mag_raw = F.smooth_l1_loss(
        torch.log1p(output.magnitude_flow_normalized),
        torch.log1p(flow_mag_target),
        reduction="none",
        beta=0.5,
    )
    magnitude_state = _masked_mean(state_mag_raw, state_active)
    magnitude_flow = _masked_mean(flow_mag_raw, flow_active)

    predicted_state = output.raw_delta_states_physical[..., list(STATE_EFFECT_CHANNELS_V110)]
    predicted_flow = output.raw_delta_flows_physical
    dense_state = F.smooth_l1_loss(
        predicted_state / state_scale, true_state / state_scale, beta=0.5
    )
    dense_flow = F.smooth_l1_loss(
        predicted_flow / flow_scale, true_flow / flow_scale, beta=0.5
    )

    # Teach the response trend/timing without recurrent rollout or a generic
    # smoothness prior: predicted finite differences must match SWMM finite
    # differences over the irregular retained-time grid.
    minutes = output.response_minutes.to(true_state)
    dt = (minutes[1:] - minutes[:-1]).clamp_min(1.0)
    pred_state_norm = predicted_state / state_scale
    true_state_norm = true_state / state_scale
    pred_flow_norm = predicted_flow / flow_scale
    true_flow_norm = true_flow / flow_scale
    state_slope_pred = (pred_state_norm[:, :, 1:] - pred_state_norm[:, :, :-1]) / dt[
        None, None, :, None, None
    ]
    state_slope_true = (true_state_norm[:, :, 1:] - true_state_norm[:, :, :-1]) / dt[
        None, None, :, None, None
    ]
    flow_slope_pred = (pred_flow_norm[:, :, 1:] - pred_flow_norm[:, :, :-1]) / dt[
        None, None, :, None
    ]
    flow_slope_true = (true_flow_norm[:, :, 1:] - true_flow_norm[:, :, :-1]) / dt[
        None, None, :, None
    ]
    temporal_state = F.smooth_l1_loss(state_slope_pred, state_slope_true, beta=0.25)
    temporal_flow = F.smooth_l1_loss(flow_slope_pred, flow_slope_true, beta=0.25)

    active = 0.5 * (active_state + active_flow)
    sign = 0.5 * (sign_state + sign_flow)
    magnitude = 0.5 * (magnitude_state + magnitude_flow)
    dense = 0.5 * (dense_state + dense_flow)
    temporal = 0.5 * (temporal_state + temporal_flow)
    total = (
        contract.active_weight * active
        + contract.sign_weight * sign
        + contract.magnitude_weight * magnitude
        + contract.dense_delta_weight * dense
        + contract.temporal_difference_weight * temporal
    )
    return total, {
        "loss": float(total.detach()),
        "active": float(active.detach()),
        "active_state": float(active_state.detach()),
        "active_flow": float(active_flow.detach()),
        "sign": float(sign.detach()),
        "sign_state": float(sign_state.detach()),
        "sign_flow": float(sign_flow.detach()),
        "magnitude": float(magnitude.detach()),
        "magnitude_state": float(magnitude_state.detach()),
        "magnitude_flow": float(magnitude_flow.detach()),
        "dense_delta": float(dense.detach()),
        "temporal_difference": float(temporal.detach()),
        "state_active_fraction": float(state_active.float().mean().detach()),
        "flow_active_fraction": float(flow_active.float().mean().detach()),
    }


__all__ = [
    "EffectScalesV110",
    "derive_effect_scales_v110",
    "hydraulic_effect_loss_v110",
]
