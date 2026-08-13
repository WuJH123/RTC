"""V11.1 active-conditional scales and balanced direct signed loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import STATE_EFFECT_CHANNELS_V110
from .step2_hydraulic_objective_v110 import _sqrt_pos_weight
from .step2_train_response_v60 import V60GroupBatch, V60TrainCache
from .step2_v110_contract import ActiveEffectContractV110, HydraulicHorizonV110
from .step2_v111_contract import V111LossContract


@dataclass(frozen=True)
class EffectScalesV111:
    state_magnitude_scale: np.ndarray
    state_active_threshold: np.ndarray
    flow_magnitude_scale: np.ndarray
    flow_active_threshold: np.ndarray
    state_active_pos_weight: np.ndarray
    flow_active_pos_weight: np.ndarray
    state_local_active_support: np.ndarray
    flow_local_active_support: np.ndarray
    state_fallback_fraction: float
    flow_fallback_fraction: float
    state_active_quantiles: dict[str, list[float]]
    flow_active_quantiles: dict[str, list[float]]
    source_group_count: int
    contract: str = "PROJECT7_STEP2_V111_ACTIVE_CONDITIONAL_SCALE_V1"


def _quantile_or_floor(values: np.ndarray, q: float, floor: float) -> float:
    return max(float(np.quantile(values, q)) if values.size else 0.0, float(floor))


def derive_effect_scales_v111(
    cache: V60TrainCache,
    fit_d2_names: Sequence[str],
    prepared: PreparedStaticV60,
    *,
    horizon: HydraulicHorizonV110 = HydraulicHorizonV110(),
    contract: ActiveEffectContractV110 = ActiveEffectContractV110(),
    scale_contract: V111LossContract = V111LossContract(),
) -> EffectScalesV111:
    """Derive thresholds as in V11, then scales from active TrainFit cells only."""
    horizon.validate()
    contract.validate()
    scale_contract.validate()
    if not fit_d2_names or any(not n.startswith("D2::") for n in fit_d2_names):
        raise ValueError("V111 scales require non-empty TrainFit D2 names only")
    idx = np.asarray(horizon.indices(), dtype=np.int64)
    state_values: list[np.ndarray] = []
    flow_values: list[np.ndarray] = []
    for name in fit_d2_names:
        entry = cache.entry(name)
        ref = int(entry.reference_index)
        cand = [i for i in entry.indices if i != ref]
        if not cand:
            continue
        states = np.asarray(entry.arrays["target_states"], dtype=np.float64)
        flows = np.asarray(entry.arrays["target_actuator_flows"], dtype=np.float64)
        state_values.append(
            np.abs(states[cand][:, idx] - states[ref][idx][None])[..., list(STATE_EFFECT_CHANNELS_V110)]
        )
        flow_values.append(np.abs(flows[cand][:, idx] - flows[ref][idx][None]))
    if not state_values or not flow_values:
        raise ValueError("V111 cannot derive scales from an empty D2 corpus")
    state_abs = np.concatenate(state_values, axis=0)  # [sample,T,N,5]
    flow_abs = np.concatenate(flow_values, axis=0)  # [sample,T,A]
    all_state_p90 = np.quantile(state_abs, contract.quantile, axis=(0, 1))
    all_flow_p90 = np.quantile(flow_abs, contract.quantile, axis=(0, 1))
    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float64)
    capacity = prepared.storage_capacity_m3.detach().cpu().numpy().astype(np.float64)
    depth_floor = np.maximum(contract.depth_floor_m,
                             contract.depth_capacity_fraction * np.maximum(max_depth, contract.depth_floor_m))
    volume_floor = np.maximum(contract.volume_floor_m3,
                              contract.storage_capacity_fraction * capacity)
    floors = np.stack((depth_floor, np.full_like(depth_floor, contract.flood_floor_m3s),
                       volume_floor, np.full_like(depth_floor, contract.node_flow_floor_m3s),
                       np.full_like(depth_floor, contract.node_flow_floor_m3s)), axis=-1)
    state_threshold = np.maximum(contract.quantile_fraction * all_state_p90, floors)
    flow_floor = np.full_like(all_flow_p90, contract.actuator_flow_floor_m3s)
    flow_threshold = np.maximum(contract.quantile_fraction * all_flow_p90, flow_floor)
    state_active = state_abs >= state_threshold[None, None]
    flow_active = flow_abs >= flow_threshold[None, None]
    global_state: list[np.ndarray] = []
    for channel in range(5):
        vals = state_abs[..., channel][state_active[..., channel]]
        global_state.append(vals)
    global_flow = [flow_abs[..., a][flow_active[..., a]] for a in range(flow_abs.shape[-1])]
    global_state_all = [state_abs[..., c][state_active[..., c]] for c in range(5)]
    state_scale = np.empty_like(state_threshold, dtype=np.float32)
    flow_scale = np.empty_like(flow_threshold, dtype=np.float32)
    state_counts = state_active.sum(axis=(0, 1)).astype(np.int64)
    flow_counts = flow_active.sum(axis=(0, 1)).astype(np.int64)
    state_fallback = 0
    for node in range(state_abs.shape[-2]):
        for channel in range(5):
            local = state_abs[..., channel][:, :, node][state_active[..., channel][:, :, node]]
            fallback = local.size < scale_contract.local_support_min
            pool = global_state_all[channel] if fallback else local
            state_scale[node, channel] = max(_quantile_or_floor(pool, 0.75, state_threshold[node, channel]),
                                              float(state_threshold[node, channel]))
            state_fallback += int(fallback)
    flow_fallback = 0
    for actuator in range(flow_abs.shape[-1]):
        local = flow_abs[..., actuator][flow_active[..., actuator]]
        fallback = local.size < scale_contract.local_support_min
        pool = np.concatenate(global_flow) if fallback and any(x.size for x in global_flow) else local
        flow_scale[actuator] = max(_quantile_or_floor(pool, 0.75, flow_threshold[actuator]),
                                   float(flow_threshold[actuator]))
        flow_fallback += int(fallback)
    state_q = {name: [float(np.quantile(np.concatenate(global_state_all), q)) if any(x.size for x in global_state_all) else 0.0
                      for q in (0.5, 0.9, 0.95, 0.99, 1.0)]
               for name in ("p50", "p90", "p95", "p99", "max")}
    # Keep a compact, unambiguous quantile record; the key names are the
    # statistics and the values are per-channel below.
    state_quantiles = {key: [float(np.quantile(vals, q)) if vals.size else 0.0
                             for vals in global_state_all]
                       for key, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1.0))}
    flow_all = np.concatenate(global_flow) if any(x.size for x in global_flow) else np.empty(0)
    flow_quantiles = {key: [float(np.quantile(vals, q)) if vals.size else 0.0
                            for vals in global_flow]
                      for key, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1.0))}
    return EffectScalesV111(
        state_magnitude_scale=state_scale,
        state_active_threshold=state_threshold.astype(np.float32),
        flow_magnitude_scale=flow_scale,
        flow_active_threshold=flow_threshold.astype(np.float32),
        state_active_pos_weight=_sqrt_pos_weight(state_active, axes=(0, 1, 2)),
        flow_active_pos_weight=_sqrt_pos_weight(flow_active, axes=(0, 1)),
        state_local_active_support=state_counts,
        flow_local_active_support=flow_counts,
        state_fallback_fraction=state_fallback / float(state_counts.size),
        flow_fallback_fraction=flow_fallback / float(flow_counts.size),
        state_active_quantiles=state_quantiles,
        flow_active_quantiles=flow_quantiles,
        source_group_count=len(fit_d2_names),
    )


def _balanced_direct(pred: torch.Tensor, truth: torch.Tensor,
                     scale: torch.Tensor, threshold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = truth.abs() >= threshold
    active_loss = F.smooth_l1_loss(pred / scale, truth / scale, beta=0.5, reduction="none")
    inactive_loss = F.smooth_l1_loss(pred / threshold, torch.zeros_like(pred), beta=0.5, reduction="none")
    per_channel = []
    active_parts = []
    inactive_parts = []
    for channel in range(pred.shape[-1]):
        am = active[..., channel]
        im = ~am
        a = active_loss[..., channel][am].mean() if bool(am.any()) else pred.new_zeros(())
        i = inactive_loss[..., channel][im].mean() if bool(im.any()) else pred.new_zeros(())
        if bool(am.any()) and bool(im.any()):
            value = 0.5 * a + 0.5 * i
        elif bool(am.any()):
            value = a
        else:
            value = i
        per_channel.append(value)
        active_parts.append(a)
        inactive_parts.append(i)
    return torch.stack(per_channel).mean(), torch.stack(active_parts).mean(), torch.stack(inactive_parts).mean()


def hydraulic_effect_loss_v111(output, batch: V60GroupBatch, scales: EffectScalesV111,
                               *, contract: V111LossContract = V111LossContract()) -> tuple[torch.Tensor, dict[str, float]]:
    contract.validate()
    idx = output.horizon_indices
    true_ref_state = batch.true_reference_states.index_select(1, idx)[:, None]
    true_cand_state = batch.true_candidate_states.index_select(2, idx)
    true_state = (true_cand_state - true_ref_state)[..., list(STATE_EFFECT_CHANNELS_V110)]
    true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None]
    true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
    true_flow = true_cand_flow - true_ref_flow
    state_scale = torch.as_tensor(scales.state_magnitude_scale, device=true_state.device, dtype=true_state.dtype)[None, None, None]
    state_threshold = torch.as_tensor(scales.state_active_threshold, device=true_state.device, dtype=true_state.dtype)[None, None, None]
    flow_scale = torch.as_tensor(scales.flow_magnitude_scale, device=true_flow.device, dtype=true_flow.dtype)[None, None, None]
    flow_threshold = torch.as_tensor(scales.flow_active_threshold, device=true_flow.device, dtype=true_flow.dtype)[None, None, None]
    pred_state = output.raw_delta_states_physical[..., list(STATE_EFFECT_CHANNELS_V110)]
    pred_flow = output.raw_delta_flows_physical
    direct_state, active_state_direct, inactive_state_direct = _balanced_direct(pred_state, true_state, state_scale, state_threshold)
    direct_flow, active_flow_direct, inactive_flow_direct = _balanced_direct(pred_flow, true_flow, flow_scale, flow_threshold)
    direct = 0.5 * (direct_state + direct_flow)
    minutes = output.response_minutes.to(true_state)
    dt = (minutes[1:] - minutes[:-1]).clamp_min(1.0)
    ps = pred_state / state_scale
    ts = true_state / state_scale
    pf = pred_flow / flow_scale
    tf = true_flow / flow_scale
    temporal_state = F.smooth_l1_loss((ps[:, :, 1:] - ps[:, :, :-1]) / dt[None, None, :, None, None],
                                      (ts[:, :, 1:] - ts[:, :, :-1]) / dt[None, None, :, None, None], beta=.25)
    temporal_flow = F.smooth_l1_loss((pf[:, :, 1:] - pf[:, :, :-1]) / dt[None, None, :, None],
                                     (tf[:, :, 1:] - tf[:, :, :-1]) / dt[None, None, :, None], beta=.25)
    temporal = 0.5 * (temporal_state + temporal_flow)
    state_active = true_state.abs() >= state_threshold
    flow_active = true_flow.abs() >= flow_threshold
    active_aux = 0.5 * (
        F.binary_cross_entropy_with_logits(output.active_state_logits, state_active.to(pred_state), reduction="mean")
        + F.binary_cross_entropy_with_logits(output.active_flow_logits, flow_active.to(pred_flow), reduction="mean")
    )
    sign_state = F.binary_cross_entropy_with_logits(output.sign_state_logits, (true_state > 0).to(pred_state), reduction="none")
    sign_flow = F.binary_cross_entropy_with_logits(output.sign_flow_logits, (true_flow > 0).to(pred_flow), reduction="none")
    sign_aux = 0.5 * ((sign_state[state_active].mean() if bool(state_active.any()) else pred_state.new_zeros(()))
                      + (sign_flow[flow_active].mean() if bool(flow_active.any()) else pred_flow.new_zeros(())))
    state_mag_target = true_state.abs() / state_scale
    flow_mag_target = true_flow.abs() / flow_scale
    mag_state = F.smooth_l1_loss(torch.log1p(output.magnitude_state_normalized), torch.log1p(state_mag_target), reduction="none")
    mag_flow = F.smooth_l1_loss(torch.log1p(output.magnitude_flow_normalized), torch.log1p(flow_mag_target), reduction="none")
    magnitude_aux = 0.5 * ((mag_state[state_active].mean() if bool(state_active.any()) else pred_state.new_zeros(()))
                            + (mag_flow[flow_active].mean() if bool(flow_active.any()) else pred_flow.new_zeros(())))
    total = (contract.direct_weight * direct + contract.temporal_weight * temporal
             + contract.active_aux_weight * active_aux + contract.sign_aux_weight * sign_aux
             + contract.magnitude_aux_weight * magnitude_aux)
    return total, {
        "loss": float(total.detach()), "direct_signed": float(direct.detach()),
        "direct_active": float((active_state_direct + active_flow_direct).mul(.5).detach()),
        "direct_inactive": float((inactive_state_direct + inactive_flow_direct).mul(.5).detach()),
        "direct_state": float(direct_state.detach()), "direct_flow": float(direct_flow.detach()),
        "temporal": float(temporal.detach()), "active_aux": float(active_aux.detach()),
        "sign_aux": float(sign_aux.detach()), "magnitude_aux": float(magnitude_aux.detach()),
        "weighted_direct": float((contract.direct_weight * direct).detach()),
        "weighted_temporal": float((contract.temporal_weight * temporal).detach()),
        "weighted_active_aux": float((contract.active_aux_weight * active_aux).detach()),
        "weighted_sign_aux": float((contract.sign_aux_weight * sign_aux).detach()),
        "weighted_magnitude_aux": float((contract.magnitude_aux_weight * magnitude_aux).detach()),
        "state_active_fraction": float(state_active.float().mean().detach()),
        "flow_active_fraction": float(flow_active.float().mean().detach()),
    }


__all__ = ["EffectScalesV111", "derive_effect_scales_v111", "hydraulic_effect_loss_v111"]
