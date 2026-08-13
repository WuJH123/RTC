"""Effect-aware hydraulic supervision for Project7 Step2 V7.0."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import HydraulicResponseSurrogateV70
from .step2_train_response_v60 import (
    InputNormalizationV60,
    V60GroupBatch,
    V60TrainCache,
    hydraulic_critical_weights_v60,
)
from .step2_train_response_v70 import TargetScalesV70
from .step2_v60_contract import MultiResolutionHorizonV60
from .step2_v70_contract import HydraulicEffectLossContractV70


def derive_onset_positive_weight_v70(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    *,
    epsilon_m3s: float = 1e-7,
    cap: float = 50.0,
) -> float:
    """Class imbalance from true dry->flood transitions, TrainFit only."""
    positive = total = 0
    for name in fit_names:
        entry, arrays, ref = cache.entry(name), cache.entry(name).arrays, cache.entry(name).reference_index
        candidates = [i for i in entry.indices if i != ref]
        current = np.asarray(arrays["initial_state"][ref], dtype=np.float64)[..., 2] > epsilon_m3s
        future = np.asarray(arrays["target_states"][candidates], dtype=np.float64)[..., 2] > epsilon_m3s
        transition = np.zeros_like(future, dtype=bool)
        transition[:, 0] = future[:, 0] & ~current[None]
        transition[:, 1:] = future[:, 1:] & ~future[:, :-1]
        positive += int(transition.sum())
        total += int(transition.size)
    prevalence = positive / max(total, 1)
    return float(min(max((1.0 - prevalence) / max(prevalence, 1e-12), 1.0), cap))


def _initial_flood_physical(
    batch: V60GroupBatch, normalization: InputNormalizationV60
) -> torch.Tensor:
    mean = torch.as_tensor(
        float(normalization.state_mean[2]),
        dtype=batch.initial_state.dtype,
        device=batch.initial_state.device,
    )
    std = torch.as_tensor(
        max(float(normalization.state_std[2]), 1e-6),
        dtype=batch.initial_state.dtype,
        device=batch.initial_state.device,
    )
    return batch.initial_state[..., 2] * std + mean


def retained_onset_targets_v70(
    full_states: torch.Tensor,
    indices: torch.Tensor,
    *,
    initial_flood_m3s: torch.Tensor,
    epsilon_m3s: float,
) -> torch.Tensor:
    """True transition target over each retained multi-resolution interval."""
    if full_states.ndim != 5:
        raise ValueError("V7 onset states must be [B,C,H,N,S]")
    flood = full_states[..., 2] > float(epsilon_m3s)
    current = initial_flood_m3s > float(epsilon_m3s)
    current = current[:, None].expand(full_states.shape[0], full_states.shape[1], -1)
    transition = torch.zeros_like(flood)
    transition[:, :, 0] = flood[:, :, 0] & ~current
    transition[:, :, 1:] = flood[:, :, 1:] & ~flood[:, :, :-1]
    result: list[torch.Tensor] = []
    previous = -1
    for raw in indices.detach().cpu().tolist():
        current_idx = int(raw)
        result.append(
            transition[:, :, previous + 1 : current_idx + 1].any(dim=2)
        )
        previous = current_idx
    return torch.stack(result, dim=2).to(full_states.dtype)


def _focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float,
    gamma: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    pt = target * probability + (1.0 - target) * (1.0 - probability)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.as_tensor(
            float(pos_weight), dtype=logits.dtype, device=logits.device
        ),
        reduction="none",
    )
    return torch.pow((1.0 - pt).clamp_min(1e-6), float(gamma)) * bce


def hydraulic_effect_loss_v70(
    output: Any,
    batch: V60GroupBatch,
    prepared: PreparedStaticV60,
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    *,
    onset_positive_weight: float,
    horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(),
    contract: HydraulicEffectLossContractV70 = HydraulicEffectLossContractV70(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reference fidelity + explicit action-effect fidelity + true onset transition."""
    contract.validate()
    idx = output.horizon_indices
    true_ref = batch.true_reference_states.index_select(1, idx)[:, None].expand_as(
        output.reference_states_physical
    )
    true_cand = batch.true_candidate_states.index_select(2, idx)
    true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None].expand_as(
        output.reference_flows_physical
    )
    true_cand_flow = batch.true_candidate_flows.index_select(2, idx)

    weights = hydraulic_critical_weights_v60(true_cand, prepared)
    time_weight = torch.as_tensor(
        horizon.weights(), dtype=weights.dtype, device=weights.device
    ).reshape(1, 1, -1, 1)
    weight = weights * time_weight
    state_scale = torch.as_tensor(
        scales.base.state_scale, dtype=weights.dtype, device=weights.device
    )
    delta_scale = torch.as_tensor(
        scales.state_delta_scale, dtype=weights.dtype, device=weights.device
    )
    flow_scale = torch.as_tensor(
        scales.base.flow_scale, dtype=weights.dtype, device=weights.device
    )
    flow_delta_scale = torch.as_tensor(
        scales.flow_delta_scale, dtype=weights.dtype, device=weights.device
    )

    true_delta = true_cand - true_ref
    absolute_terms: list[torch.Tensor] = []
    delta_terms: list[torch.Tensor] = []
    for channel in (0, 2, 3, 4, 5):
        ref_err = F.smooth_l1_loss(
            output.reference_states_physical[..., channel] / state_scale[channel],
            true_ref[..., channel] / state_scale[channel],
            reduction="none",
            beta=0.5,
        )
        cand_err = F.smooth_l1_loss(
            output.candidate_states_physical[..., channel] / state_scale[channel],
            true_cand[..., channel] / state_scale[channel],
            reduction="none",
            beta=0.5,
        )
        absolute_terms.append(
            0.5 * ((ref_err * weight).mean() + (cand_err * weight).mean())
        )
        delta_err = F.smooth_l1_loss(
            output.delta_states_physical[..., channel] / delta_scale[channel],
            true_delta[..., channel] / delta_scale[channel],
            reduction="none",
            beta=0.5,
        )
        delta_terms.append((delta_err * weight).mean())
    absolute_state = torch.stack(absolute_terms).mean()
    delta_state = torch.stack(delta_terms).mean()

    absolute_flow = 0.5 * (
        F.smooth_l1_loss(
            output.reference_flows_physical / flow_scale,
            true_ref_flow / flow_scale,
            beta=0.5,
        )
        + F.smooth_l1_loss(
            output.candidate_flows_physical / flow_scale,
            true_cand_flow / flow_scale,
            beta=0.5,
        )
    )
    delta_flow = F.smooth_l1_loss(
        output.delta_flows_physical / flow_delta_scale,
        (true_cand_flow - true_ref_flow) / flow_delta_scale,
        beta=0.5,
    )

    initial_flood = _initial_flood_physical(batch, normalization)
    ref_full = batch.true_reference_states[:, None].expand(
        batch.true_candidate_states.shape[0],
        batch.true_candidate_states.shape[1],
        -1,
        -1,
        -1,
    )
    ref_target = retained_onset_targets_v70(
        ref_full,
        idx,
        initial_flood_m3s=initial_flood,
        epsilon_m3s=contract.onset_epsilon_m3s,
    )
    cand_target = retained_onset_targets_v70(
        batch.true_candidate_states,
        idx,
        initial_flood_m3s=initial_flood,
        epsilon_m3s=contract.onset_epsilon_m3s,
    )
    onset_raw = 0.5 * (
        _focal_bce(
            output.reference_flood_onset_logits,
            ref_target,
            pos_weight=onset_positive_weight,
            gamma=contract.onset_gamma,
        )
        + _focal_bce(
            output.candidate_flood_onset_logits,
            cand_target,
            pos_weight=onset_positive_weight,
            gamma=contract.onset_gamma,
        )
    )
    onset = (onset_raw * weight).mean()

    total = (
        contract.absolute_state_weight * absolute_state
        + contract.delta_state_weight * delta_state
        + contract.absolute_flow_weight * absolute_flow
        + contract.delta_flow_weight * delta_flow
        + contract.onset_transition_weight * onset
    )
    return total, {
        "loss": float(total.detach()),
        "absolute_state": float(absolute_state.detach()),
        "delta_state": float(delta_state.detach()),
        "absolute_flow": float(absolute_flow.detach()),
        "delta_flow": float(delta_flow.detach()),
        "onset_transition": float(onset.detach()),
    }


def evaluate_onset_transition_v70(
    model: HydraulicResponseSurrogateV70,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    *,
    device: torch.device | str,
    epsilon_m3s: float = 1e-7,
) -> dict[str, float]:
    target_device = torch.device(device)
    model.to(target_device).eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target_device)
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                prepared,
            )
            target = retained_onset_targets_v70(
                batch.true_candidate_states,
                output.horizon_indices,
                initial_flood_m3s=_initial_flood_physical(batch, normalization),
                epsilon_m3s=epsilon_m3s,
            ).bool()
            predicted = output.candidate_flood_onset_logits > 0
            tp += int((predicted & target).sum())
            tn += int((~predicted & ~target).sum())
            fp += int((predicted & ~target).sum())
            fn += int((~predicted & target).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "balanced_accuracy": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "positive_count": int(tp + fn),
        "negative_count": int(tn + fp),
    }


__all__ = [
    "derive_onset_positive_weight_v70",
    "evaluate_onset_transition_v70",
    "hydraulic_effect_loss_v70",
    "retained_onset_targets_v70",
]
