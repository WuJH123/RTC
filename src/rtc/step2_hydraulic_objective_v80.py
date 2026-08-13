"""Sparse direct counterfactual hydraulic-effect objective for Step2 V8.0."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v80_contract import DirectHydraulicEffectLossContractV80


def initial_flood_physical_v80(
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


def retained_onset_targets_v80(
    full_states: torch.Tensor,
    indices: torch.Tensor,
    *,
    initial_flood_m3s: torch.Tensor,
    epsilon_m3s: float,
) -> torch.Tensor:
    """True dry-to-flood transition over each retained multi-resolution interval."""
    if full_states.ndim != 5:
        raise ValueError("V8 onset states must be [B,C,H,N,S]")
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
        result.append(transition[:, :, previous + 1 : current_idx + 1].any(dim=2))
        previous = current_idx
    return torch.stack(result, dim=2).to(full_states.dtype)


def derive_onset_sqrt_positive_weight_v80(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    normalization: InputNormalizationV60,
    *,
    contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
) -> float:
    """TrainFit-only square-root class correction; milder than V7's capped inverse prevalence."""
    contract.validate()
    positive = total = 0
    for name in fit_names:
        entry, arrays, ref = cache.entry(name), cache.entry(name).arrays, cache.entry(name).reference_index
        candidates = [i for i in entry.indices if i != ref]
        current_norm = np.asarray(arrays["initial_state"][ref], dtype=np.float64)[..., 2]
        current = (
            current_norm * max(float(normalization.state_std[2]), 1e-6)
            + float(normalization.state_mean[2])
        ) > contract.onset_epsilon_m3s
        future = np.asarray(arrays["target_states"][candidates], dtype=np.float64)[..., 2] > contract.onset_epsilon_m3s
        transition = np.zeros_like(future, dtype=bool)
        transition[:, 0] = future[:, 0] & ~current[None]
        transition[:, 1:] = future[:, 1:] & ~future[:, :-1]
        positive += int(transition.sum())
        total += int(transition.size)
    negative = max(total - positive, 0)
    weight = np.sqrt(negative / max(positive, 1))
    return float(min(max(weight, 1.0), contract.onset_max_sqrt_pos_weight))


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
        pos_weight=torch.as_tensor(float(pos_weight), dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    return (1.0 - pt).clamp_min(1e-6).pow(float(gamma)) * bce


def _effect_terms(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    scale: torch.Tensor,
    *,
    active_fraction: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense regularizer, active-effect error, response magnitude, active sign, active share."""
    scaled_pred = predicted / scale
    scaled_truth = truth.detach() / scale
    element_error = F.smooth_l1_loss(
        scaled_pred,
        scaled_truth,
        reduction="none",
        beta=float(beta),
    )
    dense = element_error.mean()
    active = scaled_truth.abs() >= float(active_fraction)
    active_loss = element_error[active].mean() if active.any() else dense.new_zeros(())

    reduce_dims = tuple(range(2, predicted.ndim))
    pred_rms = torch.sqrt(torch.mean(torch.square(scaled_pred), dim=reduce_dims) + 1e-12)
    true_rms = torch.sqrt(torch.mean(torch.square(scaled_truth), dim=reduce_dims) + 1e-12)
    magnitude = F.smooth_l1_loss(
        torch.log1p(pred_rms),
        torch.log1p(true_rms),
        beta=float(beta),
    )
    sign = (
        F.softplus(-torch.sign(scaled_truth[active]) * scaled_pred[active]).mean()
        if active.any()
        else dense.new_zeros(())
    )
    return dense, active_loss, magnitude, sign, active.float().mean()


def hydraulic_effect_loss_v80(
    output: Any,
    batch: V60GroupBatch,
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    *,
    onset_positive_weight: float,
    contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train direct signed effects; absolute reference-model error is deliberately excluded."""
    contract.validate()
    idx = output.horizon_indices
    true_ref = batch.true_reference_states.index_select(1, idx)[:, None].expand_as(
        output.reference_states_physical
    )
    true_cand = batch.true_candidate_states.index_select(2, idx)
    true_delta = true_cand - true_ref
    true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None].expand_as(
        output.reference_flows_physical
    )
    true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
    true_flow_delta = true_cand_flow - true_ref_flow

    state_scale = torch.as_tensor(
        scales.state_delta_scale, dtype=true_delta.dtype, device=true_delta.device
    ).clamp_min(1e-6)
    channel_weights = {0: 1.0, 2: 2.0, 3: 2.0, 4: 0.5, 5: 0.5}
    state_parts: list[tuple[float, tuple[torch.Tensor, ...]]] = []
    for channel, channel_weight in channel_weights.items():
        state_parts.append(
            (
                channel_weight,
                _effect_terms(
                    output.delta_states_physical[..., channel],
                    true_delta[..., channel],
                    state_scale[channel],
                    active_fraction=contract.active_effect_fraction,
                    beta=contract.smooth_l1_beta,
                ),
            )
        )
    state_weight = sum(weight for weight, _ in state_parts)
    dense_state = sum(weight * terms[0] for weight, terms in state_parts) / state_weight
    active_state = sum(weight * terms[1] for weight, terms in state_parts) / state_weight
    magnitude_state = sum(weight * terms[2] for weight, terms in state_parts) / state_weight
    sign_state = sum(weight * terms[3] for weight, terms in state_parts) / state_weight
    active_state_fraction = sum(weight * terms[4] for weight, terms in state_parts) / state_weight

    flow_scale = torch.as_tensor(
        scales.flow_delta_scale,
        dtype=true_flow_delta.dtype,
        device=true_flow_delta.device,
    ).reshape(1, 1, 1, -1).clamp_min(1e-6)
    dense_flow, active_flow, magnitude_flow, sign_flow, active_flow_fraction = _effect_terms(
        output.delta_flows_physical,
        true_flow_delta,
        flow_scale,
        active_fraction=contract.active_effect_fraction,
        beta=contract.smooth_l1_beta,
    )

    initial_flood = initial_flood_physical_v80(batch, normalization)
    ref_full = batch.true_reference_states[:, None].expand(
        batch.true_candidate_states.shape[0],
        batch.true_candidate_states.shape[1],
        -1,
        -1,
        -1,
    )
    ref_target = retained_onset_targets_v80(
        ref_full,
        idx,
        initial_flood_m3s=initial_flood,
        epsilon_m3s=contract.onset_epsilon_m3s,
    )
    cand_target = retained_onset_targets_v80(
        batch.true_candidate_states,
        idx,
        initial_flood_m3s=initial_flood,
        epsilon_m3s=contract.onset_epsilon_m3s,
    )
    onset = 0.5 * (
        _focal_bce(
            output.reference_flood_onset_logits,
            ref_target,
            pos_weight=onset_positive_weight,
            gamma=contract.onset_gamma,
        ).mean()
        + _focal_bce(
            output.candidate_flood_onset_logits,
            cand_target,
            pos_weight=onset_positive_weight,
            gamma=contract.onset_gamma,
        ).mean()
    )

    total = (
        contract.dense_state_weight * dense_state
        + contract.active_state_weight * active_state
        + contract.magnitude_state_weight * magnitude_state
        + contract.sign_state_weight * sign_state
        + contract.dense_flow_weight * dense_flow
        + contract.active_flow_weight * active_flow
        + contract.magnitude_flow_weight * magnitude_flow
        + contract.sign_flow_weight * sign_flow
        + contract.onset_weight * onset
    )
    return total, {
        "loss": float(total.detach()),
        "dense_state": float(dense_state.detach()),
        "active_state": float(active_state.detach()),
        "magnitude_state": float(magnitude_state.detach()),
        "sign_state": float(sign_state.detach()),
        "dense_flow": float(dense_flow.detach()),
        "active_flow": float(active_flow.detach()),
        "magnitude_flow": float(magnitude_flow.detach()),
        "sign_flow": float(sign_flow.detach()),
        "onset": float(onset.detach()),
        "active_state_fraction": float(active_state_fraction.detach()),
        "active_flow_fraction": float(active_flow_fraction.detach()),
    }


__all__ = [
    "derive_onset_sqrt_positive_weight_v80",
    "hydraulic_effect_loss_v80",
    "initial_flood_physical_v80",
    "retained_onset_targets_v80",
]
