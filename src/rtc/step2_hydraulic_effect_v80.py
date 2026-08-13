"""Sparse counterfactual hydraulic-effect training/evaluation for Step2 V8.0."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v80 import DirectHydraulicEffectSurrogateV80, PreparedStaticV80
from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v70_contract import MultiResolutionHorizonV60
from .step2_v80_contract import DirectHydraulicEffectLossContractV80


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


def retained_onset_targets_v80(
    full_states: torch.Tensor,
    indices: torch.Tensor,
    *,
    initial_flood_m3s: torch.Tensor,
    epsilon_m3s: float,
) -> torch.Tensor:
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
    """Data-derived square-root imbalance correction from TrainFit only."""
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
    ratio = np.sqrt(negative / max(positive, 1))
    return float(min(max(ratio, 1.0), contract.onset_max_sqrt_pos_weight))


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
    """Dense regularizer + active effect + group magnitude + active sign + active fraction."""
    scaled_pred = predicted / scale
    scaled_truth = truth.detach() / scale
    error = F.smooth_l1_loss(
        scaled_pred,
        scaled_truth,
        reduction="none",
        beta=float(beta),
    )
    dense = error.mean()
    active = scaled_truth.abs() >= float(active_fraction)
    active_loss = error[active].mean() if active.any() else dense.new_zeros(())

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
    """Primary supervision is candidate-minus-reference, with sparse active-effect emphasis."""
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
        terms = _effect_terms(
            output.delta_states_physical[..., channel],
            true_delta[..., channel],
            state_scale[channel],
            active_fraction=contract.active_effect_fraction,
            beta=contract.smooth_l1_beta,
        )
        state_parts.append((channel_weight, terms))

    weight_sum = sum(weight for weight, _ in state_parts)
    dense_state = sum(weight * terms[0] for weight, terms in state_parts) / weight_sum
    active_state = sum(weight * terms[1] for weight, terms in state_parts) / weight_sum
    magnitude_state = sum(weight * terms[2] for weight, terms in state_parts) / weight_sum
    sign_state = sum(weight * terms[3] for weight, terms in state_parts) / weight_sum
    active_state_fraction = sum(weight * terms[4] for weight, terms in state_parts) / weight_sum

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

    initial_flood = _initial_flood_physical(batch, normalization)
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


def _event_balanced(records: Sequence[dict[str, Any]], key: str) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            by_event[str(record["event"])].append(value)
    values = [float(np.mean(items)) for items in by_event.values() if items]
    return float(np.mean(values)) if values else float("nan")


def _effect_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    scale: float,
    active_fraction: float,
    prefix: str,
) -> dict[str, float]:
    error = predicted - truth
    mse = float(np.mean(np.square(error)))
    zero_mse = float(np.mean(np.square(truth)))
    response = float(np.mean(np.abs(predicted))) / max(float(np.mean(np.abs(truth))), 1e-12)
    active = np.abs(truth) >= float(active_fraction) * max(float(scale), 1e-12)
    active_mse = float(np.mean(np.square(error[active]))) if active.any() else float("nan")
    active_zero_mse = float(np.mean(np.square(truth[active]))) if active.any() else float("nan")
    active_sign = (
        float(np.mean(np.sign(predicted[active]) == np.sign(truth[active])))
        if active.any()
        else float("nan")
    )
    return {
        f"{prefix}_rmse": float(np.sqrt(mse)),
        f"{prefix}_response_ratio": response,
        f"{prefix}_skill_vs_zero": 1.0 - mse / max(zero_mse, 1e-12),
        f"{prefix}_active_skill_vs_zero": (
            1.0 - active_mse / max(active_zero_mse, 1e-12)
            if np.isfinite(active_mse)
            else float("nan")
        ),
        f"{prefix}_active_sign": active_sign,
        f"{prefix}_active_fraction": float(active.mean()),
    }


def evaluate_hydraulic_effect_v80(
    model: DirectHydraulicEffectSurrogateV80,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV80,
    scales: TargetScalesV70,
    *,
    device: torch.device | str,
    contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
) -> dict[str, Any]:
    target_device = torch.device(device)
    model.to(target_device).eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target_device)
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
            )
            idx = output.horizon_indices
            true_ref = batch.true_reference_states.index_select(1, idx)[:, None].expand_as(
                output.reference_states_physical
            )
            true_cand = batch.true_candidate_states.index_select(2, idx)
            true_delta = (true_cand - true_ref).detach().cpu().numpy().astype(np.float64)
            pred_delta = output.delta_states_physical.detach().cpu().numpy().astype(np.float64)
            true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None].expand_as(
                output.reference_flows_physical
            )
            true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
            true_flow = (true_cand_flow - true_ref_flow).detach().cpu().numpy().astype(np.float64)
            pred_flow = output.delta_flows_physical.detach().cpu().numpy().astype(np.float64)
            event = f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}"
            record: dict[str, Any] = {"event": event}
            for channel, label in ((0, "depth_m"), (2, "flood_m3s"), (3, "storage_m3")):
                record.update(
                    _effect_record(
                        pred_delta[..., channel],
                        true_delta[..., channel],
                        scale=float(scales.state_delta_scale[channel]),
                        active_fraction=contract.active_effect_fraction,
                        prefix=f"delta_{label}",
                    )
                )
            flow_scale = float(np.median(scales.flow_delta_scale))
            record.update(
                _effect_record(
                    pred_flow,
                    true_flow,
                    scale=flow_scale,
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                )
            )
            records.append(record)

    metric_names = sorted({key for record in records for key in record if key != "event"})
    return {
        "groups": int(len(records)),
        "events": int(len({record["event"] for record in records})),
        "scientific_primary": "event_balanced_direct_candidate_minus_reference_effect",
        **{name: _event_balanced(records, name) for name in metric_names},
    }


def _onset_histogram(
    model: DirectHydraulicEffectSurrogateV80,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV80,
    *,
    device: torch.device | str,
    contract: DirectHydraulicEffectLossContractV80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(
        contract.onset_logit_min,
        contract.onset_logit_max,
        contract.onset_threshold_bins + 1,
        dtype=np.float64,
    )
    positive = np.zeros(contract.onset_threshold_bins, dtype=np.int64)
    negative = np.zeros(contract.onset_threshold_bins, dtype=np.int64)
    target_device = torch.device(device)
    model.to(target_device).eval()
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target_device)
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
            )
            target = retained_onset_targets_v80(
                batch.true_candidate_states,
                output.horizon_indices,
                initial_flood_m3s=_initial_flood_physical(batch, normalization),
                epsilon_m3s=contract.onset_epsilon_m3s,
            ).bool().cpu().numpy().reshape(-1)
            logits = output.candidate_flood_onset_logits.detach().cpu().numpy().reshape(-1)
            logits = np.clip(logits, edges[0] + 1e-12, edges[-1] - 1e-12)
            positive += np.histogram(logits[target], bins=edges)[0]
            negative += np.histogram(logits[~target], bins=edges)[0]
    return edges, positive, negative


def calibrate_onset_threshold_v80(
    model: DirectHydraulicEffectSurrogateV80,
    cache: V60TrainCache,
    fit_names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV80,
    *,
    device: torch.device | str,
    contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
) -> dict[str, float]:
    """TrainFit-only threshold calibration by binned F1; Holdout never selects threshold."""
    edges, positive, negative = _onset_histogram(
        model, cache, fit_names, normalization, prepared, device=device, contract=contract
    )
    tp = np.cumsum(positive[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(negative[::-1])[::-1].astype(np.float64)
    total_positive = float(positive.sum())
    fn = total_positive - tp
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(total_positive, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    prevalence = total_positive / max(float(positive.sum() + negative.sum()), 1.0)
    return {
        "threshold_logit": float(edges[best]),
        "fit_f1": float(f1[best]),
        "fit_precision": float(precision[best]),
        "fit_recall": float(recall[best]),
        "fit_prevalence": float(prevalence),
        "fit_precision_lift": float(precision[best] / max(prevalence, 1e-12)),
    }


def evaluate_onset_threshold_v80(
    model: DirectHydraulicEffectSurrogateV80,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV80,
    *,
    threshold_logit: float,
    device: torch.device | str,
    contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
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
                batch.previous_actuator_flow,
                prepared,
            )
            target = retained_onset_targets_v80(
                batch.true_candidate_states,
                output.horizon_indices,
                initial_flood_m3s=_initial_flood_physical(batch, normalization),
                epsilon_m3s=contract.onset_epsilon_m3s,
            ).bool()
            predicted = output.candidate_flood_onset_logits >= float(threshold_logit)
            tp += int((predicted & target).sum())
            tn += int((~predicted & ~target).sum())
            fp += int((predicted & ~target).sum())
            fn += int((~predicted & target).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    prevalence = (tp + fn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold_logit": float(threshold_logit),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "prevalence": prevalence,
        "precision_lift": precision / max(prevalence, 1e-12),
        "positive_count": int(tp + fn),
        "negative_count": int(tn + fp),
    }


__all__ = [
    "calibrate_onset_threshold_v80",
    "derive_onset_sqrt_positive_weight_v80",
    "evaluate_hydraulic_effect_v80",
    "evaluate_onset_threshold_v80",
    "hydraulic_effect_loss_v80",
    "retained_onset_targets_v80",
]
