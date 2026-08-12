"""Training and evidence contracts for Project7 Step2 V7.0."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import (
    ControlValueSurrogateV70,
    DirectValueOutputV70,
    HydraulicResponseSurrogateV70,
)
from .step2_train_response_v60 import (
    InputNormalizationV60,
    TargetScalesV60,
    V60GroupBatch,
    V60TrainCache,
    derive_target_scales_v60,
    hydraulic_critical_weights_v60,
)
from .step2_v60_contract import MultiResolutionHorizonV60
from .step2_v70_contract import (
    DirectValueLossContractV70,
    HydraulicEffectLossContractV70,
)


@dataclass(frozen=True)
class TargetScalesV70:
    base: TargetScalesV60
    direct_tfv_scale_m3: float
    state_delta_scale: np.ndarray
    flow_delta_scale: np.ndarray
    onset_positive_weight: float


def derive_target_scales_v70(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    *,
    horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(),
    onset_epsilon_m3s: float = 1e-7,
) -> TargetScalesV70:
    """Derive all V7 scales from TrainFit only."""
    if not fit_names:
        raise ValueError("V7 target scales need TrainFit groups")
    base = derive_target_scales_v60(cache, fit_names, horizon=horizon)
    idx = np.asarray(horizon.indices(), dtype=np.int64)
    d3_abs: list[float] = []
    state_sq = None
    flow_sq = None
    state_count = flow_count = 0
    onset_pos = onset_total = 0

    for name in fit_names:
        entry, arrays, ref = cache.entry(name), cache.entry(name).arrays, cache.entry(name).reference_index
        candidates = [i for i in entry.indices if i != ref]
        ref_tfv = float(np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum())
        cand_tfv = np.asarray(arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64).sum(axis=1)
        if entry.source_kind.upper() == "D3":
            d3_abs.extend(np.abs(cand_tfv - ref_tfv).tolist())

        ref_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)
        cand_state = np.asarray(arrays["target_states"][candidates], dtype=np.float64)
        delta_state = cand_state[:, idx] - ref_state[None, idx]
        sq = np.square(delta_state).reshape(-1, delta_state.shape[-1]).sum(axis=0)
        state_sq = sq if state_sq is None else state_sq + sq
        state_count += int(np.prod(delta_state.shape[:-1]))

        ref_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)
        cand_flow = np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64)
        delta_flow = cand_flow[:, idx] - ref_flow[None, idx]
        fsq = np.square(delta_flow).reshape(-1, delta_flow.shape[-1]).sum(axis=0)
        flow_sq = fsq if flow_sq is None else flow_sq + fsq
        flow_count += int(np.prod(delta_flow.shape[:-1]))

        flood = cand_state[..., 2] > float(onset_epsilon_m3s)
        transition = flood[:, 1:] & ~flood[:, :-1]
        onset_pos += int(transition.sum())
        onset_total += int(transition.size)

    absolute = np.asarray(d3_abs, dtype=np.float64)
    if absolute.size == 0 or not np.isfinite(absolute).all():
        raise ValueError("V7 direct TFV scale requires targeted TrainFit D3")
    positive = absolute[absolute > 1e-9]
    robust = float(np.quantile(positive if positive.size else absolute, 0.75))
    direct_scale = max(robust, 100.0)
    prevalence = onset_pos / max(onset_total, 1)
    pos_weight = min(max((1.0 - prevalence) / max(prevalence, 1e-9), 1.0), 50.0)
    return TargetScalesV70(
        base=base,
        direct_tfv_scale_m3=direct_scale,
        state_delta_scale=np.maximum(
            np.sqrt(state_sq / max(state_count, 1)).astype(np.float32), 1e-6
        ),
        flow_delta_scale=np.maximum(
            np.sqrt(flow_sq / max(flow_count, 1)).astype(np.float32), 1e-6
        ),
        onset_positive_weight=float(pos_weight),
    )


def _pairwise_losses(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    scale: torch.Tensor,
    *,
    min_effect_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError("V7 pairwise loss expects [B,C]")
    p = (predicted[:, :, None] - predicted[:, None, :]) / scale
    t = (truth.detach()[:, :, None] - truth.detach()[:, None, :]) / scale
    c = predicted.shape[1]
    upper = torch.triu(
        torch.ones(c, c, dtype=torch.bool, device=predicted.device), diagonal=1
    )[None]
    informative = upper & (t.abs() >= float(min_effect_fraction))
    if not informative.any():
        informative = upper
    mag = F.smooth_l1_loss(
        p[informative], t[informative], reduction="mean", beta=0.5
    )
    sign = F.softplus(-torch.sign(t[informative]) * p[informative]).mean()
    return mag, sign


def value_loss_v70(
    output: DirectValueOutputV70,
    truth: torch.Tensor,
    *,
    scale_m3: float,
    contract: DirectValueLossContractV70 = DirectValueLossContractV70(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Magnitude-calibrated objective; ordering can no longer hide scale collapse."""
    contract.validate()
    predicted = output.delta_tfv_m3
    if predicted.shape != truth.shape:
        raise ValueError("V7 value prediction/target shape mismatch")
    scale = torch.as_tensor(
        max(float(scale_m3), 1.0), dtype=predicted.dtype, device=predicted.device
    )
    transformed_truth = torch.asinh(truth.detach() / scale)
    transformed = F.smooth_l1_loss(
        output.normalized_delta_tfv, transformed_truth, beta=0.5
    )
    physical = F.smooth_l1_loss(
        (predicted - truth.detach()) / scale,
        torch.zeros_like(predicted),
        beta=0.5,
    )
    pair_mag, pair_sign = _pairwise_losses(
        predicted,
        truth,
        scale,
        min_effect_fraction=contract.pair_min_effect_fraction,
    )
    total = (
        contract.transformed_magnitude_weight * transformed
        + contract.physical_magnitude_weight * physical
        + contract.pairwise_difference_weight * pair_mag
        + contract.pairwise_sign_weight * pair_sign
    )
    return total, {
        "loss": float(total.detach()),
        "transformed_magnitude": float(transformed.detach()),
        "physical_magnitude": float(physical.detach()),
        "pairwise_difference": float(pair_mag.detach()),
        "pairwise_sign": float(pair_sign.detach()),
    }


def _retained_onset_targets(
    full_states: torch.Tensor,
    indices: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Whether a dry->flood transition occurred since the previous retained point."""
    flood = full_states[..., 2] > float(epsilon)
    transition = torch.zeros_like(flood)
    transition[..., 0, :] = flood[..., 0, :]
    transition[..., 1:, :] = flood[..., 1:, :] & ~flood[..., :-1, :]
    result: list[torch.Tensor] = []
    previous = -1
    for raw in indices.detach().cpu().tolist():
        current = int(raw)
        result.append(transition[..., previous + 1 : current + 1, :].any(dim=-2))
        previous = current
    return torch.stack(result, dim=-2).to(full_states.dtype)


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


def hydraulic_loss_v70(
    output,
    batch: V60GroupBatch,
    prepared: PreparedStaticV60,
    scales: TargetScalesV70,
    *,
    horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(),
    contract: HydraulicEffectLossContractV70 = HydraulicEffectLossContractV70(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Absolute-reference fidelity plus explicit counterfactual hydraulic-effect fidelity."""
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

    channels = (0, 2, 3, 4, 5)
    abs_losses = []
    delta_losses = []
    true_delta = true_cand - true_ref
    for channel in channels:
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
        abs_losses.append(0.5 * ((ref_err * weight).mean() + (cand_err * weight).mean()))
        d_err = F.smooth_l1_loss(
            output.delta_states_physical[..., channel] / delta_scale[channel],
            true_delta[..., channel] / delta_scale[channel],
            reduction="none",
            beta=0.5,
        )
        delta_losses.append((d_err * weight).mean())
    absolute_state = torch.stack(abs_losses).mean()
    delta_state = torch.stack(delta_losses).mean()

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

    ref_full = batch.true_reference_states[:, None].expand(
        batch.true_candidate_states.shape[0],
        batch.true_candidate_states.shape[1],
        -1,
        -1,
        -1,
    )
    cand_full = batch.true_candidate_states
    ref_onset = _retained_onset_targets(
        ref_full, idx, epsilon=contract.onset_epsilon_m3s
    )
    cand_onset = _retained_onset_targets(
        cand_full, idx, epsilon=contract.onset_epsilon_m3s
    )
    onset_raw = 0.5 * (
        _focal_bce(
            output.reference_flood_onset_logits,
            ref_onset,
            pos_weight=scales.onset_positive_weight,
            gamma=contract.onset_gamma,
        )
        + _focal_bce(
            output.candidate_flood_onset_logits,
            cand_onset,
            pos_weight=scales.onset_positive_weight,
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


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(np.float64)


def _spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(pred) < 2 or np.allclose(pred, pred[0]) or np.allclose(truth, truth[0]):
        return float("nan")
    return float(np.corrcoef(_rank(pred), _rank(truth))[0, 1])


def _pairwise(pred: np.ndarray, truth: np.ndarray) -> float:
    total = correct = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if abs(float(truth[i] - truth[j])) <= 1e-9:
                continue
            total += 1
            correct += int(
                np.sign(pred[i] - pred[j]) == np.sign(truth[i] - truth[j])
            )
    return float(correct / total) if total else float("nan")


def evaluate_value_v70(
    model: ControlValueSurrogateV70,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    target = torch.device(device)
    model.to(target).eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target)
            out = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
            )
            pred = out.delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            entry = cache.entry(name)
            best_pred, best_true = int(np.argmin(pred)), int(np.argmin(truth))
            nonzero = np.abs(truth) > 1e-9
            records.append(
                {
                    "event": f"{entry.rainfall_group}::{entry.event_id}",
                    "rank": _spearman(pred, truth),
                    "pairwise": _pairwise(pred, truth),
                    "sign": float(np.mean(np.sign(pred[nonzero]) == np.sign(truth[nonzero])))
                    if nonzero.any()
                    else float("nan"),
                    "top1": float(best_pred == best_true),
                    "regret": float(truth[best_pred] - truth[best_true]),
                    "mae": float(np.mean(np.abs(pred - truth))),
                    "truth_spread": float(truth.max() - truth.min()),
                    "pred_spread": float(pred.max() - pred.min()),
                    "truth_abs": float(np.mean(np.abs(truth))),
                    "pred_abs": float(np.mean(np.abs(pred))),
                }
            )

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_event[record["event"]].append(record)

    def event_mean(key: str) -> float:
        values = []
        for group_records in by_event.values():
            raw = [float(r[key]) for r in group_records if np.isfinite(float(r[key]))]
            if raw:
                values.append(float(np.mean(raw)))
        return float(np.mean(values)) if values else float("nan")

    result = {
        "groups": len(records),
        "events": len(by_event),
        "rank": event_mean("rank"),
        "pairwise": event_mean("pairwise"),
        "sign_accuracy": event_mean("sign"),
        "top1_rate": event_mean("top1"),
        "mean_regret_m3": event_mean("regret"),
        "max_regret_m3": max((r["regret"] for r in records), default=float("nan")),
        "tfv_mae_m3": event_mean("mae"),
        "truth_spread_m3": event_mean("truth_spread"),
        "predicted_spread_m3": event_mean("pred_spread"),
        "mean_abs_truth_m3": event_mean("truth_abs"),
        "mean_abs_prediction_m3": event_mean("pred_abs"),
        "scientific_primary": "event_balanced",
    }
    result["spread_ratio"] = result["predicted_spread_m3"] / max(
        result["truth_spread_m3"], 1e-6
    )
    result["response_ratio"] = result["mean_abs_prediction_m3"] / max(
        result["mean_abs_truth_m3"], 1e-6
    )
    result["response_collapse"] = result["spread_ratio"] < 1e-3
    return result


def evaluate_value_strata_v70(
    model: ControlValueSurrogateV70,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    *,
    device: torch.device | str,
    q33_m3: float,
    q67_m3: float,
) -> dict[str, Any]:
    target = torch.device(device)
    model.to(target).eval()
    buckets: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target)
            out = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
            )
            pred = out.delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            event = f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}"
            absolute = np.abs(truth)
            masks = {
                "small": absolute < float(q33_m3),
                "medium": (absolute >= float(q33_m3)) & (absolute < float(q67_m3)),
                "large": absolute >= float(q67_m3),
            }
            for label, mask in masks.items():
                if not mask.any():
                    continue
                p, t = pred[mask], truth[mask]
                buckets[label].append(
                    {
                        "event": event,
                        "count": int(mask.sum()),
                        "rank": _spearman(p, t),
                        "pairwise": _pairwise(p, t),
                        "mae": float(np.mean(np.abs(p - t))),
                        "truth_abs": float(np.mean(np.abs(t))),
                        "pred_abs": float(np.mean(np.abs(p))),
                        "truth_spread": float(t.max() - t.min()) if len(t) > 1 else 0.0,
                        "pred_spread": float(p.max() - p.min()) if len(p) > 1 else 0.0,
                    }
                )
    result: dict[str, Any] = {}
    for label, rows in buckets.items():
        by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_event[row["event"]].append(row)
        def evmean(key: str) -> float:
            event_values = []
            for items in by_event.values():
                vals = [float(x[key]) for x in items if np.isfinite(float(x[key]))]
                if vals:
                    event_values.append(float(np.mean(vals)))
            return float(np.mean(event_values)) if event_values else float("nan")
        truth_abs = evmean("truth_abs")
        pred_abs = evmean("pred_abs")
        truth_spread = evmean("truth_spread")
        pred_spread = evmean("pred_spread")
        result[label] = {
            "candidate_count": int(sum(x["count"] for x in rows)),
            "events": int(len(by_event)),
            "rank": evmean("rank"),
            "pairwise": evmean("pairwise"),
            "tfv_mae_m3": evmean("mae"),
            "response_ratio": pred_abs / max(truth_abs, 1e-6),
            "spread_ratio": pred_spread / max(truth_spread, 1e-6),
        }
    return result


def evaluate_onset_transition_v70(
    model: HydraulicResponseSurrogateV70,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    *,
    device: torch.device | str,
    epsilon: float = 1e-7,
) -> dict[str, float]:
    target_device = torch.device(device)
    model.to(target_device).eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target_device)
            out = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                prepared,
            )
            target = _retained_onset_targets(
                batch.true_candidate_states,
                out.horizon_indices,
                epsilon=epsilon,
            ).bool()
            pred = out.candidate_flood_onset_logits > 0
            tp += int((pred & target).sum())
            tn += int((~pred & ~target).sum())
            fp += int((pred & ~target).sum())
            fn += int((~pred & target).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "balanced_accuracy": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "positive_count": tp + fn,
        "negative_count": tn + fp,
    }


__all__ = [
    "TargetScalesV70",
    "derive_target_scales_v70",
    "evaluate_onset_transition_v70",
    "evaluate_value_v70",
    "evaluate_value_strata_v70",
    "hydraulic_loss_v70",
    "value_loss_v70",
]
