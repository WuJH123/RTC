"""Event-balanced evaluation for Project7 Step2 V8.0 direct hydraulic effects."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v80 import DirectHydraulicEffectSurrogateV80, PreparedStaticV80
from .step2_hydraulic_objective_v80 import (
    initial_flood_physical_v80,
    retained_onset_targets_v80,
)
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v80_contract import DirectHydraulicEffectLossContractV80


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
    truth_abs = float(np.mean(np.abs(truth)))
    predicted_abs = float(np.mean(np.abs(predicted)))
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
        f"{prefix}_response_ratio": predicted_abs / max(truth_abs, 1e-12),
        f"{prefix}_skill_vs_zero": 1.0 - mse / max(zero_mse, 1e-12),
        f"{prefix}_active_skill_vs_zero": (
            1.0 - active_mse / max(active_zero_mse, 1e-12)
            if np.isfinite(active_mse)
            else float("nan")
        ),
        f"{prefix}_active_sign": active_sign,
        f"{prefix}_active_fraction": float(active.mean()),
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


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
            pred_cand = output.candidate_states_physical.detach().cpu().numpy().astype(np.float64)
            cand_truth = true_cand.detach().cpu().numpy().astype(np.float64)
            pred_cand_flow = output.candidate_flows_physical.detach().cpu().numpy().astype(np.float64)
            cand_flow_truth = true_cand_flow.detach().cpu().numpy().astype(np.float64)
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
            record.update(
                _effect_record(
                    pred_flow,
                    true_flow,
                    scale=float(np.median(scales.flow_delta_scale)),
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                )
            )
            record.update(
                {
                    "candidate_depth_rmse_m": _rmse(pred_cand[..., 0] - cand_truth[..., 0]),
                    "candidate_flood_rmse_m3s": _rmse(pred_cand[..., 2] - cand_truth[..., 2]),
                    "candidate_storage_rmse_m3": _rmse(pred_cand[..., 3] - cand_truth[..., 3]),
                    "candidate_managed_flow_rmse_m3s": _rmse(pred_cand_flow - cand_flow_truth),
                }
            )
            records.append(record)

    metric_names = sorted({key for record in records for key in record if key != "event"})
    metrics = {name: _event_balanced(records, name) for name in metric_names}
    primary_skills = [
        metrics.get("delta_depth_m_skill_vs_zero", float("nan")),
        metrics.get("delta_flood_m3s_skill_vs_zero", float("nan")),
        metrics.get("delta_storage_m3_skill_vs_zero", float("nan")),
        metrics.get("delta_managed_flow_m3s_skill_vs_zero", float("nan")),
    ]
    finite_skills = [value for value in primary_skills if np.isfinite(value)]
    return {
        "groups": int(len(records)),
        "events": int(len({record["event"] for record in records})),
        "scientific_primary": "event_balanced_direct_candidate_minus_reference_effect",
        **metrics,
        "all_primary_effect_skills_positive": bool(finite_skills and all(value > 0 for value in finite_skills)),
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
                initial_flood_m3s=initial_flood_physical_v80(batch, normalization),
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
    """Select the operating threshold on TrainFit only; Holdout never tunes it."""
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
                initial_flood_m3s=initial_flood_physical_v80(batch, normalization),
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
    "evaluate_hydraulic_effect_v80",
    "evaluate_onset_threshold_v80",
]
