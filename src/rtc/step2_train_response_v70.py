"""Direct objective training/evaluation helpers for Project7 Step2 V7.0.

Hydraulic-effect supervision intentionally lives only in step2_hydraulic_effect_v70.py
so V7 has one authoritative dry-to-flood transition definition.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import ControlValueSurrogateV70, DirectValueOutputV70
from .step2_train_response_v60 import (
    InputNormalizationV60,
    TargetScalesV60,
    V60TrainCache,
    derive_target_scales_v60,
)
from .step2_v60_contract import MultiResolutionHorizonV60
from .step2_v70_contract import DirectValueLossContractV70


@dataclass(frozen=True)
class TargetScalesV70:
    base: TargetScalesV60
    direct_tfv_scale_m3: float
    state_delta_scale: np.ndarray
    flow_delta_scale: np.ndarray


def derive_target_scales_v70(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    *,
    horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(),
) -> TargetScalesV70:
    """All physical scales are frozen from TrainFit only."""
    if not fit_names:
        raise ValueError("V7 target scales need TrainFit groups")
    base = derive_target_scales_v60(cache, fit_names, horizon=horizon)
    indices = np.asarray(horizon.indices(), dtype=np.int64)
    d3_abs: list[float] = []
    state_square = None
    flow_square = None
    state_count = flow_count = 0
    for name in fit_names:
        entry, arrays, ref = cache.entry(name), cache.entry(name).arrays, cache.entry(name).reference_index
        candidates = [i for i in entry.indices if i != ref]
        ref_tfv = float(np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum())
        cand_tfv = np.asarray(arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64).sum(axis=1)
        if entry.source_kind.upper() == "D3":
            d3_abs.extend(np.abs(cand_tfv - ref_tfv).tolist())

        ref_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)
        cand_state = np.asarray(arrays["target_states"][candidates], dtype=np.float64)
        state_delta = cand_state[:, indices] - ref_state[None, indices]
        square = np.square(state_delta).reshape(-1, state_delta.shape[-1]).sum(axis=0)
        state_square = square if state_square is None else state_square + square
        state_count += int(np.prod(state_delta.shape[:-1]))

        ref_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)
        cand_flow = np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64)
        flow_delta = cand_flow[:, indices] - ref_flow[None, indices]
        square = np.square(flow_delta).reshape(-1, flow_delta.shape[-1]).sum(axis=0)
        flow_square = square if flow_square is None else flow_square + square
        flow_count += int(np.prod(flow_delta.shape[:-1]))

    absolute = np.asarray(d3_abs, dtype=np.float64)
    if absolute.size == 0 or not np.isfinite(absolute).all():
        raise ValueError("V7 direct TFV scale requires targeted TrainFit D3")
    positive = absolute[absolute > 1e-9]
    robust = float(np.quantile(positive if positive.size else absolute, 0.75))
    return TargetScalesV70(
        base=base,
        direct_tfv_scale_m3=max(robust, 100.0),
        state_delta_scale=np.maximum(
            np.sqrt(state_square / max(state_count, 1)).astype(np.float32), 1e-6
        ),
        flow_delta_scale=np.maximum(
            np.sqrt(flow_square / max(flow_count, 1)).astype(np.float32), 1e-6
        ),
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
    predicted_difference = (predicted[:, :, None] - predicted[:, None, :]) / scale
    truth_difference = (truth.detach()[:, :, None] - truth.detach()[:, None, :]) / scale
    count = predicted.shape[1]
    upper = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=predicted.device), diagonal=1
    )[None]
    informative = upper & (truth_difference.abs() >= float(min_effect_fraction))
    if not informative.any():
        informative = upper.expand_as(truth_difference)
    else:
        informative = informative.expand_as(truth_difference)
    magnitude = F.smooth_l1_loss(
        predicted_difference[informative],
        truth_difference[informative],
        reduction="mean",
        beta=0.5,
    )
    sign = F.softplus(
        -torch.sign(truth_difference[informative]) * predicted_difference[informative]
    ).mean()
    return magnitude, sign


def value_loss_v70(
    output: DirectValueOutputV70,
    truth: torch.Tensor,
    *,
    scale_m3: float,
    contract: DirectValueLossContractV70 = DirectValueLossContractV70(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Magnitude first; ordering cannot be obtained by collapsing physical scale."""
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
    pair_magnitude, pair_sign = _pairwise_losses(
        predicted,
        truth,
        scale,
        min_effect_fraction=contract.pair_min_effect_fraction,
    )
    total = (
        contract.transformed_magnitude_weight * transformed
        + contract.physical_magnitude_weight * physical
        + contract.pairwise_difference_weight * pair_magnitude
        + contract.pairwise_sign_weight * pair_sign
    )
    return total, {
        "loss": float(total.detach()),
        "transformed_magnitude": float(transformed.detach()),
        "physical_magnitude": float(physical.detach()),
        "pairwise_difference": float(pair_magnitude.detach()),
        "pairwise_sign": float(pair_sign.detach()),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(np.float64)


def _spearman(predicted: np.ndarray, truth: np.ndarray) -> float:
    if len(predicted) < 2 or np.allclose(predicted, predicted[0]) or np.allclose(truth, truth[0]):
        return float("nan")
    return float(np.corrcoef(_rank(predicted), _rank(truth))[0, 1])


def _pairwise_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    total = correct = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if abs(float(truth[i] - truth[j])) <= 1e-9:
                continue
            total += 1
            correct += int(
                np.sign(predicted[i] - predicted[j]) == np.sign(truth[i] - truth[j])
            )
    return float(correct / total) if total else float("nan")


def _event_balanced(records: Sequence[dict[str, Any]], key: str) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            by_event[str(record["event"])].append(value)
    means = [float(np.mean(values)) for values in by_event.values() if values]
    return float(np.mean(means)) if means else float("nan")


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
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
            )
            predicted = output.delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            best_predicted, best_truth = int(np.argmin(predicted)), int(np.argmin(truth))
            nonzero = np.abs(truth) > 1e-9
            entry = cache.entry(name)
            records.append(
                {
                    "event": f"{entry.rainfall_group}::{entry.event_id}",
                    "rank": _spearman(predicted, truth),
                    "pairwise": _pairwise_accuracy(predicted, truth),
                    "sign": float(np.mean(np.sign(predicted[nonzero]) == np.sign(truth[nonzero]))) if nonzero.any() else float("nan"),
                    "top1": float(best_predicted == best_truth),
                    "regret": float(truth[best_predicted] - truth[best_truth]),
                    "mae": float(np.mean(np.abs(predicted - truth))),
                    "truth_spread": float(truth.max() - truth.min()),
                    "predicted_spread": float(predicted.max() - predicted.min()),
                    "truth_abs": float(np.mean(np.abs(truth))),
                    "predicted_abs": float(np.mean(np.abs(predicted))),
                }
            )
    result = {
        "groups": int(len(records)),
        "events": int(len({r["event"] for r in records})),
        "rank": _event_balanced(records, "rank"),
        "pairwise": _event_balanced(records, "pairwise"),
        "sign_accuracy": _event_balanced(records, "sign"),
        "top1_rate": _event_balanced(records, "top1"),
        "mean_regret_m3": _event_balanced(records, "regret"),
        "max_regret_m3": max((float(r["regret"]) for r in records), default=float("nan")),
        "tfv_mae_m3": _event_balanced(records, "mae"),
        "truth_spread_m3": _event_balanced(records, "truth_spread"),
        "predicted_spread_m3": _event_balanced(records, "predicted_spread"),
        "mean_abs_truth_m3": _event_balanced(records, "truth_abs"),
        "mean_abs_prediction_m3": _event_balanced(records, "predicted_abs"),
        "scientific_primary": "event_balanced",
    }
    result["spread_ratio"] = result["predicted_spread_m3"] / max(result["truth_spread_m3"], 1e-6)
    result["response_ratio"] = result["mean_abs_prediction_m3"] / max(result["mean_abs_truth_m3"], 1e-6)
    result["response_collapse"] = bool(result["spread_ratio"] < 1e-3)
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
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
            )
            predicted = output.delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            absolute = np.abs(truth)
            event = f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}"
            masks = {
                "small": absolute < float(q33_m3),
                "medium": (absolute >= float(q33_m3)) & (absolute < float(q67_m3)),
                "large": absolute >= float(q67_m3),
            }
            for label, mask in masks.items():
                if not mask.any():
                    continue
                p, t = predicted[mask], truth[mask]
                buckets[label].append(
                    {
                        "event": event,
                        "count": int(mask.sum()),
                        "rank": _spearman(p, t),
                        "pairwise": _pairwise_accuracy(p, t),
                        "mae": float(np.mean(np.abs(p - t))),
                        "truth_abs": float(np.mean(np.abs(t))),
                        "predicted_abs": float(np.mean(np.abs(p))),
                        "truth_spread": float(t.max() - t.min()) if len(t) > 1 else 0.0,
                        "predicted_spread": float(p.max() - p.min()) if len(p) > 1 else 0.0,
                    }
                )
    result: dict[str, Any] = {}
    for label, records in buckets.items():
        truth_abs = _event_balanced(records, "truth_abs")
        predicted_abs = _event_balanced(records, "predicted_abs")
        truth_spread = _event_balanced(records, "truth_spread")
        predicted_spread = _event_balanced(records, "predicted_spread")
        result[label] = {
            "candidate_count": int(sum(r["count"] for r in records)),
            "events": int(len({r["event"] for r in records})),
            "rank": _event_balanced(records, "rank"),
            "pairwise": _event_balanced(records, "pairwise"),
            "tfv_mae_m3": _event_balanced(records, "mae"),
            "response_ratio": predicted_abs / max(truth_abs, 1e-6),
            "spread_ratio": predicted_spread / max(truth_spread, 1e-6),
        }
    return result


__all__ = [
    "TargetScalesV70",
    "derive_target_scales_v70",
    "evaluate_value_strata_v70",
    "evaluate_value_v70",
    "value_loss_v70",
]
