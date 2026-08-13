"""V11.1 evaluation: pooled event skill plus leakage diagnostics.

The legacy group-ratio metric is retained for historical comparison.  The
primary skill pools SSE over all groups/candidates/times/nodes inside each
event before taking the ratio, preventing near-zero groups from dominating.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import STATE_EFFECT_CHANNELS_V110
from .step2_hydraulic_objective_v111 import EffectScalesV111
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_v111_contract import V111LossContract

STATE_NAMES_V111 = ("depth_m", "flood_m3s", "volume_m3", "total_inflow_m3s", "total_outflow_m3s")


def event_pooled_skill_vs_zero(records: Sequence[dict[str, Any]]) -> float:
    """Pool squared errors within each event, then mean event ratios."""
    by_event: dict[str, tuple[float, float]] = {}
    for record in records:
        event = str(record["event_id"])
        pred = np.asarray(record["pred"], dtype=np.float64)
        truth = np.asarray(record["truth"], dtype=np.float64)
        if pred.shape != truth.shape:
            raise ValueError("V111 pooled-skill record prediction/target shape mismatch")
        model_sse, zero_sse = by_event.get(event, (0.0, 0.0))
        by_event[event] = (model_sse + float(np.square(pred - truth).sum()),
                           zero_sse + float(np.square(truth).sum()))
    values = [1.0 - model / zero for model, zero in by_event.values() if zero > 1.0e-12]
    return float(np.mean(values)) if values else float("nan")


def _skill(pred: np.ndarray, truth: np.ndarray) -> float:
    zero = float(np.square(truth).mean())
    return float(1.0 - np.square(pred - truth).mean() / zero) if zero > 1.0e-12 else float("nan")


def _legacy_event_group_skill(records: Sequence[dict[str, Any]], key: str) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = _skill(np.asarray(record["pred_by_channel"][key]), np.asarray(record["truth_by_channel"][key]))
        if np.isfinite(value):
            by_event[str(record["event_id"])].append(value)
    event_values = [float(np.mean(vals)) for vals in by_event.values() if vals]
    return float(np.mean(event_values)) if event_values else float("nan")


def _leakage(pred: np.ndarray, truth: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    inactive = np.abs(truth) < threshold
    values = np.abs(pred[inactive]) if inactive.any() else np.empty(0)
    if not values.size:
        return {"inactive_pred_rms": float("nan"), "inactive_pred_p95_abs": float("nan"),
                "inactive_pred_max_abs": float("nan"), "inactive_leakage_ratio": float("nan")}
    return {
        "inactive_pred_rms": float(np.sqrt(np.mean(values ** 2))),
        "inactive_pred_p95_abs": float(np.quantile(values, .95)),
        "inactive_pred_max_abs": float(values.max()),
        "inactive_leakage_ratio": float(np.sqrt(np.mean(values ** 2)) / max(float(np.mean(threshold)), 1e-12)),
    }


def evaluate_arrays_v111(
    *, event_ids: Sequence[str], pred_state: np.ndarray, true_state: np.ndarray,
    pred_flow: np.ndarray, true_flow: np.ndarray, scales: EffectScalesV111,
) -> dict[str, Any]:
    """Evaluate already-materialized arrays.

    State arrays use [sample,time,node,5] in the public V111 channel order;
    flow arrays use [sample,time,actuator].  This function is intentionally
    independent of a model/cache so its semantics are easy to unit test.
    """
    if pred_state.shape != true_state.shape or pred_flow.shape != true_flow.shape:
        raise ValueError("V111 evaluation arrays are misaligned")
    if len(event_ids) != pred_state.shape[0]:
        raise ValueError("V111 event IDs must align with samples")
    state_scale = np.asarray(scales.state_magnitude_scale)[None, None]
    state_threshold = np.asarray(scales.state_active_threshold)[None, None]
    flow_scale = np.asarray(scales.flow_magnitude_scale)[None, None]
    flow_threshold = np.asarray(scales.flow_active_threshold)[None, None]
    output: dict[str, Any] = {"primary_event_pooled_skill_vs_zero": {}, "legacy_group_skill_vs_zero": {},
                               "response_ratio": {}, "leakage": {}}
    event_values = sorted(set(str(x) for x in event_ids))
    for channel, name in enumerate(STATE_NAMES_V111):
        pooled = []
        group_records = []
        for i, event in enumerate(event_ids):
            pooled.append({"event_id": event, "pred": pred_state[i, ..., channel], "truth": true_state[i, ..., channel]})
            group_records.append({"event_id": event, "pred_by_channel": {name: pred_state[i, ..., channel]},
                                  "truth_by_channel": {name: true_state[i, ..., channel]}})
        output["primary_event_pooled_skill_vs_zero"][name] = event_pooled_skill_vs_zero(pooled)
        output["legacy_group_skill_vs_zero"][name] = _legacy_event_group_skill(group_records, name)
        output["response_ratio"][name] = float(np.sqrt(np.mean(pred_state[..., channel] ** 2)) /
                                                max(float(np.sqrt(np.mean(true_state[..., channel] ** 2))), 1e-12))
        output["leakage"][name] = _leakage(pred_state[..., channel], true_state[..., channel], state_threshold[..., channel])
    for actuator_name, array_pred, array_true, threshold in (("managed_flow_m3s", pred_flow, true_flow, flow_threshold),):
        pooled = [{"event_id": event_ids[i], "pred": array_pred[i], "truth": array_true[i]} for i in range(len(event_ids))]
        group_records = [{"event_id": event_ids[i], "pred_by_channel": {actuator_name: array_pred[i]},
                          "truth_by_channel": {actuator_name: array_true[i]}} for i in range(len(event_ids))]
        output["primary_event_pooled_skill_vs_zero"][actuator_name] = event_pooled_skill_vs_zero(pooled)
        output["legacy_group_skill_vs_zero"][actuator_name] = _legacy_event_group_skill(group_records, actuator_name)
        output["response_ratio"][actuator_name] = float(np.sqrt(np.mean(array_pred ** 2)) /
                                                         max(float(np.sqrt(np.mean(array_true ** 2))), 1e-12))
        output["leakage"][actuator_name] = _leakage(array_pred, array_true, threshold)
    output["event_count"] = len(event_values)
    return output


__all__ = ["STATE_NAMES_V111", "event_pooled_skill_vs_zero", "evaluate_arrays_v111"]
