"""Event-balanced evaluation for Project7 Step2 V11 Hydraulic Effect."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import ActuatorSetHydraulicResponseV110, STATE_EFFECT_CHANNELS_V110
from .step2_hydraulic_objective_v110 import EffectScalesV110
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_v110_contract import ActuatorSetHydraulicContractV110

STATE_NAMES_V110 = ("depth_m", "flood_m3s", "volume_m3", "total_inflow_m3s", "total_outflow_m3s")


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _event_balanced(records: Sequence[dict[str, Any]], key: str) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            by_event[str(record["event_id"])].append(value)
    return _finite_mean([_finite_mean(values) for values in by_event.values()])


def _skill(pred: np.ndarray, truth: np.ndarray) -> float:
    mse = float(np.mean(np.square(pred - truth)))
    zero = float(np.mean(np.square(truth)))
    return float(1.0 - mse / max(zero, 1.0e-12)) if zero > 1.0e-12 else float("nan")


def _response_ratio(pred: np.ndarray, truth: np.ndarray) -> float:
    p = float(np.sqrt(np.mean(np.square(pred))))
    t = float(np.sqrt(np.mean(np.square(truth))))
    return float(p / max(t, 1.0e-12)) if t > 1.0e-12 else float("nan")


def _active_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    threshold: np.ndarray,
    active_probability: np.ndarray,
) -> dict[str, float]:
    true_active = np.abs(truth) >= threshold
    pred_active = active_probability >= 0.5
    tp = int(np.logical_and(pred_active, true_active).sum())
    fp = int(np.logical_and(pred_active, ~true_active).sum())
    fn = int(np.logical_and(~pred_active, true_active).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    if true_active.any():
        sign = float(np.mean(np.sign(pred[true_active]) == np.sign(truth[true_active])))
        magnitude_ratio = float(
            np.mean(np.abs(pred[true_active]))
            / max(float(np.mean(np.abs(truth[true_active]))), 1.0e-12)
        )
    else:
        sign = magnitude_ratio = float("nan")
    return {
        "active_precision": float(precision),
        "active_recall": float(recall),
        "active_f1": float(f1),
        "active_sign_accuracy": sign,
        "active_magnitude_ratio": magnitude_ratio,
        "true_active_fraction": float(true_active.mean()),
    }


def _timing_errors(
    pred: np.ndarray,
    truth: np.ndarray,
    true_active: np.ndarray,
    pred_active: np.ndarray,
    response_minutes: np.ndarray,
) -> tuple[float, float]:
    pred_mass = np.sqrt(np.mean(np.square(pred), axis=tuple(range(3, pred.ndim))))
    true_mass = np.sqrt(np.mean(np.square(truth), axis=tuple(range(3, truth.ndim))))
    pred_support = pred_active.any(axis=tuple(range(3, pred_active.ndim)))
    true_support = true_active.any(axis=tuple(range(3, true_active.ndim)))
    peak_errors: list[float] = []
    onset_errors: list[float] = []
    for b in range(pred_mass.shape[0]):
        for c in range(pred_mass.shape[1]):
            if np.max(true_mass[b, c]) > 0:
                pidx = int(np.argmax(pred_mass[b, c]))
                tidx = int(np.argmax(true_mass[b, c]))
                peak_errors.append(abs(float(response_minutes[pidx] - response_minutes[tidx])))
            true_times = np.flatnonzero(true_support[b, c])
            pred_times = np.flatnonzero(pred_support[b, c])
            if true_times.size:
                if pred_times.size:
                    onset_errors.append(abs(float(response_minutes[int(pred_times[0])] - response_minutes[int(true_times[0])])))
                else:
                    onset_errors.append(float(response_minutes[-1] - response_minutes[int(true_times[0])]))
    return _finite_mean(peak_errors), _finite_mean(onset_errors)


def evaluate_hydraulic_effect_v110(
    model: ActuatorSetHydraulicResponseV110,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    scales: EffectScalesV110,
    *,
    device: torch.device | str,
    contract: ActuatorSetHydraulicContractV110 = ActuatorSetHydraulicContractV110(),
) -> dict[str, Any]:
    contract.validate()
    target = torch.device(device)
    model.to(target).eval()
    records: list[dict[str, Any]] = []
    state_threshold = scales.state_active_threshold
    flow_threshold = scales.flow_active_threshold
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target)
            candidates = batch.candidate_settings.shape[1]
            for start in range(0, candidates, contract.candidate_chunk_size):
                end = min(start + contract.candidate_chunk_size, candidates)
                output = model(
                    batch.initial_state,
                    batch.rainfall,
                    batch.reference_settings,
                    batch.candidate_settings[:, start:end],
                    batch.previous_actuator_flow,
                    prepared,
                )
                idx = output.horizon_indices
                ref_state = batch.true_reference_states.index_select(1, idx)[:, None]
                cand_state = batch.true_candidate_states[:, start:end].index_select(2, idx)
                truth_state_full = cand_state - ref_state
                truth_state = truth_state_full[..., list(STATE_EFFECT_CHANNELS_V110)]
                pred_state = output.raw_delta_states_physical[..., list(STATE_EFFECT_CHANNELS_V110)]
                ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None]
                cand_flow = batch.true_candidate_flows[:, start:end].index_select(2, idx)
                truth_flow = cand_flow - ref_flow
                pred_flow = output.raw_delta_flows_physical

                ps = pred_state.detach().cpu().numpy()
                ts = truth_state.detach().cpu().numpy()
                pf = pred_flow.detach().cpu().numpy()
                tf = truth_flow.detach().cpu().numpy()
                aps = torch.sigmoid(output.active_state_logits).cpu().numpy()
                apf = torch.sigmoid(output.active_flow_logits).cpu().numpy()
                minutes = output.response_minutes.cpu().numpy()
                entry = cache.entry(name)
                record: dict[str, Any] = {
                    "group_name": name,
                    "event_id": entry.event_id,
                    "rainfall_group": entry.rainfall_group,
                }
                for channel, label in enumerate(STATE_NAMES_V110):
                    pred = ps[..., channel]
                    truth = ts[..., channel]
                    threshold = state_threshold[:, channel][None, None, None, :]
                    record[f"{label}_skill_vs_zero"] = _skill(pred, truth)
                    record[f"{label}_response_ratio"] = _response_ratio(pred, truth)
                    active = _active_metrics(pred, truth, threshold, aps[..., channel])
                    for key, value in active.items():
                        record[f"{label}_{key}"] = value
                    true_active = np.abs(truth) >= threshold
                    pred_active = aps[..., channel] >= 0.5
                    peak, onset = _timing_errors(
                        pred[..., None], truth[..., None], true_active[..., None], pred_active[..., None], minutes
                    )
                    record[f"{label}_peak_timing_error_min"] = peak
                    record[f"{label}_onset_timing_error_min"] = onset
                    for bucket, lo, hi in (
                        ("0_30", 0.0, 30.0),
                        ("30_60", 30.0, 60.0),
                        ("60_120", 60.0, 120.0),
                    ):
                        keep = (minutes > lo) & (minutes <= hi)
                        record[f"{label}_skill_{bucket}"] = (
                            _skill(pred[:, :, keep], truth[:, :, keep]) if keep.any() else float("nan")
                        )

                flow_thr = flow_threshold[None, None, None, :]
                record["managed_flow_m3s_skill_vs_zero"] = _skill(pf, tf)
                record["managed_flow_m3s_response_ratio"] = _response_ratio(pf, tf)
                active_flow = _active_metrics(pf, tf, flow_thr, apf)
                for key, value in active_flow.items():
                    record[f"managed_flow_m3s_{key}"] = value
                true_flow_active = np.abs(tf) >= flow_thr
                pred_flow_active = apf >= 0.5
                peak, onset = _timing_errors(
                    pf[..., None], tf[..., None], true_flow_active[..., None], pred_flow_active[..., None], minutes
                )
                record["managed_flow_m3s_peak_timing_error_min"] = peak
                record["managed_flow_m3s_onset_timing_error_min"] = onset
                records.append(record)

    keys = sorted({key for record in records for key in record if key not in {
        "group_name", "event_id", "rainfall_group"
    }})
    overall = {key: _event_balanced(records, key) for key in keys}
    overall["group_count"] = len(names)
    overall["event_count"] = len({cache.entry(name).event_id for name in names})
    return {"overall": overall, "records": records}


__all__ = ["STATE_NAMES_V110", "evaluate_hydraulic_effect_v110"]
