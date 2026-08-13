"""Event-balanced diagnostics for the bounded V11.3 mechanism gate."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v113 import SupportConditionedHydraulicSurrogateV113
from .step2_hydraulic_objective_v113 import V113EffectScales
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache


CHANNELS = ("depth_m", "flood_m3s", "storage_m3", "inflow_m3s", "outflow_m3s")


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _channel_metric(pred: np.ndarray | torch.Tensor, truth: np.ndarray | torch.Tensor, active: np.ndarray | torch.Tensor) -> dict[str, float]:
    p, y = _as_numpy(pred).reshape(-1).astype(np.float64), _as_numpy(truth).reshape(-1).astype(np.float64)
    denom = float(np.mean(np.square(y)))
    skill = 1.0 - float(np.mean(np.square(p - y))) / max(denom, 1e-12)
    active = _as_numpy(active).reshape(-1).astype(bool)
    sign = float(np.mean(np.sign(p[active]) == np.sign(y[active]))) if active.any() else float("nan")
    return {
        "count": int(y.size), "active_count": int(active.sum()),
        "skill_vs_zero": _safe(skill), "active_skill_vs_zero": _safe(
            1.0 - float(np.mean(np.square(p[active] - y[active]))) / max(float(np.mean(np.square(y[active]))), 1e-12)
        ) if active.any() else float("nan"),
        "rmse": _safe(float(np.sqrt(np.mean(np.square(p - y))))),
        "mae": _safe(float(np.mean(np.abs(p - y)))),
        "response_ratio": _safe(float(np.sqrt(np.mean(np.square(p))) / max(np.sqrt(np.mean(np.square(y))), 1e-12))),
        "active_sign_accuracy": sign,
        "positive_sign_accuracy": _safe(float(np.mean(np.sign(p[y > 0]) == 1.0))) if np.any(y > 0) else float("nan"),
        "negative_sign_accuracy": _safe(float(np.mean(np.sign(p[y < 0]) == -1.0))) if np.any(y < 0) else float("nan"),
        "inactive_leakage_rms": _safe(float(np.sqrt(np.mean(np.square(p[~active]))))) if np.any(~active) else 0.0,
    }


def build_oracle_support_override_v113(
    batch: Any,
    scales: V113EffectScales,
    storage_mask: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Build a diagnostic-only per-source active-support oracle for D2.

    The changed actuator identity is taken from the causal candidate/reference
    action prefix.  True hydraulic effects are used only for this explicitly
    labelled mechanism upper bound; callers must never pass this tensor to an
    online/production model.
    """
    idx = indices.to(batch.true_candidate_states.device)
    true_state = batch.true_candidate_states.index_select(2, idx) - batch.true_reference_states.index_select(1, idx)[:, None]
    truth5 = true_state[..., [0, 2, 3, 4, 5]]
    threshold = scales.state_threshold.to(truth5)[None, None, None]
    active = truth5.abs() >= threshold
    active[..., 2] &= storage_mask.to(active.device)[None, None, None]
    delta_action = batch.candidate_settings - batch.reference_settings[:, None]
    changed = delta_action.abs().amax(dim=2) > 1e-7  # [B,C,A]
    return active[:, :, None] & changed[:, :, :, None, None, None]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = sorted({str(x["event_key"]) for x in records})
    result: dict[str, Any] = {"groups": len(records), "events": len(events), "channels": {}}
    for channel in CHANNELS + ("managed_flow_m3s",):
        keys = sorted({k for r in records for k in r[channel] if k not in {"count", "active_count"}})
        by_event: dict[str, list[dict[str, float]]] = defaultdict(list)
        for r in records:
            by_event[str(r["event_key"])].append(r[channel])
        out: dict[str, float] = {}
        for key in keys:
            vals = [float(np.mean([x[key] for x in group if np.isfinite(float(x[key]))])) for group in by_event.values() if any(np.isfinite(float(x[key])) for x in group)]
            out[key] = float(np.mean(vals)) if vals else float("nan")
        out["count"] = int(sum(int(r[channel]["count"]) for r in records))
        out["active_count"] = int(sum(int(r[channel]["active_count"]) for r in records))
        result["channels"][channel] = out
    result["scientific_primary"] = "event_balanced"
    return result


def evaluate_hydraulic_v113(
    model: SupportConditionedHydraulicSurrogateV113,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: object,
    scales: V113EffectScales,
    *,
    storage_mask: torch.Tensor,
    device: torch.device | str,
    oracle_support: bool = False,
) -> dict[str, Any]:
    target_device = torch.device(device)
    model.eval().to(target_device)
    records: list[dict[str, Any]] = []
    bucket_records: dict[str, list[dict[str, Any]]] = {"0_30_min": [], "30_120_min": [], "120_360_min": []}
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target_device)
            entry = cache.entry(name)
            raw_initial = torch.from_numpy(np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float32).copy())[None].to(target_device)
            override = build_oracle_support_override_v113(batch, scales, storage_mask, model.retained_indices) if oracle_support else None
            out = model(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, prepared, initial_state_physical=raw_initial, support_override=override)
            idx = out.horizon_indices
            true_state = batch.true_candidate_states.index_select(2, idx) - batch.true_reference_states.index_select(1, idx)[:, None]
            true_flow = batch.true_candidate_flows.index_select(2, idx) - batch.true_reference_flows.index_select(1, idx)[:, None]
            pred_state = out.raw_delta_states_physical
            pred_flow = out.raw_delta_flows_physical
            active_state = true_state[..., [0, 2, 3, 4, 5]].abs() >= scales.state_threshold.to(target_device)[None, None, None]
            active_state[..., 2] &= storage_mask.to(target_device)[None, None, None]
            active_flow = true_flow.abs() >= scales.flow_threshold.to(target_device)[None, None, None]
            state_vals = {
                "depth_m": _channel_metric(pred_state[..., 0], true_state[..., 0], active_state[..., 0]),
                "flood_m3s": _channel_metric(pred_state[..., 2], true_state[..., 2], active_state[..., 1]),
                "storage_m3": _channel_metric(pred_state[..., 3], true_state[..., 3], active_state[..., 2]),
                "inflow_m3s": _channel_metric(pred_state[..., 4], true_state[..., 4], active_state[..., 3]),
                "outflow_m3s": _channel_metric(pred_state[..., 5], true_state[..., 5], active_state[..., 4]),
                "managed_flow_m3s": _channel_metric(pred_flow, true_flow, active_flow),
            }
            record = {"event_key": f"{entry.rainfall_group}::{entry.event_id}", **state_vals}
            records.append(record)
            minutes = out.response_minutes.detach().cpu().numpy()
            for key, mask in (("0_30_min", minutes <= 30.0), ("30_120_min", (minutes > 30.0) & (minutes <= 120.0)), ("120_360_min", minutes > 120.0)):
                if np.any(mask):
                    m = torch.as_tensor(mask, dtype=torch.bool, device=target_device)
                    bstate = true_state[:, :, m]
                    bpred = pred_state[:, :, m]
                    bflow = true_flow[:, :, m]
                    bam = active_state[:, :, m]
                    bfm = active_flow[:, :, m]
                    bucket_records[key].append({"event_key": record["event_key"],
                        "depth_m": _channel_metric(bpred[..., 0], bstate[..., 0], bam[..., 0]),
                        "flood_m3s": _channel_metric(bpred[..., 2], bstate[..., 2], bam[..., 1]),
                        "storage_m3": _channel_metric(bpred[..., 3], bstate[..., 3], bam[..., 2]),
                        "inflow_m3s": _channel_metric(bpred[..., 4], bstate[..., 4], bam[..., 3]),
                        "outflow_m3s": _channel_metric(bpred[..., 5], bstate[..., 5], bam[..., 4]),
                        "managed_flow_m3s": _channel_metric(bflow, bpred.new_tensor(bflow), bfm) if False else _channel_metric(out.raw_delta_flows_physical[:, :, m], true_flow[:, :, m], bfm)})
    return {"overall": _aggregate(records), "horizon_buckets": {k: _aggregate(v) if v else None for k, v in bucket_records.items()}}


__all__ = ["evaluate_hydraulic_v113", "CHANNELS", "build_oracle_support_override_v113"]
