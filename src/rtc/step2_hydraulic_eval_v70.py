"""Event-balanced counterfactual hydraulic-effect evaluation for Step2 V7.0."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import HydraulicResponseSurrogateV70
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan")


def _event_balanced(records: Sequence[dict[str, Any]], key: str) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            by_event[str(record["event"])].append(value)
    values = [float(np.mean(items)) for items in by_event.values() if items]
    return float(np.mean(values)) if values else float("nan")


def evaluate_hydraulic_effect_v70(
    model: HydraulicResponseSurrogateV70,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    scales: TargetScalesV70,
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
                prepared,
            )
            idx = output.horizon_indices
            true_ref = batch.true_reference_states.index_select(1, idx)[:, None].expand_as(
                output.reference_states_physical
            )
            true_cand = batch.true_candidate_states.index_select(2, idx)
            true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None].expand_as(
                output.reference_flows_physical
            )
            true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
            pred_delta = (output.candidate_states_physical - output.reference_states_physical).detach().cpu().numpy().astype(np.float64)
            true_delta = (true_cand - true_ref).detach().cpu().numpy().astype(np.float64)
            pred_flow_delta = output.delta_flows_physical.detach().cpu().numpy().astype(np.float64)
            true_flow_delta = (true_cand_flow - true_ref_flow).detach().cpu().numpy().astype(np.float64)
            event = f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}"
            record = {
                "event": event,
                "delta_depth_rmse_m": _rmse(pred_delta[..., 0] - true_delta[..., 0]),
                "delta_flood_rmse_m3s": _rmse(pred_delta[..., 2] - true_delta[..., 2]),
                "delta_storage_rmse_m3": _rmse(pred_delta[..., 3] - true_delta[..., 3]),
                "delta_inflow_rmse_m3s": _rmse(pred_delta[..., 4] - true_delta[..., 4]),
                "delta_outflow_rmse_m3s": _rmse(pred_delta[..., 5] - true_delta[..., 5]),
                "delta_managed_flow_rmse_m3s": _rmse(pred_flow_delta - true_flow_delta),
            }
            for channel, label in ((0, "depth"), (2, "flood"), (3, "storage")):
                truth_abs = float(np.mean(np.abs(true_delta[..., channel])))
                pred_abs = float(np.mean(np.abs(pred_delta[..., channel])))
                record[f"delta_{label}_response_ratio"] = pred_abs / max(truth_abs, 1e-9)
            records.append(record)

    metrics = [
        "delta_depth_rmse_m",
        "delta_flood_rmse_m3s",
        "delta_storage_rmse_m3",
        "delta_inflow_rmse_m3s",
        "delta_outflow_rmse_m3s",
        "delta_managed_flow_rmse_m3s",
        "delta_depth_response_ratio",
        "delta_flood_response_ratio",
        "delta_storage_response_ratio",
    ]
    return {
        "groups": int(len(records)),
        "events": int(len({r["event"] for r in records})),
        "scientific_primary": "event_balanced_candidate_minus_reference_effect",
        **{metric: _event_balanced(records, metric) for metric in metrics},
        "state_delta_scales": np.asarray(scales.state_delta_scale).tolist(),
        "flow_delta_scale_median": float(np.median(scales.flow_delta_scale)),
    }


__all__ = ["evaluate_hydraulic_effect_v70"]
