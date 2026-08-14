"""Audit V127 autograd against exact InternalHoldout D2 SWMM finite differences.

The authoritative D2 branches provide constant-setting perturbations.  Predictions use
causal Step1 state and causal rainfall at the same checkpoint, matching online Step2
inputs.  No InternalHoldout row is used for model fitting or calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.checkpoint_v127 import input_normalization_from_v127_checkpoint, load_step2_v127
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60

V127_D2_GRADIENT_AUDIT_CONTRACT = "PROJECT7_V127_INTERNAL_HOLDOUT_D2_CAUSAL_GRADIENT_AUDIT_V1"


def _physical(batch, normalization, device):
    dtype = batch.initial_state.dtype
    sm = torch.as_tensor(normalization.state_mean, dtype=dtype, device=device)
    ss = torch.as_tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1e-6)
    rm = torch.as_tensor(normalization.rainfall_mean, dtype=dtype, device=device)
    rs = torch.as_tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1e-6)
    fm = torch.as_tensor(normalization.flow_mean, dtype=dtype, device=device)
    fs = torch.as_tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1e-6)
    return batch.initial_state * ss + sm, batch.rainfall * rs + rm, batch.previous_actuator_flow * fs + fm


def _predicted_constant_gradient(
    model, *, graph, initial, rainfall, flow, base_sequence: np.ndarray,
    actuator_index: int, base_setting: float, flood_rate_index: int,
) -> float:
    device = initial.device
    scalar = torch.tensor(float(base_setting), dtype=initial.dtype, device=device, requires_grad=True)
    settings = torch.as_tensor(base_sequence, dtype=initial.dtype, device=device).clone()
    # D2 semantics: one actuator setting is held at the probed scalar over the horizon.
    mask = torch.zeros_like(settings)
    mask[:, int(actuator_index)] = 1.0
    settings = settings * (1.0 - mask) + scalar * mask
    output = model.objective_rollout(
        initial_state=initial,
        rainfall=rainfall,
        settings=settings[None],
        previous_actuator_flow=flow,
        actuator_upstream=torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        actuator_physics=torch.as_tensor(graph.actuator_physics, dtype=initial.dtype, device=device),
        static_node_features=torch.as_tensor(graph.static_node_features, dtype=initial.dtype, device=device),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        flood_rate_index=int(flood_rate_index),
        priority_indices=None,
        dt_seconds=300.0,
    )
    gradient = torch.autograd.grad(output.tfv_m3.sum(), scalar)[0]
    return float(gradient.detach())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, payload = load_step2_v127(args.step2, graph=graph, device=device)
    normalization = input_normalization_from_v127_checkpoint(payload)
    base = V60TrainCache(args.cache_manifest)
    rain = load_causal_forecast_store_v123(args.causal_store)
    state = load_causal_state_store_v127(args.causal_state_store)
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain), state)
    fit, holdout = deterministic_rainfall_split_v60(
        base, names=sorted(base.names("D2") + base.targeted_d3_names()), holdout_fraction=0.20
    )
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    if len(hold_d2) != 32:
        raise ValueError("V127 D2 gradient audit requires frozen 32 InternalHoldout groups")
    actuator_index = {aid: i for i, aid in enumerate(graph.actuator_ids)}
    rows: list[dict[str, object]] = []
    for name in hold_d2:
        entry = base.entry(name)
        arrays = entry.arrays
        required = {"actuator_id", "requested_setting", "base_setting", "exact_node_flood_volume_m3", "settings"}
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{name}: D2 cache lacks gradient provenance {missing}")
        batch = online.batch(name, normalization, device)
        initial, rainfall, flow = _physical(batch, normalization, device)
        tfv = np.asarray(arrays["exact_node_flood_volume_m3"], dtype=np.float64).sum(axis=1)
        aids = np.asarray(arrays["actuator_id"]).astype(str)
        requested = np.asarray(arrays["requested_setting"], dtype=np.float64)
        base_setting = np.asarray(arrays["base_setting"], dtype=np.float64)
        reference_sequence = np.asarray(arrays["settings"][entry.reference_index], dtype=np.float32)
        for aid in sorted(set(aids.tolist()) - {""}):
            if aid not in actuator_index:
                raise ValueError(f"{name}: D2 actuator {aid} absent from graph")
            idx = np.flatnonzero(aids == aid)
            if not idx.size:
                continue
            b = float(base_setting[idx[0]])
            if not np.allclose(base_setting[idx], b, rtol=0.0, atol=1e-10):
                raise ValueError(f"{name}/{aid}: D2 base setting drift")
            center = idx[np.isclose(requested[idx], b, rtol=0.0, atol=1e-10)]
            below = idx[requested[idx] < b - 1e-10]
            above = idx[requested[idx] > b + 1e-10]
            if center.size:
                mid = int(center[0])
            else:
                mid = int(entry.reference_index)
            if below.size and above.size:
                lo = int(below[np.argmax(requested[below])])
                hi = int(above[np.argmin(requested[above])])
                du = float(requested[hi] - requested[lo])
                truth = float((tfv[hi] - tfv[lo]) / du)
                method = "central"
            elif above.size:
                hi = int(above[np.argmin(requested[above])])
                du = float(requested[hi] - b)
                truth = float((tfv[hi] - tfv[mid]) / du)
                method = "forward_bound"
            elif below.size:
                lo = int(below[np.argmax(requested[below])])
                du = float(b - requested[lo])
                truth = float((tfv[mid] - tfv[lo]) / du)
                method = "backward_bound"
            else:
                continue
            if du <= 1e-12:
                continue
            predicted = _predicted_constant_gradient(
                model,
                graph=graph,
                initial=initial,
                rainfall=rainfall,
                flow=flow,
                base_sequence=reference_sequence,
                actuator_index=actuator_index[aid],
                base_setting=b,
                flood_rate_index=int(args.flood_rate_index),
            )
            rows.append({
                "group": name,
                "rainfall_group": str(entry.rainfall_group),
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "actuator_id": aid,
                "base_setting": b,
                "finite_difference_method": method,
                "true_tfv_gradient_m3_per_setting": truth,
                "predicted_tfv_gradient_m3_per_setting": predicted,
            })
    detail = pd.DataFrame.from_records(rows)
    if detail.empty:
        raise RuntimeError("V127 InternalHoldout D2 gradient audit produced no cases")
    group_metrics: list[dict[str, float]] = []
    for _, group in detail.groupby("rainfall_group", sort=True):
        truth = group["true_tfv_gradient_m3_per_setting"].to_numpy(dtype=float)
        pred = group["predicted_tfv_gradient_m3_per_setting"].to_numpy(dtype=float)
        mask = np.abs(truth) > 1e-8
        sign = float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask]))) if mask.any() else float("nan")
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        cosine = float(np.dot(truth, pred) / denom) if denom > 1e-12 else float("nan")
        group_metrics.append({
            "sign": sign,
            "cosine": cosine,
            "mae": float(np.mean(np.abs(pred - truth))),
        })
    def mean(key: str) -> float:
        values = np.asarray([row[key] for row in group_metrics], dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")
    metrics = {
        "contract": V127_D2_GRADIENT_AUDIT_CONTRACT,
        "gradient_cases": len(detail),
        "gradient_rainfall_groups": int(detail["rainfall_group"].nunique()),
        "tfv_gradient_sign_accuracy": mean("sign"),
        "tfv_gradient_cosine_similarity": mean("cosine"),
        "tfv_gradient_mae": mean("mae"),
        "split": "InternalHoldout D2 only",
        "causal_step1_state": True,
        "causal_rainfall": True,
        "used_for_training": False,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "D2_INTERNAL_HOLDOUT_GRADIENT_DETAIL.csv", index=False)
    (out / "D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
