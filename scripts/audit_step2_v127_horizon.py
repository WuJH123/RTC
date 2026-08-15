"""Audit V127 hydraulic rollout error growth from H30 to H360.

This is a read-only development diagnostic. It separates short-horizon hydraulic fidelity
from autoregressive error accumulation without changing the model or using holdout outcomes
for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rtc.checkpoint_v127 import input_normalization_from_v127_checkpoint, load_step2_v127
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import InputNormalizationV60, V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import _denormalize_group, _static

V127_HORIZON_AUDIT_CONTRACT = "PROJECT7_V127_HYDRAULIC_ROLLOUT_HORIZON_AUDIT_V1"
HORIZONS = (6, 12, 24, 36, 48, 72)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=float)
    return float(values.mean()) if values.size else float("nan")


def _evaluate(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    branch_chunk: int,
) -> dict[str, Any]:
    static = _static(graph, device)
    model.eval().to(device)
    per_horizon: dict[int, list[dict[str, float]]] = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for name in names:
            data = _denormalize_group(cache.batch(name, normalization, device), normalization)
            branches = int(data["settings"].shape[0])
            sums: dict[int, dict[str, float]] = {
                h: {
                    "depth_sq": 0.0,
                    "depth_n": 0.0,
                    "flood_abs": 0.0,
                    "flood_n": 0.0,
                    "flow_abs": 0.0,
                    "flow_n": 0.0,
                    "endpoint_depth_sq": 0.0,
                    "endpoint_depth_n": 0.0,
                }
                for h in HORIZONS
            }
            for start in range(0, branches, branch_chunk):
                stop = min(start + branch_chunk, branches)
                index = torch.arange(start, stop, dtype=torch.long, device=device)
                output = model.rollout(
                    data["initial"].index_select(0, index),
                    data["rainfall"].index_select(0, index),
                    data["settings"].index_select(0, index),
                    data["previous_flow"].index_select(0, index),
                    static["up"],
                    static["down"],
                    static["physics"],
                    static["static"],
                    static["edges"],
                )
                truth_states = data["states"].index_select(0, index)
                truth_flows = data["flows"].index_select(0, index)
                for horizon in HORIZONS:
                    pd = output.states[:, :horizon, :, depth_index]
                    td = truth_states[:, :horizon, :, depth_index]
                    pf = output.states[:, :horizon, :, flood_rate_index]
                    tf = truth_states[:, :horizon, :, flood_rate_index]
                    pq = output.actuator_flows[:, :horizon]
                    tq = truth_flows[:, :horizon]
                    ed = output.states[:, horizon - 1, :, depth_index]
                    etd = truth_states[:, horizon - 1, :, depth_index]
                    item = sums[horizon]
                    item["depth_sq"] += float(torch.sum(torch.square(pd - td)))
                    item["depth_n"] += float(pd.numel())
                    item["flood_abs"] += float(torch.sum(torch.abs(pf - tf)))
                    item["flood_n"] += float(pf.numel())
                    item["flow_abs"] += float(torch.sum(torch.abs(pq - tq)))
                    item["flow_n"] += float(pq.numel())
                    item["endpoint_depth_sq"] += float(torch.sum(torch.square(ed - etd)))
                    item["endpoint_depth_n"] += float(ed.numel())
            for horizon, item in sums.items():
                per_horizon[horizon].append(
                    {
                        "rollout_depth_rmse_m": math_sqrt(item["depth_sq"] / max(item["depth_n"], 1.0)),
                        "endpoint_depth_rmse_m": math_sqrt(item["endpoint_depth_sq"] / max(item["endpoint_depth_n"], 1.0)),
                        "rollout_flood_rate_mae_m3s": item["flood_abs"] / max(item["flood_n"], 1.0),
                        "rollout_actuator_flow_mae_m3s": item["flow_abs"] / max(item["flow_n"], 1.0),
                    }
                )
    report: dict[str, Any] = {"groups": int(len(names)), "horizons": {}}
    for horizon in HORIZONS:
        rows = per_horizon[horizon]
        report["horizons"][str(horizon * 5)] = {
            "minutes": int(horizon * 5),
            "rollout_depth_rmse_m": _mean(rows, "rollout_depth_rmse_m"),
            "endpoint_depth_rmse_m": _mean(rows, "endpoint_depth_rmse_m"),
            "rollout_flood_rate_mae_m3s": _mean(rows, "rollout_flood_rate_mae_m3s"),
            "rollout_actuator_flow_mae_m3s": _mean(rows, "rollout_actuator_flow_mae_m3s"),
        }
    return report


def math_sqrt(value: float) -> float:
    return float(np.sqrt(max(float(value), 0.0)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--branch-chunk", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.branch_chunk <= 0:
        raise ValueError("V127 horizon audit branch chunk must be positive")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, payload = load_step2_v127(args.step2, graph=graph, device=device)
    normalization = input_normalization_from_v127_checkpoint(payload)
    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain = load_causal_forecast_store_v123(args.causal_store)
    state = load_causal_state_store_v127(args.causal_state_store)
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain), state)
    d4_fit = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain), state)
    d4_audit = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain), state)

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    specs = {
        "trainfit_d2": (base_online, [x for x in fit if x.startswith("D2::")]),
        "trainfit_d3": (base_online, [x for x in fit if x.startswith("D3::")]),
        "internal_holdout_d2": (base_online, [x for x in holdout if x.startswith("D2::")]),
        "internal_holdout_d3": (base_online, [x for x in holdout if x.startswith("D3::")]),
        "d4_fit": (d4_fit, d4_fit_raw.names(D4_SOURCE_KIND)),
        "d4_audit": (d4_audit, d4_audit_raw.names(D4_SOURCE_KIND)),
    }
    results = {
        key: _evaluate(
            model,
            cache=cache,
            names=names,
            normalization=normalization,
            graph=graph,
            device=device,
            depth_index=int(args.depth_index),
            flood_rate_index=int(args.flood_rate_index),
            branch_chunk=int(args.branch_chunk),
        )
        for key, (cache, names) in specs.items()
    }
    report = {
        "contract": V127_HORIZON_AUDIT_CONTRACT,
        "step2_sha256": _sha(args.step2),
        "results": results,
        "metric_semantics": {
            "hydraulic_truth": "authoritative SWMM target state/managed-flow trajectories sampled at 300 s",
            "initial_input": "causal Step1 reconstruction",
            "purpose": "diagnose autoregressive error growth; not a replacement for exact SWMM TFV",
        },
        "boundary": {
            "read_only": True,
            "internal_holdout_used_for_training": False,
            "d4_audit_used_for_training": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
