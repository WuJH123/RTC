from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _spearman, _static, _truth_node_volume
from .step2_train_v127_control import (
    V127ControlTrainingDesign,
    _pairwise_accuracy,
    informative_pair_threshold_v127,
)
from .step2_train_v127_streaming import _cpu_group, _cuda_peak, _select_to_device

V127_FAST_EVAL_CONTRACT = (
    "PROJECT7_V127_FAST_FUSED_RANKING_HORIZON_V1_ONE_H360_ROLLOUT_PER_BRANCH"
)
HORIZONS = (6, 12, 24, 36, 48, 72)


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _sqrt(value: float) -> float:
    return float(np.sqrt(max(float(value), 0.0)))


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text)


def _group_once(
    model: Any,
    *,
    cpu: dict[str, torch.Tensor],
    truth: np.ndarray,
    static: dict[str, torch.Tensor],
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    branch_chunk: int,
    design: V127ControlTrainingDesign,
) -> tuple[dict[str, float], dict[int, dict[str, float]], int]:
    branch_count = int(cpu["settings"].shape[0])
    if truth.shape != (branch_count,):
        raise RuntimeError("V127 fast evaluation label/action order mismatch")
    hard_values: list[float] = []
    smooth_values: list[float] = []
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
    rollout_calls = 0
    for start in range(0, branch_count, branch_chunk):
        stop = min(start + branch_chunk, branch_count)
        positions = np.arange(start, stop, dtype=np.int64)
        chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
        output = model.objective_rollout(
            initial_state=chunk["initial"],
            rainfall=chunk["rainfall"],
            settings=chunk["settings"],
            previous_actuator_flow=chunk["previous_flow"],
            actuator_upstream=static["up"],
            actuator_downstream=static["down"],
            actuator_physics=static["physics"],
            static_node_features=static["static"],
            edge_index=static["edges"],
            flood_rate_index=int(flood_rate_index),
            priority_indices=None,
            dt_seconds=300.0,
        )
        rollout_calls += 1
        hard_values.extend(output.tfv_m3.detach().cpu().numpy().astype(float).tolist())
        smooth_values.extend(
            output.optimization_tfv_m3.detach().cpu().numpy().astype(float).tolist()
        )
        for horizon in HORIZONS:
            pd = output.rollout.states[:, :horizon, :, depth_index]
            td = chunk["states"][:, :horizon, :, depth_index]
            pf = output.rollout.states[:, :horizon, :, flood_rate_index]
            tf = chunk["states"][:, :horizon, :, flood_rate_index]
            pq = output.rollout.actuator_flows[:, :horizon]
            tq = chunk["flows"][:, :horizon]
            ed = output.rollout.states[:, horizon - 1, :, depth_index]
            etd = chunk["states"][:, horizon - 1, :, depth_index]
            item = sums[horizon]
            item["depth_sq"] += float(torch.sum(torch.square(pd - td)))
            item["depth_n"] += float(pd.numel())
            item["flood_abs"] += float(torch.sum(torch.abs(pf - tf)))
            item["flood_n"] += float(pf.numel())
            item["flow_abs"] += float(torch.sum(torch.abs(pq - tq)))
            item["flow_n"] += float(pq.numel())
            item["endpoint_depth_sq"] += float(torch.sum(torch.square(ed - etd)))
            item["endpoint_depth_n"] += float(ed.numel())
        del chunk, output

    hard = np.asarray(hard_values, dtype=float)
    smooth = np.asarray(smooth_values, dtype=float)
    if hard.shape != truth.shape or smooth.shape != truth.shape:
        raise RuntimeError("V127 fast evaluation prediction length mismatch")
    threshold = informative_pair_threshold_v127(float(truth[0]), design)
    optimum = float(np.min(truth))
    optimum_mask = np.isclose(truth, optimum, rtol=0.0, atol=1.0e-9)
    hard_selected = int(np.argmin(hard))
    smooth_selected = int(np.argmin(smooth))
    selected_regret = float(truth[hard_selected] - optimum)
    span = max(float(np.max(truth) - optimum), 1.0)
    ranking = {
        "hard_rank": float(_spearman(hard, truth)),
        "smooth_rank": float(_spearman(smooth, truth)),
        "hard_pair": float(_pairwise_accuracy(truth, hard)),
        "informative_pair": float(_pairwise_accuracy(truth, hard, threshold=threshold)),
        "smooth_pair": float(_pairwise_accuracy(truth, smooth)),
        "hard_top1": float(bool(optimum_mask[hard_selected])),
        "smooth_top1": float(bool(optimum_mask[smooth_selected])),
        "hard_mae": float(np.mean(np.abs(hard - truth))),
        "hard_delta_mae": float(
            np.mean(np.abs((hard[1:] - hard[0]) - (truth[1:] - truth[0])))
        ),
        "smooth_delta_mae": float(
            np.mean(np.abs((smooth[1:] - smooth[0]) - (truth[1:] - truth[0])))
        ),
        "regret": selected_regret,
        "normalized_regret": selected_regret / span,
    }
    horizon_rows: dict[int, dict[str, float]] = {}
    for horizon, item in sums.items():
        horizon_rows[horizon] = {
            "rollout_depth_rmse_m": _sqrt(item["depth_sq"] / max(item["depth_n"], 1.0)),
            "endpoint_depth_rmse_m": _sqrt(
                item["endpoint_depth_sq"] / max(item["endpoint_depth_n"], 1.0)
            ),
            "rollout_flood_rate_mae_m3s": item["flood_abs"] / max(item["flood_n"], 1.0),
            "rollout_actuator_flow_mae_m3s": item["flow_abs"] / max(item["flow_n"], 1.0),
        }
    return ranking, horizon_rows, rollout_calls


def evaluate_groups_fast_v127(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int = 0,
    flood_rate_index: int = 2,
    requested_branch_chunk: int = 12,
    design: V127ControlTrainingDesign | None = None,
) -> tuple[dict[str, float | str], dict[str, Any], dict[str, float | int | str]]:
    """Compute ranking and H30-H360 hydraulic evidence from one H360 pass.

    Read-only inference can safely use a larger branch batch than H360 training because no
    autograd graph is retained. If a CUDA OOM occurs, the current group is retried with half
    the inference chunk; no model state or scientific sample set changes.
    """
    if requested_branch_chunk <= 0:
        raise ValueError("V127 fast evaluation branch chunk must be positive")
    if design is None:
        design = V127ControlTrainingDesign(evaluation_branch_chunk=requested_branch_chunk)
    design.validate()
    model.eval().to(device)
    static = _static(graph, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ranking_rows: list[dict[str, float]] = []
    horizon_groups: dict[int, list[dict[str, float]]] = {h: [] for h in HORIZONS}
    total_branches = 0
    rollout_calls = 0
    oom_fallbacks = 0
    minimum_effective_chunk = int(requested_branch_chunk)

    with torch.inference_mode():
        for name in names:
            cpu = _cpu_group(cache, name, normalization)
            truth = _truth_node_volume(cache, name).sum(axis=1).astype(np.float64)
            total_branches += int(cpu["settings"].shape[0])
            effective = int(requested_branch_chunk)
            while True:
                try:
                    ranking, horizons, calls = _group_once(
                        model,
                        cpu=cpu,
                        truth=truth,
                        static=static,
                        device=device,
                        depth_index=int(depth_index),
                        flood_rate_index=int(flood_rate_index),
                        branch_chunk=effective,
                        design=design,
                    )
                    break
                except RuntimeError as exc:
                    if device.type != "cuda" or not _is_cuda_oom(exc) or effective <= 1:
                        raise
                    oom_fallbacks += 1
                    effective = max(1, effective // 2)
                    minimum_effective_chunk = min(minimum_effective_chunk, effective)
                    torch.cuda.empty_cache()
            ranking_rows.append(ranking)
            for horizon, row in horizons.items():
                horizon_groups[horizon].append(row)
            rollout_calls += calls
            del cpu

    ranking_metrics: dict[str, float | str] = {
        "groups": float(len(names)),
        "rank": _mean([r["hard_rank"] for r in ranking_rows]),
        "pairwise": _mean([r["hard_pair"] for r in ranking_rows]),
        "top1": _mean([r["hard_top1"] for r in ranking_rows]),
        "tfv_mae_m3": _mean([r["hard_mae"] for r in ranking_rows]),
        "selected_regret_m3": _mean([r["regret"] for r in ranking_rows]),
        "hard_tfv_rank": _mean([r["hard_rank"] for r in ranking_rows]),
        "hard_tfv_pairwise": _mean([r["hard_pair"] for r in ranking_rows]),
        "hard_tfv_informative_pairwise": _mean([r["informative_pair"] for r in ranking_rows]),
        "hard_tfv_top1": _mean([r["hard_top1"] for r in ranking_rows]),
        "smooth_tfv_rank": _mean([r["smooth_rank"] for r in ranking_rows]),
        "smooth_tfv_pairwise": _mean([r["smooth_pair"] for r in ranking_rows]),
        "smooth_tfv_top1": _mean([r["smooth_top1"] for r in ranking_rows]),
        "hard_tfv_delta_mae_m3": _mean([r["hard_delta_mae"] for r in ranking_rows]),
        "smooth_tfv_delta_mae_m3": _mean([r["smooth_delta_mae"] for r in ranking_rows]),
        "normalized_selected_regret": _mean([r["normalized_regret"] for r in ranking_rows]),
        "authoritative_label_order": "reference_then_candidates",
        "prediction_metric": "hard surrogate TFV vs authoritative SWMM cumulative TFV",
        "smooth_metric_role": "optimization/action-effect diagnostic only",
        "informative_pair_threshold": "max(1 m3, 0.001 * reference TFV)",
    }
    horizon_report: dict[str, Any] = {"groups": int(len(names)), "horizons": {}}
    for horizon in HORIZONS:
        rows = horizon_groups[horizon]
        horizon_report["horizons"][str(horizon * 5)] = {
            "minutes": int(horizon * 5),
            "rollout_depth_rmse_m": _mean([r["rollout_depth_rmse_m"] for r in rows]),
            "endpoint_depth_rmse_m": _mean([r["endpoint_depth_rmse_m"] for r in rows]),
            "rollout_flood_rate_mae_m3s": _mean([r["rollout_flood_rate_mae_m3s"] for r in rows]),
            "rollout_actuator_flow_mae_m3s": _mean([r["rollout_actuator_flow_mae_m3s"] for r in rows]),
        }
    telemetry: dict[str, float | int | str] = {
        "contract": V127_FAST_EVAL_CONTRACT,
        "groups": int(len(names)),
        "branches": int(total_branches),
        "h360_rollout_calls": int(rollout_calls),
        "requested_branch_chunk": int(requested_branch_chunk),
        "minimum_effective_branch_chunk": int(minimum_effective_chunk),
        "cuda_oom_fallbacks": int(oom_fallbacks),
        "gradient_mode": "torch.inference_mode",
        "ranking_and_horizon_share_same_rollout": "true",
        **_cuda_peak(device),
    }
    return ranking_metrics, horizon_report, telemetry


__all__ = [
    "HORIZONS",
    "V127_FAST_EVAL_CONTRACT",
    "evaluate_groups_fast_v127",
]
