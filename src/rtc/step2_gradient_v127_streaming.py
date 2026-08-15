from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .step2_gradient_v127 import (
    V127GradientTrainingDesign,
    _hard_center_tfv_v127,
    _lookup,
    _predicted_symmetric_difference_v127,
    _scale,
    predicted_directional_gradient_v127,
)
from .step2_train_response_v60 import InputNormalizationV60

V127_GRADIENT_STREAMING_CONTRACT = (
    "PROJECT7_V127_D5_GRADIENT_STREAMING_V1_CAUSAL_INPUTS_ONLY_TO_CUDA"
)


def _physical_inputs_streaming(
    cache: Any,
    name: str,
    normalization: InputNormalizationV60,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage only online-admissible inputs; target trajectories remain on CPU/mmap."""

    batch = cache.batch(name, normalization, torch.device("cpu"))
    dtype = batch.initial_state.dtype
    state_std = torch.as_tensor(normalization.state_std, dtype=dtype)
    state_mean = torch.as_tensor(normalization.state_mean, dtype=dtype)
    rain_std = torch.as_tensor(normalization.rainfall_std, dtype=dtype)
    rain_mean = torch.as_tensor(normalization.rainfall_mean, dtype=dtype)
    flow_std = torch.as_tensor(normalization.flow_std, dtype=dtype)
    flow_mean = torch.as_tensor(normalization.flow_mean, dtype=dtype)
    initial = batch.initial_state * state_std.clamp_min(1.0e-6) + state_mean
    rainfall = batch.rainfall * rain_std.clamp_min(1.0e-6) + rain_mean
    previous = batch.previous_actuator_flow * flow_std.clamp_min(1.0e-6) + flow_mean
    return initial.to(device), rainfall.to(device), previous.to(device)


def _peak(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"cuda_peak_allocated_gb": 0.0, "cuda_peak_reserved_gb": 0.0}
    gib = float(1024**3)
    return {
        "cuda_peak_allocated_gb": float(torch.cuda.max_memory_allocated(device) / gib),
        "cuda_peak_reserved_gb": float(torch.cuda.max_memory_reserved(device) / gib),
    }


def train_d5_gradient_streaming_v127(
    model,
    *,
    cache: Any,
    cases: Sequence[dict[str, Any]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127GradientTrainingDesign = V127GradientTrainingDesign(),
) -> list[dict[str, float | int | str]]:
    design.validate()
    scale = _scale(cases)
    lookup = _lookup(cache)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    model.train().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(42)
    for epoch in range(1, design.epochs + 1):
        order = rng.permutation(len(cases))
        losses: list[float] = []
        hits: list[float] = []
        for index in order:
            case = cases[int(index)]
            name = lookup.get((str(case["event_id"]), str(case["checkpoint_id"])))
            if name is None:
                raise KeyError("V127 D5 gradient case lacks base causal input group")
            initial, rainfall, flow = _physical_inputs_streaming(
                cache, name, normalization, device
            )
            optimizer.zero_grad(set_to_none=True)
            predicted = _predicted_symmetric_difference_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                plus_sequence=np.asarray(case["plus_sequence"]),
                minus_sequence=np.asarray(case["minus_sequence"]),
                epsilon=float(case["epsilon"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
            hard_center_tfv = _hard_center_tfv_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_sequence=np.asarray(case["center_sequence"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
            truth = torch.as_tensor(
                float(case["true_tfv_gradient"]), dtype=predicted.dtype, device=device
            )
            magnitude = F.smooth_l1_loss(predicted / scale, truth / scale, beta=0.5)
            sign_loss = (
                F.softplus(-torch.sign(truth) * predicted / scale)
                if abs(float(case["true_tfv_gradient"])) > 1.0e-8
                else predicted.new_zeros(())
            )
            hard_truth = torch.as_tensor(
                float(case["center_tfv_m3"]), dtype=hard_center_tfv.dtype, device=device
            )
            hard_scale = max(abs(float(case["center_tfv_m3"])), 1000.0)
            center_loss = F.smooth_l1_loss(
                (hard_center_tfv - hard_truth) / hard_scale,
                torch.zeros_like(hard_center_tfv),
                beta=0.5,
            )
            loss = (
                design.magnitude_weight * magnitude
                + design.sign_weight * sign_loss
                + design.hard_center_tfv_weight * center_loss
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V127 streaming D5 gradient loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            if abs(float(case["true_tfv_gradient"])) > 1.0e-8:
                hits.append(float(torch.sign(predicted.detach()) == torch.sign(truth)))
        row: dict[str, float | int | str] = {
            "stage": "D5_1308var_symmetric_gradient_finetune_streaming",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_sign_accuracy": float(np.mean(hits)) if hits else float("nan"),
            "gradient_scale_m3_per_fraction": scale,
            "second_order_parameter_backprop": "false",
            "execution_contract": V127_GRADIENT_STREAMING_CONTRACT,
            **_peak(device),
        }
        history.append(row)
        print("[V127_D5_STREAM] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def evaluate_d5_gradients_streaming_v127(
    model,
    *,
    cache: Any,
    cases: Sequence[dict[str, Any]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    lookup = _lookup(cache)
    rows: list[dict[str, Any]] = []
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for case in cases:
        name = lookup.get((str(case["event_id"]), str(case["checkpoint_id"])))
        if name is None:
            raise KeyError("V127 D5 audit case lacks base causal input group")
        initial, rainfall, flow = _physical_inputs_streaming(cache, name, normalization, device)
        with torch.enable_grad():
            predicted, _ = predicted_directional_gradient_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_fractions=np.asarray(case["center_fractions"]),
                direction_fractions=np.asarray(case["direction_fractions"]),
                active_target=np.asarray(case["active_target"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
        rows.append(
            {
                "rainfall_group": str(case["rainfall_group"]),
                "event_id": str(case["event_id"]),
                "checkpoint_id": str(case["checkpoint_id"]),
                "center_id": str(case["center_id"]),
                "center_family": str(case.get("center_family", "")),
                "direction_id": str(case["direction_id"]),
                "direction_family": str(case.get("direction_family", "")),
                "true_tfv_gradient_m3_per_fraction": float(case["true_tfv_gradient"]),
                "predicted_tfv_gradient_m3_per_fraction": float(predicted.detach()),
            }
        )
    detail = pd.DataFrame(rows)
    per_rain: list[tuple[float, float, float]] = []
    for _, group in detail.groupby("rainfall_group", sort=True):
        truth = group["true_tfv_gradient_m3_per_fraction"].to_numpy(dtype=float)
        pred = group["predicted_tfv_gradient_m3_per_fraction"].to_numpy(dtype=float)
        mask = np.abs(truth) > 1.0e-8
        sign = (
            float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask])))
            if mask.any()
            else float("nan")
        )
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        cosine = float(np.dot(truth, pred) / denom) if denom > 1.0e-12 else float("nan")
        per_rain.append((sign, cosine, float(np.mean(np.abs(pred - truth)))))
    array = np.asarray(per_rain, dtype=float)
    if array.size == 0 or not np.isfinite(array[:, :2]).any(axis=0).all():
        raise RuntimeError("V127 streaming D5 audit cannot produce finite sign/cosine metrics")
    metrics: dict[str, float | str] = {
        "gradient_cases": float(len(detail)),
        "gradient_rainfall_groups": float(detail["rainfall_group"].nunique()),
        "tfv_gradient_sign_accuracy": float(np.nanmean(array[:, 0])),
        "tfv_gradient_cosine_similarity": float(np.nanmean(array[:, 1])),
        "tfv_gradient_mae_m3_per_fraction": float(np.nanmean(array[:, 2])),
        "gradient_variable_space": "exact online 12x109 L-BFGS-B fraction tensor",
        "audit_prediction_semantics": "smooth-TFV autograd through online V127 decoder",
        "training_prediction_semantics": "smooth-TFV symmetric finite difference on D5-FIT only",
        "execution_contract": V127_GRADIENT_STREAMING_CONTRACT,
        **_peak(device),
    }
    return detail, metrics


__all__ = [
    "V127_GRADIENT_STREAMING_CONTRACT",
    "evaluate_d5_gradients_streaming_v127",
    "train_d5_gradient_streaming_v127",
]
