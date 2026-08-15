from __future__ import annotations

from collections import defaultdict
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
    _objective_output,
    _predicted_symmetric_difference_v127,
    _scale,
    _setting_bounds,
)
from .step2_gradient_v127_streaming import _peak, _physical_inputs_streaming
from .step2_train_response_v60 import InputNormalizationV60
from .step3_mpc_v127 import ContinuousMPCDesignV127, decode_fractional_targets_v127

V127_GRADIENT_FAST_CONTRACT = (
    "PROJECT7_V127_D5_FAST_V1_INPUT_CACHE_CENTER_GRADIENT_REUSE"
)


def _cached_inputs(
    cache: Any,
    *,
    lookup: dict[tuple[str, str], str],
    cases: Sequence[dict[str, Any]],
    normalization: InputNormalizationV60,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    result: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for case in cases:
        name = lookup.get((str(case["event_id"]), str(case["checkpoint_id"])))
        if name is None:
            raise KeyError("V127 D5 case lacks base causal input group")
        if name not in result:
            result[name] = _physical_inputs_streaming(cache, name, normalization, device)
    return result


def _case_name(
    lookup: dict[tuple[str, str], str], case: dict[str, Any]
) -> str:
    name = lookup.get((str(case["event_id"]), str(case["checkpoint_id"])))
    if name is None:
        raise KeyError("V127 D5 case lacks base causal input group")
    return name


def train_d5_gradient_fast_v127(
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
    """Same per-case optimizer schedule as V127 streaming D5, without repeated input loading."""
    design.validate()
    scale = _scale(cases)
    lookup = _lookup(cache)
    inputs = _cached_inputs(
        cache,
        lookup=lookup,
        cases=cases,
        normalization=normalization,
        device=device,
    )
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
            name = _case_name(lookup, case)
            initial, rainfall, flow = inputs[name]
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
                raise RuntimeError("V127 fast D5 gradient loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            if abs(float(case["true_tfv_gradient"])) > 1.0e-8:
                hits.append(float(torch.sign(predicted.detach()) == torch.sign(truth)))
        row: dict[str, float | int | str] = {
            "stage": "D5_1308var_symmetric_gradient_finetune_fast",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_sign_accuracy": float(np.mean(hits)) if hits else float("nan"),
            "gradient_scale_m3_per_fraction": scale,
            "second_order_parameter_backprop": "false",
            "execution_contract": V127_GRADIENT_FAST_CONTRACT,
            "cached_causal_input_groups": int(len(inputs)),
            **_peak(device),
        }
        history.append(row)
        print("[V127_D5_FAST] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def _full_fraction_gradient(
    model,
    *,
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    center_fractions: np.ndarray,
    active_target: np.ndarray,
    graph: Any,
    flood_rate_index: int,
) -> torch.Tensor:
    device = initial_state.device
    lo, hi = _setting_bounds(graph, dtype=initial_state.dtype, device=device)
    fractions = torch.as_tensor(
        center_fractions, dtype=initial_state.dtype, device=device
    ).requires_grad_(True)
    mpc_design = ContinuousMPCDesignV127(
        min_improvement_vs_rbc_m3=0.0,
        movement_penalty_m3=0.0,
    )
    sequence = decode_fractional_targets_v127(
        fractions,
        active_target=torch.as_tensor(active_target, dtype=initial_state.dtype, device=device),
        min_setting=lo,
        max_setting=hi,
        design=mpc_design,
    )
    output = _objective_output(
        model,
        initial_state=initial_state,
        rainfall=rainfall,
        previous_flow=previous_flow,
        settings=sequence[None],
        graph=graph,
        flood_rate_index=flood_rate_index,
    )
    gradient = torch.autograd.grad(output.optimization_tfv_m3.sum(), fractions)[0]
    if gradient.shape != fractions.shape or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("V127 D5 fast audit produced invalid full fraction gradient")
    return gradient.detach()


def evaluate_d5_gradients_fast_v127(
    model,
    *,
    cache: Any,
    cases: Sequence[dict[str, Any]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Reuse one exact 12x109 autograd tensor for every direction sharing a D5 center."""
    lookup = _lookup(cache)
    inputs = _cached_inputs(
        cache,
        lookup=lookup,
        cases=cases,
        normalization=normalization,
        device=device,
    )
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    by_center: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_center[str(case["center_id"])].append(case)
    predicted_by_direction: dict[str, float] = {}
    center_rollouts = 0
    for center_id, center_cases in by_center.items():
        first = center_cases[0]
        name = _case_name(lookup, first)
        initial, rainfall, flow = inputs[name]
        center_fraction = np.asarray(first["center_fractions"])
        active_target = np.asarray(first["active_target"])
        for other in center_cases[1:]:
            if _case_name(lookup, other) != name:
                raise RuntimeError(f"V127 D5 center {center_id} spans multiple causal inputs")
            if not np.allclose(np.asarray(other["center_fractions"]), center_fraction, rtol=0.0, atol=0.0):
                raise RuntimeError(f"V127 D5 center {center_id} has inconsistent fractions")
            if not np.allclose(np.asarray(other["active_target"]), active_target, rtol=0.0, atol=0.0):
                raise RuntimeError(f"V127 D5 center {center_id} has inconsistent active target")
        with torch.enable_grad():
            full = _full_fraction_gradient(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_fractions=center_fraction,
                active_target=active_target,
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
        center_rollouts += 1
        for case in center_cases:
            direction = torch.as_tensor(
                np.asarray(case["direction_fractions"]), dtype=full.dtype, device=full.device
            )
            predicted_by_direction[str(case["direction_id"])] = float(torch.sum(full * direction))

    rows: list[dict[str, Any]] = []
    for case in cases:
        did = str(case["direction_id"])
        rows.append(
            {
                "rainfall_group": str(case["rainfall_group"]),
                "event_id": str(case["event_id"]),
                "checkpoint_id": str(case["checkpoint_id"]),
                "center_id": str(case["center_id"]),
                "center_family": str(case.get("center_family", "")),
                "direction_id": did,
                "direction_family": str(case.get("direction_family", "")),
                "true_tfv_gradient_m3_per_fraction": float(case["true_tfv_gradient"]),
                "predicted_tfv_gradient_m3_per_fraction": float(predicted_by_direction[did]),
            }
        )
    detail = pd.DataFrame(rows)
    per_rain: list[tuple[float, float, float]] = []
    for _, group in detail.groupby("rainfall_group", sort=True):
        truth = group["true_tfv_gradient_m3_per_fraction"].to_numpy(dtype=float)
        pred = group["predicted_tfv_gradient_m3_per_fraction"].to_numpy(dtype=float)
        mask = np.abs(truth) > 1.0e-8
        sign = float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask]))) if mask.any() else float("nan")
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        cosine = float(np.dot(truth, pred) / denom) if denom > 1.0e-12 else float("nan")
        per_rain.append((sign, cosine, float(np.mean(np.abs(pred - truth)))))
    array = np.asarray(per_rain, dtype=float)
    if array.size == 0 or not np.isfinite(array[:, :2]).any(axis=0).all():
        raise RuntimeError("V127 fast D5 audit cannot produce finite sign/cosine metrics")
    metrics: dict[str, float | str] = {
        "gradient_cases": float(len(detail)),
        "gradient_centers": float(len(by_center)),
        "gradient_rainfall_groups": float(detail["rainfall_group"].nunique()),
        "tfv_gradient_sign_accuracy": float(np.nanmean(array[:, 0])),
        "tfv_gradient_cosine_similarity": float(np.nanmean(array[:, 1])),
        "tfv_gradient_mae_m3_per_fraction": float(np.nanmean(array[:, 2])),
        "gradient_variable_space": "exact online 12x109 L-BFGS-B fraction tensor",
        "audit_prediction_semantics": "full smooth-TFV fraction autograd projected onto frozen D5 directions",
        "training_prediction_semantics": "smooth-TFV symmetric finite difference on D5-FIT only",
        "execution_contract": V127_GRADIENT_FAST_CONTRACT,
        "surrogate_h360_autograd_rollouts": float(center_rollouts),
        "center_gradient_reused_across_directions": "true",
        "cached_causal_input_groups": float(len(inputs)),
        **_peak(device),
    }
    return detail, metrics


__all__ = [
    "V127_GRADIENT_FAST_CONTRACT",
    "evaluate_d5_gradients_fast_v127",
    "train_d5_gradient_fast_v127",
]
