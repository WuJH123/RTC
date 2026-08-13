"""Read-only diagnostics for the Step2 V4.1 response-calibration audit.

The helpers in this module operate on existing development/train arrays and frozen
checkpoints.  They do not launch SWMM, train a model, or inspect Validation/Final.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .step2_action_pathway_audit_v4 import direct_pair_delta_tfv

if TYPE_CHECKING:
    from torch import nn


def magnitude_statistics(values: np.ndarray, *, zero_atol: float = 0.0) -> dict[str, float | int]:
    """Return the frozen magnitude summary used by V4.1 target-scale tables."""

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError("magnitude statistics require at least one finite value")
    absolute = np.abs(finite)
    q25, q75 = np.quantile(absolute, (0.25, 0.75))
    return {
        "count": int(finite.size),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "median_abs": float(np.median(absolute)),
        "iqr_abs": float(q75 - q25),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
        "zero_fraction": float(np.mean(absolute <= float(zero_atol))),
    }


def cumulative_trapezoid_pair_delta_tfv(
    candidate_initial_rate: np.ndarray,
    reference_initial_rate: np.ndarray,
    candidate_future_rate: np.ndarray,
    reference_future_rate: np.ndarray,
    elapsed_seconds: np.ndarray,
) -> np.ndarray:
    """Integrate candidate-minus-reference flooding rate on each elapsed interval.

    The returned array contains the cumulative network volume difference after every
    future model step.  Candidate and reference rates are clipped independently to the
    physical non-negative domain before subtraction, while the resulting effect remains
    signed.
    """

    cand0 = np.asarray(candidate_initial_rate, dtype=np.float64)
    ref0 = np.asarray(reference_initial_rate, dtype=np.float64)
    cand = np.asarray(candidate_future_rate, dtype=np.float64)
    ref = np.asarray(reference_future_rate, dtype=np.float64)
    elapsed = np.asarray(elapsed_seconds, dtype=np.float64)
    if cand.shape != ref.shape or cand0.shape != ref0.shape:
        raise ValueError("candidate/reference flooding-rate shapes differ")
    if cand.ndim != 3 or cand0.shape != (cand.shape[0], cand.shape[2]):
        raise ValueError("rates must be initial [B,N] and future [B,H,N]")
    if elapsed.shape != (cand.shape[0], cand.shape[1] + 1):
        raise ValueError("elapsed_seconds must be [B,H+1]")
    if not all(np.isfinite(value).all() for value in (cand0, ref0, cand, ref, elapsed)):
        raise ValueError("TFV integration inputs must be finite")
    dt = np.diff(elapsed, axis=1)
    if np.any(dt <= 0.0):
        raise ValueError("elapsed_seconds must increase strictly")
    delta_initial = np.maximum(cand0, 0.0) - np.maximum(ref0, 0.0)
    delta_future = np.maximum(cand, 0.0) - np.maximum(ref, 0.0)
    previous = np.concatenate((delta_initial[:, None, :], delta_future[:, :-1, :]), axis=1)
    interval_volume = 0.5 * (previous + delta_future) * dt[:, :, None]
    return np.cumsum(interval_volume.sum(axis=2), axis=1)


def gradient_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite cosine, or NaN when either gradient is exactly zero."""

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("gradient vectors must have identical shapes")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return float("nan")
    result = float(np.dot(a, b) / (a_norm * b_norm))
    return min(1.0, max(-1.0, result)) if math.isfinite(result) else result


def reference_forward_accounting(*, candidate_count: int) -> dict[str, int | float]:
    """Expose the V4 pair-stack reference duplication for one checkpoint group."""

    count = int(candidate_count)
    if count <= 0:
        raise ValueError("candidate_count must be positive")
    current = 2 * count
    return {
        "candidate_count": count,
        "current_reference_forward_rows": current,
        "deduplicated_reference_forward_rows": 1,
        "reference_forward_reduction": float(current),
    }


def head_depth_consistency(states: np.ndarray, invert_elevation_m: np.ndarray) -> dict[str, float | int]:
    """Measure the physical identity ``head = invert elevation + depth``."""

    values = np.asarray(states, dtype=np.float64)
    invert = np.asarray(invert_elevation_m, dtype=np.float64).reshape(-1)
    if values.ndim < 3 or values.shape[-1] < 2:
        raise ValueError("states must end in node and state-channel axes")
    if values.shape[-2] != invert.size:
        raise ValueError("invert elevation count does not match the state node axis")
    shape = (1,) * (values.ndim - 2) + (invert.size,)
    residual = values[..., 1] - values[..., 0] - invert.reshape(shape)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        raise ValueError("head-depth consistency requires finite states")
    absolute = np.abs(finite)
    return {
        "count": int(finite.size),
        "mean_abs_residual_m": float(np.mean(absolute)),
        "rms_residual_m": float(np.sqrt(np.mean(np.square(finite)))),
        "p99_abs_residual_m": float(np.quantile(absolute, 0.99)),
        "max_abs_residual_m": float(np.max(absolute)),
        "within_1e_5_fraction": float(np.mean(absolute <= 1e-5)),
    }


def _parameter_group_entries(model: nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter, object | None]]]:
    named = dict(model.named_parameters())

    def prefixes(*values: str) -> list[tuple[str, torch.nn.Parameter, object | None]]:
        return [(name, parameter, None) for name, parameter in named.items() if name.startswith(values)]

    flooding: list[tuple[str, torch.nn.Parameter, object | None]] = []
    for name in (
        "reference_state_head.weight",
        "reference_state_head.bias",
        "effect_state_head.weight",
        "effect_state_head.bias",
    ):
        parameter = named.get(name)
        if parameter is None:
            continue
        rows = (2,) if name.startswith("reference_") else (2, int(model.state_dim) + 2)
        flooding.append((name, parameter, rows))
    return {
        "reference_encoder": prefixes("reference_state_encoder", "reference_state_head", "reference_flow_encoder", "reference_flow_head"),
        "actuator_encoder": prefixes("actuator_identity"),
        "action_effect_encoder": prefixes("effect_flow_encoder", "effect_state_encoder"),
        "trajectory_effect_head": prefixes("effect_flow_head", "effect_state_head"),
        "tfv_head": prefixes("direct_delta_tfv_head", "tfv_head"),
        "flooding_head": flooding,
    }


def parameter_group_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count the trainable entries addressed by each required loss-audit group."""

    result: dict[str, int] = {}
    for group, entries in _parameter_group_entries(model).items():
        count = 0
        for _name, parameter, rows in entries:
            count += int(parameter.numel()) if rows is None else int(parameter[list(rows)].numel())
        result[group] = count
    return result


def current_v4_loss_components(
    output: object,
    batch: dict[str, torch.Tensor],
    norm: object,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Reproduce V4 losses while exposing its non-authoritative TFV target.

    This is intentionally an audit of the frozen V4 implementation.  The additional
    authoritative loss is diagnostic only and is not included in the V4 weighted total.
    """

    ref = slice(0, None, 2)
    cand = slice(1, None, 2)
    pred_ref_state = output.reference_states[ref]
    pred_cand_state = output.candidate_states[cand]
    pred_ref_flow = output.reference_flows[ref]
    pred_cand_flow = output.candidate_flows[cand]
    true_ref_state = batch["target_states"][ref]
    true_cand_state = batch["target_states"][cand]
    true_ref_flow = batch["target_actuator_flows"][ref]
    true_cand_flow = batch["target_actuator_flows"][cand]
    state_mean = torch.as_tensor(
        norm.state_mean, device=pred_ref_state.device, dtype=pred_ref_state.dtype
    )
    state_std = torch.as_tensor(
        norm.state_std, device=pred_ref_state.device, dtype=pred_ref_state.dtype
    )
    pred_ref_physical = pred_ref_state * state_std + state_mean
    pred_cand_physical = pred_cand_state * state_std + state_mean
    true_ref_physical = batch["target_states_physical"][ref]
    true_cand_physical = batch["target_states_physical"][cand]
    dt = batch["elapsed_seconds"][ref][:, 1:] - batch["elapsed_seconds"][ref][:, :-1]
    smooth_scale = max(float(0.01 * np.asarray(norm.state_std)[2]), 1e-4)
    predicted_rate_delta = direct_pair_delta_tfv(
        pred_cand_physical[..., 2],
        pred_ref_physical[..., 2],
        dt_seconds=dt,
        smooth=True,
        softplus_scale=smooth_scale,
    )
    true_rate_delta = direct_pair_delta_tfv(
        true_cand_physical[..., 2],
        true_ref_physical[..., 2],
        dt_seconds=dt,
        smooth=False,
    ).detach()
    authoritative_delta = (
        batch["exact_node_flood_volume_m3"][cand].sum(dim=1)
        - batch["exact_node_flood_volume_m3"][ref].sum(dim=1)
    ).detach()
    current_scale = true_rate_delta.abs().clamp_min(1.0)
    authoritative_scale = authoritative_delta.abs().clamp_min(1.0)
    meaningful = true_rate_delta.abs() >= 1.0
    if bool(meaningful.any()):
        ranking = F.softplus(
            -true_rate_delta.sign() * predicted_rate_delta / current_scale
        )[meaningful].mean()
    else:
        ranking = predicted_rate_delta.sum() * 0.0
    components = {
        "absolute_state": F.mse_loss(pred_ref_state, true_ref_state)
        + F.mse_loss(pred_cand_state, true_cand_state),
        "absolute_flow": F.mse_loss(pred_ref_flow, true_ref_flow)
        + F.mse_loss(pred_cand_flow, true_cand_flow),
        "delta_state": F.mse_loss(
            pred_cand_state - pred_ref_state, true_cand_state - true_ref_state
        ),
        "delta_flow": F.mse_loss(
            pred_cand_flow - pred_ref_flow, true_cand_flow - true_ref_flow
        ),
        "delta_tfv_rate_rectangle": F.smooth_l1_loss(
            (predicted_rate_delta - true_rate_delta) / current_scale,
            torch.zeros_like(current_scale),
        ),
        "ranking_sign": ranking,
        "physical_nonnegative_penalty": F.relu(-pred_cand_physical[..., :4]).square().mean()
        + F.relu(-pred_ref_physical[..., :4]).square().mean(),
        "authoritative_exact_tfv_diagnostic": F.smooth_l1_loss(
            (predicted_rate_delta - authoritative_delta) / authoritative_scale,
            torch.zeros_like(authoritative_scale),
        ),
        "effect_energy_diagnostic": output.delta_states[cand].square().mean()
        + output.delta_flows[cand].square().mean(),
    }
    diagnostic = {
        "predicted_rate_rectangle_delta_tfv_m3": predicted_rate_delta,
        "true_rate_rectangle_delta_tfv_m3": true_rate_delta,
        "true_authoritative_delta_tfv_m3": authoritative_delta,
    }
    return components, diagnostic


__all__ = [
    "cumulative_trapezoid_pair_delta_tfv",
    "current_v4_loss_components",
    "gradient_cosine",
    "head_depth_consistency",
    "magnitude_statistics",
    "parameter_group_parameter_counts",
    "reference_forward_accounting",
]
