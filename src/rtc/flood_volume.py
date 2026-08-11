from __future__ import annotations

import torch
import torch.nn.functional as F


def _integrate_rate(
    initial_rate: torch.Tensor,
    future_rate: torch.Tensor,
    *,
    dt_seconds: float | torch.Tensor,
) -> torch.Tensor:
    rate = torch.cat([initial_rate[:, None], future_rate], dim=1)
    interval_mean = 0.5 * (rate[:, :-1] + rate[:, 1:])
    dt = torch.as_tensor(dt_seconds, device=rate.device, dtype=rate.dtype)
    if dt.numel() == 1:
        return interval_mean.sum(dim=1) * dt
    if dt.dim() == 1:
        if dt.shape[0] != future_rate.shape[1]:
            raise ValueError("dt vector does not match horizon")
        dt = dt.view(1, -1, 1)
    elif dt.dim() == 2:
        if dt.shape != future_rate.shape[:2]:
            raise ValueError("dt matrix does not match [B,H]")
        dt = dt[..., None]
    else:
        raise ValueError("dt_seconds must be scalar, [H] or [B,H]")
    if torch.any(dt <= 0):
        raise ValueError("dt_seconds must be positive")
    return (interval_mean * dt).sum(dim=1)


def trapezoid_node_flood_volume(
    initial_state: torch.Tensor,
    future_states: torch.Tensor,
    *,
    flood_rate_index: int,
    dt_seconds: float | torch.Tensor,
) -> torch.Tensor:
    """Integrate physical flooding-rate states to cumulative node volume.

    ``initial_state`` is ``[B,N,F]`` and ``future_states`` is ``[B,H,N,F]``. External
    hydraulic state is SI, so flooding rate is m3/s and the result is m3. Negative
    predicted rates are clipped because authoritative physical flooding volume cannot be
    negative. This hard physical operator is appropriate for reporting and exact-truth
    comparisons, but it has a zero derivative below zero; use
    :func:`smooth_trapezoid_node_flood_volume` for an optimization/training proxy.
    """

    if initial_state.dim() != 3 or future_states.dim() != 4:
        raise ValueError("expected initial [B,N,F] and future [B,H,N,F]")
    if initial_state.shape[0] != future_states.shape[0] or initial_state.shape[1] != future_states.shape[2]:
        raise ValueError("initial/future batch-node dimensions do not align")
    initial_rate = initial_state[..., int(flood_rate_index)].clamp_min(0.0)
    future_rate = future_states[..., int(flood_rate_index)].clamp_min(0.0)
    return _integrate_rate(initial_rate, future_rate, dt_seconds=dt_seconds)


def smooth_trapezoid_node_flood_volume(
    initial_state: torch.Tensor,
    future_states: torch.Tensor,
    *,
    flood_rate_index: int,
    dt_seconds: float | torch.Tensor,
    softplus_scale_m3s: float | torch.Tensor,
) -> torch.Tensor:
    """Differentiable positive flooding-volume proxy for training/optimization.

    A hard ``clamp_min(0)`` creates an action-gradient dead zone whenever a recurrent
    surrogate predicts a slightly negative flooding rate. For same-prefix candidate vs.
    reference learning this can make ``delta TFV`` and ranking losses report an error while
    providing no hydraulic gradient. This proxy replaces the hard clamp with a Softplus in
    physical m3/s units. Its positive zero-offset is common to a fixed node/time grid and
    therefore cancels in counterfactual differences; it is *not* authoritative SWMM TFV and
    must not replace the hard physical operator in evaluation evidence.
    """

    if initial_state.dim() != 3 or future_states.dim() != 4:
        raise ValueError("expected initial [B,N,F] and future [B,H,N,F]")
    if initial_state.shape[0] != future_states.shape[0] or initial_state.shape[1] != future_states.shape[2]:
        raise ValueError("initial/future batch-node dimensions do not align")
    scale = torch.as_tensor(
        softplus_scale_m3s,
        device=future_states.device,
        dtype=future_states.dtype,
    ).reshape(())
    scale = scale.clamp_min(1e-6)
    initial_raw = initial_state[..., int(flood_rate_index)]
    future_raw = future_states[..., int(flood_rate_index)]
    initial_rate = F.softplus(initial_raw / scale) * scale
    future_rate = F.softplus(future_raw / scale) * scale
    return _integrate_rate(initial_rate, future_rate, dt_seconds=dt_seconds)
