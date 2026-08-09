from __future__ import annotations

import torch


def trapezoid_node_flood_volume(
    initial_state: torch.Tensor,
    future_states: torch.Tensor,
    *,
    flood_rate_index: int,
    dt_seconds: float | torch.Tensor,
) -> torch.Tensor:
    """Integrate predicted flooding-rate states to cumulative node volume.

    ``initial_state`` is ``[B,N,F]`` and ``future_states`` is ``[B,H,N,F]``. External
    hydraulic state is SI, so flooding rate is m3/s and the result is m3. The initial
    current rate is included, preventing the right-endpoint-only bias that occurs when
    multiplying only future rates by ``dt``.
    """

    if initial_state.dim() != 3 or future_states.dim() != 4:
        raise ValueError("expected initial [B,N,F] and future [B,H,N,F]")
    if initial_state.shape[0] != future_states.shape[0] or initial_state.shape[1] != future_states.shape[2]:
        raise ValueError("initial/future batch-node dimensions do not align")
    initial_rate = initial_state[..., int(flood_rate_index)].clamp_min(0.0)
    future_rate = future_states[..., int(flood_rate_index)].clamp_min(0.0)
    rate = torch.cat([initial_rate[:, None], future_rate], dim=1)
    interval_mean = 0.5 * (rate[:, :-1] + rate[:, 1:])
    dt = torch.as_tensor(dt_seconds, device=rate.device, dtype=rate.dtype)
    if dt.numel() == 1:
        return interval_mean.sum(dim=1) * dt
    if dt.dim() == 1:
        if dt.shape[0] != future_states.shape[1]:
            raise ValueError("dt vector does not match horizon")
        dt = dt.view(1, -1, 1)
    elif dt.dim() == 2:
        if dt.shape != future_states.shape[:2]:
            raise ValueError("dt matrix does not match [B,H]")
        dt = dt[..., None]
    else:
        raise ValueError("dt_seconds must be scalar, [H] or [B,H]")
    if torch.any(dt <= 0):
        raise ValueError("dt_seconds must be positive")
    return (interval_mean * dt).sum(dim=1)
