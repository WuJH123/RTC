"""Loss and robustness augmentation for Project7 Step1 V12.2."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Step1LossContractV122:
    global_state_weight: float = 0.35
    unobserved_state_weight: float = 0.40
    unobserved_depth_weight: float = 0.15
    flood_active_weight: float = 0.10
    sensor_dropout_probability: float = 0.10
    flood_active_threshold_m3s: float = 1.0e-6

    def validate(self) -> None:
        weights = (
            self.global_state_weight,
            self.unobserved_state_weight,
            self.unobserved_depth_weight,
            self.flood_active_weight,
        )
        if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1.0e-9:
            raise ValueError("V122 Step1 loss weights must be non-negative and sum to one")
        if not 0.0 <= self.sensor_dropout_probability < 1.0:
            raise ValueError("invalid V122 sensor dropout probability")
        if self.flood_active_threshold_m3s < 0.0:
            raise ValueError("invalid V122 flooding threshold")


def apply_sensor_dropout_v122(
    observed_history: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop whole sensor histories, never individual future frames.

    A sensor selected for dropout is removed from all 13 causal frames, which emulates
    a missing field device rather than fabricating temporal intermittency.  At least one
    sensor is retained in every sample.
    """

    if observed_history.shape != observation_mask.shape:
        raise ValueError("Step1 observation/mask shapes differ")
    if not 0.0 <= probability < 1.0:
        raise ValueError("sensor dropout probability must lie in [0,1)")
    if probability == 0.0:
        return observed_history, observation_mask
    batch, _, nodes, channels = observation_mask.shape
    present = observation_mask.amax(dim=(1, 3)) > 0
    random = torch.rand((batch, nodes), device=observation_mask.device, generator=generator)
    keep = (~present) | (random >= float(probability))
    # Guarantee at least one originally observed sensor survives per sample.
    for b in range(batch):
        if not bool(torch.any(present[b] & keep[b])):
            first = int(torch.nonzero(present[b], as_tuple=False)[0].item())
            keep[b, first] = True
    keep4 = keep[:, None, :, None].expand(batch, observation_mask.shape[1], nodes, channels)
    dropped_mask = observation_mask * keep4.to(observation_mask.dtype)
    dropped_observed = observed_history * dropped_mask
    return dropped_observed, dropped_mask


def step1_loss_v122(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observation_mask: torch.Tensor,
    state_std: torch.Tensor,
    *,
    contract: Step1LossContractV122 = Step1LossContractV122(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Prioritise what Step1 exists to infer: unobserved hydraulically active nodes."""

    contract.validate()
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("V122 Step1 prediction/target must be aligned [B,N,D]")
    if observation_mask.ndim != 4 or observation_mask.shape[0] != target.shape[0] or observation_mask.shape[2] != target.shape[1]:
        raise ValueError("V122 Step1 observation mask is incompatible")
    scale = state_std.reshape(1, 1, -1).to(prediction).clamp_min(1.0e-6)
    error2 = torch.square((prediction - target) / scale)
    global_loss = error2.mean()

    sensor_node = observation_mask[:, -1].amax(dim=-1) > 0
    unobserved = ~sensor_node
    if bool(unobserved.any()):
        unobserved_state = error2[unobserved].mean()
        unobserved_depth = error2[..., 0][unobserved].mean()
    else:
        unobserved_state = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        unobserved_depth = torch.zeros_like(unobserved_state)

    # Channel 2 is authoritative SWMM node flooding rate in SI m3/s.
    flood_active = (target[..., 2] > float(contract.flood_active_threshold_m3s)) & unobserved
    if bool(flood_active.any()):
        flood_active_loss = error2[flood_active].mean()
    else:
        flood_active_loss = torch.zeros((), dtype=prediction.dtype, device=prediction.device)

    total = (
        contract.global_state_weight * global_loss
        + contract.unobserved_state_weight * unobserved_state
        + contract.unobserved_depth_weight * unobserved_depth
        + contract.flood_active_weight * flood_active_loss
    )
    return total, {
        "loss": float(total.detach()),
        "global_state": float(global_loss.detach()),
        "unobserved_state": float(unobserved_state.detach()),
        "unobserved_depth": float(unobserved_depth.detach()),
        "flood_active_unobserved": float(flood_active_loss.detach()),
        "unobserved_fraction": float(unobserved.float().mean().detach()),
        "flood_active_unobserved_fraction": float(flood_active.float().mean().detach()),
    }


__all__ = [
    "Step1LossContractV122",
    "apply_sensor_dropout_v122",
    "step1_loss_v122",
]
