from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .flood_volume import trapezoid_node_flood_volume
from .models import DifferentiableHydraulicWorldModel, Rollout


@dataclass(frozen=True)
class TFVFirstMPCResult:
    settings: torch.Tensor
    candidate_valid: bool
    tfv_risk_m3: float
    primary_tfv_reference_m3: float
    priority_positive_flood_deterioration_m3: float
    worst_site_flood_deterioration_m3: float
    worst_site_depth_deterioration_m: float
    tfv_near_opt_excess_m3: float


def _upper_tail_cvar(x: torch.Tensor, alpha: float) -> torch.Tensor:
    values = x.reshape(-1)
    if values.numel() == 0:
        raise ValueError("empty CVaR input")
    count = max(1, int(math.ceil((1.0 - alpha) * values.numel())))
    return torch.topk(values, k=count, largest=True).values.mean()


def _expand_blocks(block_settings: torch.Tensor, *, horizon: int, block_steps: int) -> torch.Tensor:
    if block_steps <= 0:
        raise ValueError("control_block_steps must be positive")
    return torch.repeat_interleave(block_settings, repeats=block_steps, dim=1)[:, :horizon]


def _site_vector(value: float | torch.Tensor, n: int, *, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(n)
    if tensor.numel() != n:
        raise ValueError(f"expected one value or {n} site values, got {tensor.numel()}")
    return tensor


class ContinuousTFVFirstMPC:
    """All-actuator continuous MPC: TFV primary, priority PFV soft secondary.

    Predicted cumulative flood volume is trapezoid-integrated from the *current* flooding
    rate plus all future predicted rates. This keeps instantaneous rate and cumulative
    volume semantics distinct and reduces right-endpoint integration bias.
    """

    def __init__(
        self,
        world_model: DifferentiableHydraulicWorldModel,
        *,
        depth_index: int,
        flood_rate_index: int,
        priority_indices: torch.Tensor | None,
        dt_seconds: float,
        flood_error_ucb_m3: float | torch.Tensor = 0.0,
        depth_error_ucb_m: float | torch.Tensor = 0.0,
        forecast_quantile: float = 0.95,
        tfv_cvar_alpha: float = 0.90,
        tfv_near_opt_relative: float = 0.01,
        tfv_near_opt_absolute_m3: float = 1.0,
        near_opt_penalty: float = 1e4,
        movement_tiebreak: float = 1e-6,
    ):
        if not 0.5 < forecast_quantile < 1.0 or not 0.5 < tfv_cvar_alpha < 1.0:
            raise ValueError("forecast/CVaR quantiles must lie in (0.5,1)")
        if tfv_near_opt_relative < 0 or tfv_near_opt_absolute_m3 < 0:
            raise ValueError("TFV near-optimality tolerances must be non-negative")
        self.model = world_model
        self.depth_index = int(depth_index)
        self.flood_rate_index = int(flood_rate_index)
        self.priority_indices = None if priority_indices is None else priority_indices.long()
        self.dt_seconds = float(dt_seconds)
        self.flood_error_ucb_m3 = flood_error_ucb_m3
        self.depth_error_ucb_m = depth_error_ucb_m
        self.forecast_quantile = float(forecast_quantile)
        self.tfv_cvar_alpha = float(tfv_cvar_alpha)
        self.tfv_near_opt_relative = float(tfv_near_opt_relative)
        self.tfv_near_opt_absolute_m3 = float(tfv_near_opt_absolute_m3)
        self.near_opt_penalty = float(near_opt_penalty)
        self.movement_tiebreak = float(movement_tiebreak)

    def _rollout(self, initial_state, rainfall, settings, previous_actuator_flow, actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index) -> tuple[torch.Tensor, Rollout]:
        scenarios = int(rainfall.shape[0])
        if settings.dim() == 2:
            settings = settings.unsqueeze(0)
        if initial_state.shape[0] == 1 and scenarios > 1:
            initial_state = initial_state.expand(scenarios, -1, -1)
        if previous_actuator_flow.shape[0] == 1 and scenarios > 1:
            previous_actuator_flow = previous_actuator_flow.expand(scenarios, -1)
        if actuator_physics.shape[0] == 1 and scenarios > 1:
            actuator_physics = actuator_physics.expand(scenarios, -1, -1)
        if settings.shape[0] == 1 and scenarios > 1:
            settings = settings.expand(scenarios, -1, -1)
        return initial_state, self.model.rollout(
            initial_state, rainfall, settings, previous_actuator_flow,
            actuator_upstream, actuator_downstream, actuator_physics,
            static_node_features, edge_index,
        )

    def _volumes(self, initial_state: torch.Tensor, rollout: Rollout) -> torch.Tensor:
        return trapezoid_node_flood_volume(
            initial_state, rollout.states,
            flood_rate_index=self.flood_rate_index, dt_seconds=self.dt_seconds,
        )

    def _priority_deterioration(
        self,
        initial_state: torch.Tensor,
        rollout: Rollout,
        fallback_volumes: torch.Tensor,
        fallback_depth: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.priority_indices is None or self.priority_indices.numel() == 0:
            zero = fallback_volumes.new_zeros(())
            return zero, zero, zero
        p = self.priority_indices.to(fallback_volumes.device)
        volumes = self._volumes(initial_state, rollout)
        delta_v = volumes[:, p] - fallback_volumes[:, p]
        flood_bound = torch.quantile(delta_v, self.forecast_quantile, dim=0)
        flood_bound = flood_bound + _site_vector(
            self.flood_error_ucb_m3, p.numel(), device=delta_v.device, dtype=delta_v.dtype
        )
        positive_flood = torch.relu(flood_bound).sum()
        depth = rollout.states[..., self.depth_index][..., p]
        if fallback_depth is None:
            raise ValueError("fallback priority depth is required when priority sites are configured")
        delta_h = (depth - fallback_depth).amax(dim=1)
        depth_bound = torch.quantile(delta_h, self.forecast_quantile, dim=0)
        depth_bound = depth_bound + _site_vector(
            self.depth_error_ucb_m, p.numel(), device=delta_h.device, dtype=delta_h.dtype
        )
        return positive_flood, flood_bound.max(), depth_bound.max()

    def optimize(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        current_settings: torch.Tensor,
        fallback_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        iterations: int = 120,
        learning_rate: float = 0.04,
        control_block_steps: int = 1,
    ) -> TFVFirstMPCResult:
        if iterations < 2:
            raise ValueError("iterations must be >=2")
        horizon = int(rainfall_scenarios.shape[1])
        blocks = int(math.ceil(horizon / control_block_steps))
        current = current_settings.reshape(-1).clamp(1e-4, 1.0 - 1e-4)
        logits = nn.Parameter(torch.logit(current).view(1, 1, -1).expand(1, blocks, -1).clone())
        optimiser = torch.optim.Adam([logits], lr=float(learning_rate))

        with torch.no_grad():
            fallback_initial, fallback = self._rollout(
                initial_state, rainfall_scenarios, fallback_settings, previous_actuator_flow,
                actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index,
            )
            fallback_volumes = self._volumes(fallback_initial, fallback)
            fallback_depth = None
            if self.priority_indices is not None and self.priority_indices.numel():
                p = self.priority_indices.to(fallback_volumes.device)
                fallback_depth = fallback.states[..., self.depth_index][..., p]

        primary_iterations = max(1, int(round(iterations * 0.7)))
        for _ in range(primary_iterations):
            optimiser.zero_grad(set_to_none=True)
            block_value = torch.sigmoid(logits)
            settings = _expand_blocks(block_value, horizon=horizon, block_steps=control_block_steps)
            candidate_initial, rollout = self._rollout(
                initial_state, rainfall_scenarios, settings, previous_actuator_flow,
                actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index,
            )
            tfv = self._volumes(candidate_initial, rollout).sum(dim=-1)
            loss = _upper_tail_cvar(tfv, self.tfv_cvar_alpha) + self.movement_tiebreak * (block_value[:, 0] - current).square().mean()
            if not torch.isfinite(loss):
                break
            loss.backward(); optimiser.step()

        with torch.no_grad():
            primary_settings = _expand_blocks(torch.sigmoid(logits), horizon=horizon, block_steps=control_block_steps)
            primary_initial, primary_rollout = self._rollout(
                initial_state, rainfall_scenarios, primary_settings, previous_actuator_flow,
                actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index,
            )
            primary_tfv = _upper_tail_cvar(self._volumes(primary_initial, primary_rollout).sum(dim=-1), self.tfv_cvar_alpha)
            allowed_tfv = primary_tfv + self.tfv_near_opt_absolute_m3 + self.tfv_near_opt_relative * primary_tfv.abs()

        for _ in range(max(0, iterations - primary_iterations)):
            optimiser.zero_grad(set_to_none=True)
            block_value = torch.sigmoid(logits)
            settings = _expand_blocks(block_value, horizon=horizon, block_steps=control_block_steps)
            candidate_initial, rollout = self._rollout(
                initial_state, rainfall_scenarios, settings, previous_actuator_flow,
                actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index,
            )
            tfv = _upper_tail_cvar(self._volumes(candidate_initial, rollout).sum(dim=-1), self.tfv_cvar_alpha)
            priority_soft, _, _ = self._priority_deterioration(candidate_initial, rollout, fallback_volumes, fallback_depth)
            loss = (
                priority_soft
                + self.near_opt_penalty * torch.relu(tfv - allowed_tfv).square()
                + self.movement_tiebreak * (block_value[:, 0] - current).square().mean()
            )
            if not torch.isfinite(loss):
                break
            loss.backward(); optimiser.step()

        with torch.no_grad():
            settings = _expand_blocks(torch.sigmoid(logits), horizon=horizon, block_steps=control_block_steps)
            candidate_initial, rollout = self._rollout(
                initial_state, rainfall_scenarios, settings, previous_actuator_flow,
                actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index,
            )
            tfv = _upper_tail_cvar(self._volumes(candidate_initial, rollout).sum(dim=-1), self.tfv_cvar_alpha)
            soft, worst_flood, worst_depth = self._priority_deterioration(
                candidate_initial, rollout, fallback_volumes, fallback_depth
            )
            candidate_valid = bool(torch.isfinite(settings).all().item() and torch.isfinite(tfv).item() and torch.isfinite(soft).item())
            return TFVFirstMPCResult(
                settings=settings[0].detach(),
                candidate_valid=candidate_valid,
                tfv_risk_m3=float(tfv),
                primary_tfv_reference_m3=float(primary_tfv),
                priority_positive_flood_deterioration_m3=float(soft),
                worst_site_flood_deterioration_m3=float(worst_flood),
                worst_site_depth_deterioration_m=float(worst_depth),
                tfv_near_opt_excess_m3=float(torch.relu(tfv - allowed_tfv)),
            )
