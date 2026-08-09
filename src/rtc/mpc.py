from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .models import DifferentiableHydraulicWorldModel, Rollout


@dataclass(frozen=True)
class StateLayout:
    depth_index: int
    flood_rate_index: int


@dataclass
class MPCResult:
    settings: torch.Tensor
    admissible: bool
    tfv_risk_m3: float
    worst_site_flood_deterioration_m3: float
    worst_site_depth_deterioration_m: float
    max_site_flood_margin_m3: float
    max_site_depth_margin_m: float


def _upper_tail_cvar(x: torch.Tensor, alpha: float) -> torch.Tensor:
    x = x.reshape(-1)
    count = max(1, int(round((1.0 - alpha) * x.numel())))
    return torch.topk(x, k=count, largest=True).values.mean()


def _flood_volumes(rollout: Rollout, flood_index: int, dt_seconds: float) -> torch.Tensor:
    rate = rollout.states[..., flood_index].clamp_min(0.0)
    return rate.sum(dim=1) * dt_seconds


def _site_tensor(value: float | torch.Tensor, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(n)
    if tensor.numel() != n:
        raise ValueError(f"expected one value or {n} site values, got {tensor.numel()}")
    return tensor


def _expand_control_blocks(block_settings: torch.Tensor, *, horizon: int, block_steps: int) -> torch.Tensor:
    if block_steps <= 0:
        raise ValueError("control_block_steps must be positive")
    expanded = torch.repeat_interleave(block_settings, repeats=block_steps, dim=1)
    return expanded[:, :horizon]


class ContinuousSafetyMPC:
    """Continuous all-actuator MPC with calibrated, site-wise priority safety.

    Forecast-scenario risk and independently calibrated model-error UCB are distinct.
    ``control_block_steps`` makes the optimization assume exactly the same action hold
    period that the authoritative runtime will execute.
    """

    def __init__(
        self,
        world_model: DifferentiableHydraulicWorldModel,
        *,
        layout: StateLayout,
        priority_indices: torch.Tensor,
        dt_seconds: float,
        per_site_flood_budget_m3: float | torch.Tensor,
        per_site_depth_budget_m: float | torch.Tensor,
        flood_error_ucb_m3: float | torch.Tensor = 0.0,
        depth_error_ucb_m: float | torch.Tensor = 0.0,
        forecast_safety_quantile: float = 0.95,
        tfv_cvar_alpha: float = 0.90,
        safety_penalty: float = 1e3,
        movement_tiebreak: float = 1e-5,
    ):
        if not 0.5 < forecast_safety_quantile < 1.0:
            raise ValueError("forecast_safety_quantile must lie in (0.5, 1)")
        self.model = world_model
        self.layout = layout
        self.priority_indices = priority_indices.long()
        self.dt_seconds = float(dt_seconds)
        self.per_site_flood_budget_m3 = per_site_flood_budget_m3
        self.per_site_depth_budget_m = per_site_depth_budget_m
        self.flood_error_ucb_m3 = flood_error_ucb_m3
        self.depth_error_ucb_m = depth_error_ucb_m
        self.forecast_safety_quantile = float(forecast_safety_quantile)
        self.tfv_cvar_alpha = float(tfv_cvar_alpha)
        self.safety_penalty = float(safety_penalty)
        self.movement_tiebreak = float(movement_tiebreak)

    def _rollout(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Rollout:
        scenarios = rainfall.shape[0]
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
        if settings.shape[0] != scenarios:
            raise ValueError("settings batch must be 1 or equal rainfall scenario count")
        return self.model.rollout(
            initial_state,
            rainfall,
            settings,
            previous_actuator_flow,
            actuator_upstream,
            actuator_downstream,
            actuator_physics,
            static_node_features,
            edge_index,
        )

    def _priority_ucbs(
        self,
        rollout: Rollout,
        fallback_volumes: torch.Tensor,
        fallback_priority_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        priority = self.priority_indices.to(fallback_volumes.device)
        volumes = _flood_volumes(rollout, self.layout.flood_rate_index, self.dt_seconds)
        site_delta = volumes[:, priority] - fallback_volumes[:, priority]
        priority_depth = rollout.states[..., self.layout.depth_index][..., priority]
        depth_delta = (priority_depth - fallback_priority_depth).amax(dim=1)
        flood_scenario_bound = torch.quantile(site_delta, self.forecast_safety_quantile, dim=0)
        depth_scenario_bound = torch.quantile(depth_delta, self.forecast_safety_quantile, dim=0)
        n = priority.numel()
        flood_error = _site_tensor(
            self.flood_error_ucb_m3, n, device=site_delta.device, dtype=site_delta.dtype
        )
        depth_error = _site_tensor(
            self.depth_error_ucb_m, n, device=depth_delta.device, dtype=depth_delta.dtype
        )
        return flood_scenario_bound + flood_error, depth_scenario_bound + depth_error

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
    ) -> MPCResult:
        horizon = int(rainfall_scenarios.shape[1])
        if control_block_steps <= 0:
            raise ValueError("control_block_steps must be positive")
        blocks = int(math.ceil(horizon / control_block_steps))
        current = current_settings.reshape(-1).clamp(1e-4, 1.0 - 1e-4)
        initial_logits = torch.logit(current).view(1, 1, -1).expand(1, blocks, -1).clone()
        logits = nn.Parameter(initial_logits)
        optimizer = torch.optim.Adam([logits], lr=learning_rate)

        with torch.no_grad():
            fallback = self._rollout(
                initial_state,
                rainfall_scenarios,
                fallback_settings,
                previous_actuator_flow,
                actuator_upstream,
                actuator_downstream,
                actuator_physics,
                static_node_features,
                edge_index,
            )
        fallback_volumes = _flood_volumes(fallback, self.layout.flood_rate_index, self.dt_seconds)
        priority = self.priority_indices.to(fallback_volumes.device)
        fallback_priority_depth = fallback.states[..., self.layout.depth_index][..., priority]
        n_sites = priority.numel()
        flood_budget = _site_tensor(
            self.per_site_flood_budget_m3,
            n_sites,
            device=fallback_volumes.device,
            dtype=fallback_volumes.dtype,
        )
        depth_budget = _site_tensor(
            self.per_site_depth_budget_m,
            n_sites,
            device=fallback_volumes.device,
            dtype=fallback_volumes.dtype,
        )

        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            block_settings = torch.sigmoid(logits)
            settings = _expand_control_blocks(
                block_settings, horizon=horizon, block_steps=control_block_steps
            )
            rollout = self._rollout(
                initial_state,
                rainfall_scenarios,
                settings,
                previous_actuator_flow,
                actuator_upstream,
                actuator_downstream,
                actuator_physics,
                static_node_features,
                edge_index,
            )
            volumes = _flood_volumes(rollout, self.layout.flood_rate_index, self.dt_seconds)
            tfv = volumes.sum(dim=-1)
            flood_ucb, depth_ucb = self._priority_ucbs(
                rollout, fallback_volumes, fallback_priority_depth
            )
            flood_excess = torch.relu(flood_ucb - flood_budget)
            depth_excess = torch.relu(depth_ucb - depth_budget)
            violation = flood_excess.square().mean() + depth_excess.square().mean()
            movement = (block_settings[:, 0] - current).square().mean()
            loss = (
                _upper_tail_cvar(tfv, self.tfv_cvar_alpha)
                + self.safety_penalty * violation
                + self.movement_tiebreak * movement
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            block_settings = torch.sigmoid(logits)
            settings = _expand_control_blocks(
                block_settings, horizon=horizon, block_steps=control_block_steps
            )
            rollout = self._rollout(
                initial_state,
                rainfall_scenarios,
                settings,
                previous_actuator_flow,
                actuator_upstream,
                actuator_downstream,
                actuator_physics,
                static_node_features,
                edge_index,
            )
            volumes = _flood_volumes(rollout, self.layout.flood_rate_index, self.dt_seconds)
            tfv = volumes.sum(dim=-1)
            flood_ucb, depth_ucb = self._priority_ucbs(
                rollout, fallback_volumes, fallback_priority_depth
            )
            flood_margin = flood_ucb - flood_budget
            depth_margin = depth_ucb - depth_budget
            admissible = bool((flood_margin <= 0).all().item() and (depth_margin <= 0).all().item())
            return MPCResult(
                settings=settings[0].detach(),
                admissible=admissible,
                tfv_risk_m3=float(_upper_tail_cvar(tfv, self.tfv_cvar_alpha)),
                worst_site_flood_deterioration_m3=float(flood_ucb.max()),
                worst_site_depth_deterioration_m=float(depth_ucb.max()),
                max_site_flood_margin_m3=float(flood_margin.max()),
                max_site_depth_margin_m=float(depth_margin.max()),
            )
