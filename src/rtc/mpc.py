from __future__ import annotations

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
    priority_flood_deterioration_m3: float
    priority_depth_deterioration_m: float


def _upper_tail_cvar(x: torch.Tensor, alpha: float) -> torch.Tensor:
    x = x.reshape(-1)
    count = max(1, int(round((1.0 - alpha) * x.numel())))
    return torch.topk(x, k=count, largest=True).values.mean()


def _flood_volumes(rollout: Rollout, flood_index: int, dt_seconds: float) -> torch.Tensor:
    # [scenario,H,node] -> [scenario,node]
    rate = rollout.states[..., flood_index].clamp_min(0.0)
    return rate.sum(dim=1) * dt_seconds


class ContinuousSafetyMPC:
    """Step 3: projected continuous MPC over every discovered actuator.

    There is intentionally no hard Top-K actuator selector and no binary mask. Hydraulic
    inactivity can emerge through the learned setting-to-flow path. Safety penalties help
    the optimiser, but final admission is re-evaluated as a hard condition.
    """

    def __init__(
        self,
        world_model: DifferentiableHydraulicWorldModel,
        *,
        layout: StateLayout,
        priority_indices: torch.Tensor,
        dt_seconds: float,
        priority_flood_budget_m3: float,
        priority_depth_budget_m: float,
        safety_quantile: float = 0.95,
        tfv_cvar_alpha: float = 0.90,
        safety_penalty: float = 1e3,
        movement_tiebreak: float = 1e-5,
    ):
        self.model = world_model
        self.layout = layout
        self.priority_indices = priority_indices.long()
        self.dt_seconds = float(dt_seconds)
        self.priority_flood_budget_m3 = float(priority_flood_budget_m3)
        self.priority_depth_budget_m = float(priority_depth_budget_m)
        self.safety_quantile = float(safety_quantile)
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
        if initial_state.shape[0] == 1 and scenarios > 1:
            initial_state = initial_state.expand(scenarios, -1, -1)
        if previous_actuator_flow.shape[0] == 1 and scenarios > 1:
            previous_actuator_flow = previous_actuator_flow.expand(scenarios, -1)
        if actuator_physics.shape[0] == 1 and scenarios > 1:
            actuator_physics = actuator_physics.expand(scenarios, -1, -1)
        if settings.shape[0] == 1 and scenarios > 1:
            settings = settings.expand(scenarios, -1, -1)
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
    ) -> MPCResult:
        horizon = rainfall_scenarios.shape[1]
        current = current_settings.clamp(1e-4, 1.0 - 1e-4)
        initial_logits = torch.logit(current).view(1, 1, -1).expand(1, horizon, -1).clone()
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
        fallback_volumes = _flood_volumes(
            fallback, self.layout.flood_rate_index, self.dt_seconds
        )
        priority = self.priority_indices.to(fallback_volumes.device)
        fallback_pfv = fallback_volumes[:, priority].sum(dim=-1)
        fallback_priority_depth = fallback.states[..., self.layout.depth_index][..., priority]

        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            settings = torch.sigmoid(logits)
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
            pfv = volumes[:, priority].sum(dim=-1)
            priority_depth = rollout.states[..., self.layout.depth_index][..., priority]
            depth_delta = (priority_depth - fallback_priority_depth).amax(dim=(-2, -1))
            pfv_delta = pfv - fallback_pfv
            pfv_ucb = torch.quantile(pfv_delta, self.safety_quantile)
            depth_ucb = torch.quantile(depth_delta, self.safety_quantile)
            violation = torch.relu(pfv_ucb - self.priority_flood_budget_m3).square()
            violation = violation + torch.relu(
                depth_ucb - self.priority_depth_budget_m
            ).square()
            movement = (settings[:, 0] - current).square().mean()
            loss = (
                _upper_tail_cvar(tfv, self.tfv_cvar_alpha)
                + self.safety_penalty * violation
                + self.movement_tiebreak * movement
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            settings = torch.sigmoid(logits)
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
            pfv_delta = volumes[:, priority].sum(dim=-1) - fallback_pfv
            priority_depth = rollout.states[..., self.layout.depth_index][..., priority]
            depth_delta = (priority_depth - fallback_priority_depth).amax(dim=(-2, -1))
            pfv_ucb = torch.quantile(pfv_delta, self.safety_quantile)
            depth_ucb = torch.quantile(depth_delta, self.safety_quantile)
            admissible = bool(
                pfv_ucb <= self.priority_flood_budget_m3
                and depth_ucb <= self.priority_depth_budget_m
            )
            return MPCResult(
                settings=settings[0].detach(),
                admissible=admissible,
                tfv_risk_m3=float(_upper_tail_cvar(tfv, self.tfv_cvar_alpha)),
                priority_flood_deterioration_m3=float(pfv_ucb),
                priority_depth_deterioration_m=float(depth_ucb),
            )
