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


def _expand_blocks(
    block_settings: torch.Tensor, *, horizon: int, block_steps: int
) -> torch.Tensor:
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


def _delta_vector(
    value: float | torch.Tensor | None,
    actuator_count: int,
    *,
    device,
    dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    delta = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if delta.numel() == 1:
        delta = delta.expand(actuator_count)
    if delta.numel() != actuator_count or torch.any(delta < 0):
        raise ValueError("max setting delta must be non-negative scalar or actuator vector")
    return delta


@torch.no_grad()
def _project_block_settings_(
    block_settings: torch.Tensor,
    *,
    current_settings: torch.Tensor,
    max_delta_per_update: float | torch.Tensor | None,
) -> None:
    """Project all future control blocks to a sequentially executable setting path."""

    if block_settings.ndim != 3 or block_settings.shape[0] != 1:
        raise ValueError("block_settings must have shape [1, blocks, actuator]")
    current = current_settings.reshape(-1).to(
        device=block_settings.device, dtype=block_settings.dtype
    )
    if block_settings.shape[-1] != current.numel():
        raise ValueError("current/block actuator dimensions differ")
    delta = _delta_vector(
        max_delta_per_update,
        current.numel(),
        device=block_settings.device,
        dtype=block_settings.dtype,
    )
    previous = current
    for block in range(block_settings.shape[1]):
        value = block_settings[0, block].clamp(0.0, 1.0)
        if delta is not None:
            value = torch.maximum(value, previous - delta)
            value = torch.minimum(value, previous + delta)
            value = value.clamp(0.0, 1.0)
        block_settings[0, block].copy_(value)
        previous = value


class ContinuousTFVFirstMPC:
    """All-actuator continuous MPC: TFV primary, priority PFV soft secondary.

    Holding current settings is always an executable fallback sequence. The optimizer keeps
    the best primary iterate it has actually seen and, when priority sites are configured,
    the best secondary iterate inside the TFV near-optimal set. A candidate is executable
    only if its predicted TFV beats the explicit fallback sequence by the configured small
    improvement margin. This is an engineering dominance check, not a separate safety gate.
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
        min_predicted_tfv_improvement_m3: float = 0.0,
        min_predicted_tfv_improvement_relative: float = 0.0,
    ):
        if not 0.5 < forecast_quantile < 1.0 or not 0.5 < tfv_cvar_alpha < 1.0:
            raise ValueError("forecast/CVaR quantiles must lie in (0.5,1)")
        if tfv_near_opt_relative < 0 or tfv_near_opt_absolute_m3 < 0:
            raise ValueError("TFV near-optimality tolerances must be non-negative")
        if dt_seconds <= 0:
            raise ValueError("MPC dt_seconds must be positive")
        if min_predicted_tfv_improvement_m3 < 0:
            raise ValueError("minimum predicted TFV improvement must be non-negative")
        if min_predicted_tfv_improvement_relative < 0:
            raise ValueError("relative predicted TFV improvement must be non-negative")
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
        self.min_predicted_tfv_improvement_m3 = float(
            min_predicted_tfv_improvement_m3
        )
        self.min_predicted_tfv_improvement_relative = float(
            min_predicted_tfv_improvement_relative
        )

    def _rollout(
        self,
        initial_state,
        rainfall,
        settings,
        previous_actuator_flow,
        actuator_upstream,
        actuator_downstream,
        actuator_physics,
        static_node_features,
        edge_index,
    ) -> tuple[torch.Tensor, Rollout]:
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

    def _volumes(self, initial_state: torch.Tensor, rollout: Rollout) -> torch.Tensor:
        return trapezoid_node_flood_volume(
            initial_state,
            rollout.states,
            flood_rate_index=self.flood_rate_index,
            dt_seconds=self.dt_seconds,
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
            self.flood_error_ucb_m3,
            p.numel(),
            device=delta_v.device,
            dtype=delta_v.dtype,
        )
        positive_flood = torch.relu(flood_bound).sum()
        depth = rollout.states[..., self.depth_index][..., p]
        if fallback_depth is None:
            raise ValueError("fallback priority depth is required when priority sites are configured")
        delta_h = (depth - fallback_depth).amax(dim=1)
        depth_bound = torch.quantile(delta_h, self.forecast_quantile, dim=0)
        depth_bound = depth_bound + _site_vector(
            self.depth_error_ucb_m,
            p.numel(),
            device=delta_h.device,
            dtype=delta_h.dtype,
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
        max_setting_delta_per_update: float | torch.Tensor | None = None,
    ) -> TFVFirstMPCResult:
        if iterations < 2:
            raise ValueError("iterations must be >=2")
        horizon = int(rainfall_scenarios.shape[1])
        blocks = int(math.ceil(horizon / control_block_steps))
        current = current_settings.reshape(-1).clamp(0.0, 1.0)
        block_parameter = nn.Parameter(
            current.view(1, 1, -1).expand(1, blocks, -1).clone()
        )
        _project_block_settings_(
            block_parameter,
            current_settings=current,
            max_delta_per_update=max_setting_delta_per_update,
        )
        optimiser = torch.optim.Adam([block_parameter], lr=float(learning_rate))

        def rollout_for(blocks_value: torch.Tensor):
            settings = _expand_blocks(
                blocks_value, horizon=horizon, block_steps=control_block_steps
            )
            candidate_initial, rollout = self._rollout(
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
            tfv = _upper_tail_cvar(
                self._volumes(candidate_initial, rollout).sum(dim=-1),
                self.tfv_cvar_alpha,
            )
            return settings, candidate_initial, rollout, tfv

        with torch.no_grad():
            fallback_initial, fallback = self._rollout(
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
            fallback_volumes = self._volumes(fallback_initial, fallback)
            fallback_tfv = _upper_tail_cvar(
                fallback_volumes.sum(dim=-1), self.tfv_cvar_alpha
            )
            fallback_depth = None
            if self.priority_indices is not None and self.priority_indices.numel():
                p = self.priority_indices.to(fallback_volumes.device)
                fallback_depth = fallback.states[..., self.depth_index][..., p]

            _, _, _, initial_tfv = rollout_for(block_parameter)
            best_primary_blocks = block_parameter.detach().clone()
            best_primary_tfv = float(initial_tfv)

        primary_iterations = max(1, int(round(iterations * 0.7)))
        for _ in range(primary_iterations):
            optimiser.zero_grad(set_to_none=True)
            _, _, _, tfv = rollout_for(block_parameter)
            if torch.isfinite(tfv) and float(tfv.detach()) < best_primary_tfv:
                best_primary_tfv = float(tfv.detach())
                best_primary_blocks = block_parameter.detach().clone()
            loss = tfv + self.movement_tiebreak * (
                block_parameter[:, 0] - current
            ).square().mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimiser.step()
            _project_block_settings_(
                block_parameter,
                current_settings=current,
                max_delta_per_update=max_setting_delta_per_update,
            )

        with torch.no_grad():
            _, _, _, tail_primary_tfv = rollout_for(block_parameter)
            if torch.isfinite(tail_primary_tfv) and float(tail_primary_tfv) < best_primary_tfv:
                best_primary_tfv = float(tail_primary_tfv)
                best_primary_blocks = block_parameter.detach().clone()
            block_parameter.copy_(best_primary_blocks)
            _, primary_initial, primary_rollout, primary_tfv = rollout_for(
                block_parameter
            )
            allowed_tfv = (
                primary_tfv
                + self.tfv_near_opt_absolute_m3
                + self.tfv_near_opt_relative * primary_tfv.abs()
            )
            primary_priority, _, _ = self._priority_deterioration(
                primary_initial, primary_rollout, fallback_volumes, fallback_depth
            )
            best_secondary_blocks = block_parameter.detach().clone()
            best_secondary_priority = float(primary_priority)
            best_secondary_tfv = float(primary_tfv)

        secondary_iterations = max(0, iterations - primary_iterations)
        if self.priority_indices is None or self.priority_indices.numel() == 0:
            secondary_iterations = 0

        for _ in range(secondary_iterations):
            optimiser.zero_grad(set_to_none=True)
            _, candidate_initial, rollout, tfv = rollout_for(block_parameter)
            priority_soft, _, _ = self._priority_deterioration(
                candidate_initial, rollout, fallback_volumes, fallback_depth
            )
            if (
                torch.isfinite(tfv)
                and torch.isfinite(priority_soft)
                and float(tfv.detach()) <= float(allowed_tfv) + 1e-6
            ):
                priority_value = float(priority_soft.detach())
                tfv_value = float(tfv.detach())
                if (
                    priority_value < best_secondary_priority - 1e-9
                    or (
                        abs(priority_value - best_secondary_priority) <= 1e-9
                        and tfv_value < best_secondary_tfv
                    )
                ):
                    best_secondary_priority = priority_value
                    best_secondary_tfv = tfv_value
                    best_secondary_blocks = block_parameter.detach().clone()
            loss = (
                priority_soft
                + self.near_opt_penalty * torch.relu(tfv - allowed_tfv).square()
                + self.movement_tiebreak
                * (block_parameter[:, 0] - current).square().mean()
            )
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimiser.step()
            _project_block_settings_(
                block_parameter,
                current_settings=current,
                max_delta_per_update=max_setting_delta_per_update,
            )

        with torch.no_grad():
            if secondary_iterations:
                _, tail_initial, tail_rollout, tail_tfv = rollout_for(block_parameter)
                tail_priority, _, _ = self._priority_deterioration(
                    tail_initial, tail_rollout, fallback_volumes, fallback_depth
                )
                if (
                    torch.isfinite(tail_tfv)
                    and torch.isfinite(tail_priority)
                    and float(tail_tfv) <= float(allowed_tfv) + 1e-6
                ):
                    priority_value = float(tail_priority)
                    tfv_value = float(tail_tfv)
                    if (
                        priority_value < best_secondary_priority - 1e-9
                        or (
                            abs(priority_value - best_secondary_priority) <= 1e-9
                            and tfv_value < best_secondary_tfv
                        )
                    ):
                        best_secondary_blocks = block_parameter.detach().clone()
            block_parameter.copy_(best_secondary_blocks)
            settings, candidate_initial, rollout, tfv = rollout_for(block_parameter)
            soft, worst_flood, worst_depth = self._priority_deterioration(
                candidate_initial, rollout, fallback_volumes, fallback_depth
            )

            required_improvement = max(
                self.min_predicted_tfv_improvement_m3,
                self.min_predicted_tfv_improvement_relative * abs(float(fallback_tfv)),
            )
            predicted_improvement = float(fallback_tfv - tfv)
            numeric_margin = max(1e-6, 1e-9 * max(abs(float(fallback_tfv)), 1.0))
            candidate_valid = bool(
                torch.isfinite(settings).all().item()
                and torch.isfinite(tfv).item()
                and torch.isfinite(soft).item()
                and predicted_improvement >= required_improvement + numeric_margin
            )
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
