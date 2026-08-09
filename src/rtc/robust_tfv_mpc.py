from __future__ import annotations

import math

import torch
from torch import nn

from .tfv_mpc import (
    ContinuousTFVFirstMPC as _BaseTFVMPC,
    TFVFirstMPCResult,
    _expand_blocks,
    _project_block_settings_,
    _upper_tail_cvar,
)


class ContinuousTFVFirstMPC(_BaseTFVMPC):
    """Robust public MPC used by the production runner.

    The original optimizer already has a non-empty executable set because holding the
    current settings is always available. The important engineering failure mode is instead
    accepting a numerically finite but *worse* final Adam iterate. This implementation keeps
    the best primary iterate seen so far, keeps the best priority-secondary iterate inside the
    TFV near-optimal set, and only admits a candidate that is predicted to beat the explicit
    fallback sequence by a small configurable margin.
    """

    def __init__(
        self,
        *args,
        min_predicted_tfv_improvement_m3: float = 0.0,
        min_predicted_tfv_improvement_relative: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if min_predicted_tfv_improvement_m3 < 0:
            raise ValueError("minimum predicted TFV improvement must be non-negative")
        if min_predicted_tfv_improvement_relative < 0:
            raise ValueError("relative predicted TFV improvement must be non-negative")
        self.min_predicted_tfv_improvement_m3 = float(
            min_predicted_tfv_improvement_m3
        )
        self.min_predicted_tfv_improvement_relative = float(
            min_predicted_tfv_improvement_relative
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
