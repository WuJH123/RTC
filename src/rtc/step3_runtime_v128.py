"""Execution-only acceleration for V128 continuous MPC.

All graph topology, actuator physics and node static features are immutable during an
authoritative event.  The scientific V128 scorer is unchanged; this adapter caches their
CUDA tensors instead of recreating/transferring them on every L-BFGS-B objective call.
"""
from __future__ import annotations

import torch

from .step3_mpc_v127 import _cvar
from .step3_mpc_v128 import DifferentiableRollingMPCV128

V128_RUNTIME_ACCELERATION_CONTRACT = (
    "PROJECT7_V128_MPC_RUNTIME_STATIC_TENSOR_CACHE_V1_EQUIVALENT"
)


class CachedDifferentiableRollingMPCV128(DifferentiableRollingMPCV128):
    """Mathematically identical V128 scorer with immutable device-tensor caching."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._device_static_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def _static_for(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        key = (str(state.device), str(state.dtype))
        cached = self._device_static_cache.get(key)
        if cached is None:
            cached = {
                "up": torch.as_tensor(
                    self.graph.actuator_upstream, dtype=torch.long, device=state.device
                ),
                "down": torch.as_tensor(
                    self.graph.actuator_downstream, dtype=torch.long, device=state.device
                ),
                "physics": torch.as_tensor(
                    self.graph.actuator_physics, dtype=state.dtype, device=state.device
                ),
                "static": torch.as_tensor(
                    self.graph.static_node_features, dtype=state.dtype, device=state.device
                ),
                "edges": torch.as_tensor(
                    self.graph.edge_index, dtype=torch.long, device=state.device
                ),
            }
            self._device_static_cache[key] = cached
        return cached

    def _score(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        sequence: torch.Tensor,
        flow: torch.Tensor,
        active_target: torch.Tensor,
        rbc_smooth_pfv: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scenarios = int(rainfall.shape[0])
        state = initial_state.expand(scenarios, -1, -1)
        static = self._static_for(state)
        output = self.model.objective_rollout(
            initial_state=state,
            rainfall=rainfall,
            settings=sequence[None].expand(scenarios, -1, -1),
            previous_actuator_flow=flow.expand(scenarios, -1),
            actuator_upstream=static["up"],
            actuator_downstream=static["down"],
            actuator_physics=static["physics"],
            static_node_features=static["static"],
            edge_index=static["edges"],
            flood_rate_index=self.flood_rate_index,
            priority_indices=self.priority_indices,
            dt_seconds=float(self.design.model_step_seconds),
        )
        hard_tfv = _cvar(output.tfv_m3, self.design.cvar_alpha)
        hard_pfv = _cvar(output.pfv_m3, self.design.cvar_alpha)
        smooth_tfv = _cvar(output.optimization_tfv_m3, self.design.cvar_alpha)
        smooth_pfv = _cvar(output.optimization_pfv_m3, self.design.cvar_alpha)
        pfv_penalty = smooth_tfv.new_zeros(())
        if rbc_smooth_pfv is not None and self.design.pfv_penalty_weight > 0.0:
            pfv_penalty = self.design.pfv_penalty_weight * torch.relu(
                smooth_pfv - rbc_smooth_pfv - self.design.pfv_soft_margin_m3
            )
        first = sequence[: self.design.control_block_steps].mean(dim=0)
        movement = self.design.movement_penalty_m3 * torch.mean(
            torch.square(first - active_target)
        )
        return (
            smooth_tfv + pfv_penalty + movement,
            hard_tfv,
            hard_pfv,
            smooth_tfv,
            smooth_pfv,
        )


__all__ = [
    "CachedDifferentiableRollingMPCV128",
    "V128_RUNTIME_ACCELERATION_CONTRACT",
]
