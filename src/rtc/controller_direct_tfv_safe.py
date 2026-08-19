"""Memory-safe authoritative Direct-TFV controller adapter.

The scored action is unchanged. After the normal controller has converted the selected tensor to the
exact Python target command, detach the retained telemetry copy so successive 10-minute decisions do
not keep prior autograd graphs alive on GPU.
"""
from __future__ import annotations

from dataclasses import replace

from .closed_loop import CausalObservation, ControllerAction
from .controller_direct_tfv import DirectTFVAuthoritativeController


DIRECT_TFV_RUNTIME_GRAPH_RELEASE_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RUNTIME_TELEMETRY_AUTOGRAD_RELEASE_V1"
)


class MemorySafeDirectTFVAuthoritativeController(DirectTFVAuthoritativeController):
    """Numerically identical controller that releases retained optimizer graphs after each decision."""

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        result = self._direct_mpc_adapter.last_result
        if result is not None:
            detached = result.settings.detach()
            self._direct_mpc_adapter.last_result = replace(result, settings=detached)
        diagnostics = dict(action.diagnostics or {})
        diagnostics["runtime_graph_release_contract"] = DIRECT_TFV_RUNTIME_GRAPH_RELEASE_CONTRACT
        diagnostics["runtime_retained_settings_requires_grad"] = bool(
            self._direct_mpc_adapter.last_result is not None
            and self._direct_mpc_adapter.last_result.settings.requires_grad
        )
        return ControllerAction(
            settings=action.settings,
            source=action.source,
            diagnostics=diagnostics,
        )


__all__ = [
    "DIRECT_TFV_RUNTIME_GRAPH_RELEASE_CONTRACT",
    "MemorySafeDirectTFVAuthoritativeController",
]
