"""V123 runtime label adapter over the frozen V122 target-latch controller."""
from __future__ import annotations

from .closed_loop import ControllerAction
from .controller_v122 import V122TorchMPCController


V123_CONTROLLER_CONTRACT = "PROJECT7_V123_TFV_PRIMARY_PFV_SOFT_FIRST_MOVE_CONTROLLER_V1"


class V123TorchMPCController(V122TorchMPCController):
    """Keep V122 continuity/readback logic while stamping V123 evidence labels."""

    def decide(self, *args, **kwargs):
        action = super().decide(*args, **kwargs)
        source = str(action.source)
        if source.endswith("_V122") or source == "MPC_V122":
            source = source[:-5] + "_V123" if source.endswith("_V122") else "MPC_V123"
        diagnostics = dict(action.diagnostics or {})
        diagnostics.pop("v122_controller_contract", None)
        diagnostics.pop("v122_step3_contract", None)
        diagnostics.pop("v122_readback_contract", None)
        diagnostics["v123_controller_contract"] = V123_CONTROLLER_CONTRACT
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = ["V123_CONTROLLER_CONTRACT", "V123TorchMPCController"]
