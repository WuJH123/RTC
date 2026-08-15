"""V128 controller adapter for exact scored commands under a frozen engineering envelope."""
from __future__ import annotations

from typing import Any

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .controller_v127 import V127TorchMPCController
from .engineering_v128 import V128EngineeringEnvelope
from .step2_differentiable_v128 import V128_STEP2_CONTRACT
from .step3_mpc_v128 import V128_STEP3_CONTRACT

V128_CONTROLLER_CONTRACT = (
    "PROJECT7_V128_TYPED_STEP2_PER_ACTUATOR_ENVELOPE_TARGET_LATCH_CONTROLLER_V2"
)


class V128TorchMPCController(V127TorchMPCController):
    def __init__(
        self,
        *args: Any,
        engineering_envelope: V128EngineeringEnvelope,
        **kwargs: Any,
    ) -> None:
        self.engineering_envelope = engineering_envelope
        super().__init__(*args, **kwargs)
        self.engineering_envelope.assert_graph_order(self.graph)

    def _validate_scored_first_move(
        self,
        requested: np.ndarray,
        *,
        active_target: np.ndarray,
    ) -> tuple[bool, str]:
        requested = np.asarray(requested, dtype=float).reshape(-1)
        active = np.asarray(active_target, dtype=float).reshape(-1)
        if requested.shape != active.shape or requested.size != len(
            self.engineering_envelope.actuator_ids
        ):
            return False, "shape_mismatch"
        if not np.isfinite(requested).all() or not np.isfinite(active).all():
            return False, "nonfinite"
        lo = np.asarray(self.engineering_envelope.min_setting, dtype=float)
        hi = np.asarray(self.engineering_envelope.max_setting, dtype=float)
        delta = np.asarray(
            self.engineering_envelope.max_delta_per_10min, dtype=float
        )
        if np.any(requested < lo - 1e-8) or np.any(requested > hi + 1e-8):
            return False, "engineering_bounds"
        if np.any(np.abs(requested - active) > delta + 1e-7):
            return False, "engineering_target_rate"
        return True, "pass"

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(
            obs, observation_already_recorded=observation_already_recorded
        )
        requested = np.asarray(
            [float(action.settings[aid]) for aid in self.graph.actuator_ids], dtype=float
        )
        active_target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
        executable, reason = self._validate_scored_first_move(
            requested, active_target=active_target
        )
        if not executable:
            raise RuntimeError(
                f"V128 command violates frozen engineering envelope: {reason}"
            )

        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "v128_controller_contract": V128_CONTROLLER_CONTRACT,
                "v128_step2_contract": V128_STEP2_CONTRACT,
                "v128_step3_contract": V128_STEP3_CONTRACT,
                "engineering_envelope_sha256": self.engineering_envelope.semantic_sha256,
                "engineering_envelope_source": self.engineering_envelope.source,
                "engineering_envelope_is_idealized_default": self.engineering_envelope.is_idealized_default,
                "score_equals_execute_under_engineering_envelope": True,
                "engineering_envelope_validation_reason": reason,
            }
        )
        result = self.mpc.last_result
        source = str(action.source)
        if result is not None:
            selected = str(getattr(result, "selected_source", ""))
            if selected == "continuous_lbfgsb_v128" and not source.startswith("FALLBACK"):
                source = "MPC_V128_CONTINUOUS"
            elif selected == "rbc_safety_fallback_v128" and not source.startswith("FALLBACK"):
                source = "RBC_SAFETY_V128"
            diagnostics["v128_selected_source"] = selected
        if source.endswith("_V127"):
            source = source[:-5] + "_V128"
        elif source == "MPC_V127":
            source = "MPC_V128"
        return ControllerAction(
            settings=action.settings,
            source=source,
            diagnostics=diagnostics,
        )


__all__ = ["V128_CONTROLLER_CONTRACT", "V128TorchMPCController"]
