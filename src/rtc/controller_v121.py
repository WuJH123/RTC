from __future__ import annotations

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .controller_v120 import V120TorchMPCController
from .step3_policy_v121 import FirstMoveRobustCandidatePolicyV121, V121_STEP3_CONTRACT

V121_READBACK_CONTRACT = "PROJECT7_V121_TARGET_LATCH_CURRENT_STATE_READBACK_V1"


class V121TorchMPCController(V120TorchMPCController):
    """Keep target write verification while reporting realized-setting tracking separately."""

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        previous_command = (
            None
            if self.last_requested is None
            else np.asarray(self.last_requested, dtype=float).copy()
        )
        action = super().decide(
            obs, observation_already_recorded=observation_already_recorded
        )
        diagnostics = dict(action.diagnostics or {})
        diagnostics["v121_readback_contract"] = V121_READBACK_CONTRACT

        if previous_command is not None:
            target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
            current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
            diagnostics["previous_command_target_latch_error_max"] = float(
                np.abs(target - previous_command).max(initial=0.0)
            )
            diagnostics["previous_command_current_tracking_lag_max"] = float(
                np.abs(current - previous_command).max(initial=0.0)
            )
            diagnostics["current_setting_role"] = "physical_state_and_continuity_anchor"

        wrapped = getattr(self.mpc, "policy", None)
        if isinstance(wrapped, FirstMoveRobustCandidatePolicyV121):
            result = wrapped.last_result
            if result is not None:
                diagnostics.update(
                    {
                        "v121_step3_contract": V121_STEP3_CONTRACT,
                        "selected_candidate_index": int(result.selected_candidate_index),
                        "selected_candidate_family": str(result.selected_candidate_family),
                        "selected_first_move_group": int(result.selected_first_move_group),
                        "first_move_group_count": int(result.first_move_group_count),
                        "selected_group_size": int(result.selected_group_size),
                        "hold_group_size": int(result.hold_group_size),
                        "selected_sequence_delta_tfv_m3": float(result.selected_sequence_delta_tfv_m3),
                        "robust_group_delta_tfv_m3": float(result.robust_group_delta_tfv_m3),
                        "best_vs_hold_improvement_m3": float(result.best_vs_hold_improvement_m3),
                        "first_move_mean_abs_delta": float(result.first_move_mean_abs_delta),
                        "selected_group_tail_spread_m3": float(result.selected_group_tail_spread_m3),
                        "tail_only_noop_candidates": int(result.tail_only_noop_candidates),
                    }
                )

        return ControllerAction(
            settings=action.settings,
            source=action.source,
            diagnostics=diagnostics,
        )


__all__ = ["V121_READBACK_CONTRACT", "V121TorchMPCController"]
