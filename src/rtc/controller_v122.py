"""V12.2 controller: causal sparse state -> Value MPC -> exact scored first target.

Unlike the legacy controller, V12.2 never projects an MPC action after scoring.  The
Step3 optimiser owns the executable target-command envelope; runtime only verifies that
its selected first move satisfies the frozen contract.  ``current_setting`` remains a
physical-state observation, while ``target_setting`` is the supervisory command latch.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .closed_loop import CausalObservation, ControllerAction
from .controller import TorchMPCController
from .step3_mpc_v122 import V122_STEP3_CONTRACT

V122_CONTROLLER_CONTRACT = "PROJECT7_V122_TARGET_LATCH_SCORE_EQUALS_EXECUTE_CONTROLLER_V1"
V122_READBACK_CONTRACT = "PROJECT7_V122_TARGET_WRITE_EXACT_CURRENT_TRACKING_DIAGNOSTIC_V1"


def hold_active_target_v122(
    observation: CausalObservation, horizon_steps: int
) -> np.ndarray:
    target = np.asarray(observation.actuator_target_setting, dtype=float).reshape(-1)
    if not target.size or not np.isfinite(target).all() or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("V122 active target readback is invalid")
    return np.repeat(target[None, :], int(horizon_steps), axis=0)


class V122TorchMPCController(TorchMPCController):
    """Project7 V12.2 receding-horizon controller with target-latch semantics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Fallback/no-new-command means preserving the accepted target latch.  This is
        # intentionally different from the legacy hold-current fallback.
        self.fallback_sequence_provider = hold_active_target_v122

    def _passive_action(
        self,
        *,
        obs: CausalObservation,
        source: str,
        diagnostics: dict[str, float | int | bool | str],
    ) -> ControllerAction:
        active_target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        if active_target.shape != current.shape or active_target.size != len(self.graph.actuator_ids):
            raise ValueError("V122 passive action readback shape mismatch")
        if not np.isfinite(active_target).all() or np.any((active_target < 0.0) | (active_target > 1.0)):
            raise ValueError("V122 cannot preserve an invalid target latch")
        self.last_requested = active_target.copy()
        payload = dict(diagnostics)
        payload.update(
            {
                "v122_controller_contract": V122_CONTROLLER_CONTRACT,
                "v122_step3_contract": V122_STEP3_CONTRACT,
                "v122_readback_contract": V122_READBACK_CONTRACT,
                "passive_no_new_command": True,
                "target_change_max": 0.0,
                "current_tracking_lag_max": float(
                    np.abs(current - active_target).max(initial=0.0)
                ),
                "score_equals_execute": True,
            }
        )
        return ControllerAction(
            settings=dict(zip(self.graph.actuator_ids, active_target, strict=True)),
            source=source,
            diagnostics=payload,
        )

    def _validate_scored_first_move(
        self,
        requested: np.ndarray,
        *,
        active_target: np.ndarray,
    ) -> tuple[bool, str]:
        requested = np.asarray(requested, dtype=float).reshape(-1)
        active_target = np.asarray(active_target, dtype=float).reshape(-1)
        if requested.shape != active_target.shape:
            return False, "shape_mismatch"
        if not np.isfinite(requested).all():
            return False, "nonfinite"
        if np.any((requested < -1e-9) | (requested > 1.0 + 1e-9)):
            return False, "bounds"
        bound = self.config.max_setting_delta_per_update
        if bound is not None and float(np.abs(requested - active_target).max(initial=0.0)) > float(bound) + 1e-7:
            return False, "target_rate"
        return True, "pass"

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        if not observation_already_recorded:
            self.observe(obs)
        else:
            self._validate_observation(obs)
            if self.last_observed_elapsed_seconds != obs.elapsed_seconds:
                raise ValueError("V122 decision observation was not recorded at this model step")

        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        active_target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
        if current.shape != active_target.shape or current.size != len(self.graph.actuator_ids):
            raise ValueError("V122 actuator readback shape mismatch")

        # A previous Python write is accepted iff the target latch matches it.  Realised
        # current-setting lag is reported, never used as evidence that the write failed.
        if self.last_requested is not None:
            target_error = np.abs(active_target - self.last_requested)
            current_lag = np.abs(current - self.last_requested)
            if float(target_error.max(initial=0.0)) > self.config.readback_target_tolerance:
                return self._passive_action(
                    obs=obs,
                    source="FALLBACK_TARGET_WRITE_READBACK",
                    diagnostics={
                        "previous_command_target_error_max": float(
                            target_error.max(initial=0.0)
                        ),
                        "previous_command_current_tracking_lag_max": float(
                            current_lag.max(initial=0.0)
                        ),
                        "target_readback_failed_actuators": int(
                            np.sum(target_error > self.config.readback_target_tolerance)
                        ),
                    },
                )

        if len(self.observed_history) < self.config.history_steps:
            return self._passive_action(
                obs=obs,
                source="FALLBACK_HISTORY_WARMUP",
                diagnostics={
                    "history_frames": len(self.observed_history),
                    "required_history_frames": self.config.history_steps,
                },
            )

        started = time.perf_counter()
        try:
            static = torch.as_tensor(
                self.graph.static_node_features,
                dtype=torch.float32,
                device=self.device,
            )
            edges = torch.as_tensor(
                self.graph.edge_index, dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                initial_state = self.step1(
                    torch.as_tensor(
                        np.stack(self.observed_history)[None],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        np.stack(self.mask_history)[None],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    static,
                    edges,
                    torch.as_tensor(
                        np.stack(self.context_history)[None],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            rainfall_scenarios = self.forecast.forecast(
                np.stack(self.rainfall_history),
                horizon_steps=self.config.horizon_steps,
            )
            result = self.mpc.optimize(
                initial_state=initial_state,
                rainfall_scenarios=torch.as_tensor(
                    rainfall_scenarios, dtype=torch.float32, device=self.device
                ),
                current_settings=torch.as_tensor(
                    current, dtype=torch.float32, device=self.device
                ),
                previous_requested_settings=torch.as_tensor(
                    active_target, dtype=torch.float32, device=self.device
                ),
                fallback_settings=torch.as_tensor(
                    hold_active_target_v122(obs, self.config.horizon_steps)[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                previous_actuator_flow=torch.as_tensor(
                    obs.actuator_flow_m3s[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                max_delta_per_update=self.config.max_setting_delta_per_update,
            )
            runtime_seconds = float(time.perf_counter() - started)
            budget = self.config.decision_runtime_budget_seconds
            if budget is not None and runtime_seconds > float(budget):
                return self._passive_action(
                    obs=obs,
                    source="FALLBACK_COMPUTE_DEADLINE",
                    diagnostics={
                        "decision_runtime_seconds": runtime_seconds,
                        "decision_runtime_budget_seconds": float(budget),
                    },
                )

            candidate_valid = bool(getattr(result, "candidate_valid", False))
            if not candidate_valid:
                return self._passive_action(
                    obs=obs,
                    source="PASSIVE_MPC_NO_PREDICTED_BENEFIT",
                    diagnostics={
                        "decision_runtime_seconds": runtime_seconds,
                        "predicted_delta_tfv_m3": float(
                            getattr(result, "predicted_delta_tfv_m3", 0.0)
                        ),
                    },
                )

            settings = result.settings.detach().cpu().numpy()
            if settings.ndim != 2 or settings.shape != (
                self.config.horizon_steps,
                len(self.graph.actuator_ids),
            ):
                return self._passive_action(
                    obs=obs,
                    source="FALLBACK_MPC_OUTPUT_CONTRACT",
                    diagnostics={"decision_runtime_seconds": runtime_seconds},
                )
            block = int(self.config.control_block_steps)
            if block <= 0 or not np.allclose(
                settings[:block], settings[0][None, :], rtol=0.0, atol=1e-7
            ):
                return self._passive_action(
                    obs=obs,
                    source="FALLBACK_FIRST_BLOCK_NOT_CONSTANT",
                    diagnostics={"decision_runtime_seconds": runtime_seconds},
                )
            requested = settings[0].copy()
            executable, reason = self._validate_scored_first_move(
                requested, active_target=active_target
            )
            if not executable:
                return self._passive_action(
                    obs=obs,
                    source="FALLBACK_SCORE_EXECUTE_CONTRACT",
                    diagnostics={
                        "decision_runtime_seconds": runtime_seconds,
                        "score_execute_failure": reason,
                    },
                )

            self.last_requested = requested.copy()
            target_change = np.abs(requested - active_target)
            diagnostics: dict[str, float | int | bool | str] = {
                "v122_controller_contract": V122_CONTROLLER_CONTRACT,
                "v122_step3_contract": V122_STEP3_CONTRACT,
                "v122_readback_contract": V122_READBACK_CONTRACT,
                "passive_no_new_command": False,
                "score_equals_execute": True,
                "decision_runtime_seconds": runtime_seconds,
                "decision_runtime_budget_seconds": (
                    -1.0 if budget is None else float(budget)
                ),
                "predicted_delta_tfv_m3": float(
                    getattr(result, "predicted_delta_tfv_m3", float("nan"))
                ),
                "selected_group_score_m3": float(
                    getattr(result, "selected_group_score_m3", float("nan"))
                ),
                "selected_candidate_index": int(
                    getattr(result, "selected_candidate_index", -1)
                ),
                "candidate_count": int(getattr(result, "candidate_count", -1)),
                "first_move_group_count": int(
                    getattr(result, "first_move_group_count", -1)
                ),
                "tail_only_noop_candidate_count": int(
                    getattr(result, "tail_only_noop_candidate_count", -1)
                ),
                "target_change_max": float(target_change.max(initial=0.0)),
                "target_change_l1": float(target_change.sum()),
                "current_tracking_lag_max": float(
                    np.abs(current - active_target).max(initial=0.0)
                ),
            }
            return ControllerAction(
                settings=dict(zip(self.graph.actuator_ids, requested, strict=True)),
                source="MPC_V122",
                diagnostics=diagnostics,
            )
        except Exception as exc:
            runtime_seconds = float(time.perf_counter() - started)
            return self._passive_action(
                obs=obs,
                source="FALLBACK_RUNTIME_ERROR_V122",
                diagnostics={
                    "decision_runtime_seconds": runtime_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )


__all__ = [
    "V122_CONTROLLER_CONTRACT",
    "V122_READBACK_CONTRACT",
    "V122TorchMPCController",
    "hold_active_target_v122",
]
