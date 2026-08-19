"""Development V13: policy-return-aligned first-action selection.

The V12 scenario-mean/open-loop model remains the coordinated direction generator.  A second value
model, initialized from the frozen Direct-TFV Step2 checkpoint and fine-tuned only on paired
same-prefix SWMM *policy-return* labels, judges the action that is actually committed for H10.

This is deliberately a policy-improvement layer rather than a replacement for the H120 search:
V12 proposes and refines a q95-supported target; the policy-return critic decides whether that target
has negative TFV advantage when followed by the frozen continuation policy used to create its labels.
The old open-loop first-move conformal margin does not control execution here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    policy_return_margin_m3,
)
from .step3_tfv_value_mpc_v9 import DirectTFVMPCResultV9
from .step3_tfv_value_mpc_v10 import DirectTFVScenarioMeanMPCV10


DIRECT_TFV_POLICY_RETURN_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V11_POLICY_RETURN_FIRST_ACTION"
)


@dataclass(frozen=True)
class DirectTFVMPCResultV11(DirectTFVMPCResultV9):
    policy_return_predicted_delta_tfv_m3: float = 0.0
    policy_return_margin_m3: float = 0.0
    policy_return_upper_bound_m3: float = 0.0
    policy_return_admission_passed: bool = False
    policy_return_admission_contract: str = DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT
    policy_return_estimand: str = DIRECT_TFV_POLICY_RETURN_ESTIMAND
    policy_return_parent_continuation_sha256: str = ""


class DirectTFVPolicyReturnMPCV11(DirectTFVScenarioMeanMPCV10):
    """Scenario-mean V12 direction generation plus a deployed-policy-return first-action critic."""

    policy_mode = "direct_tfv_all109_receding_mpc_v11_policy_return"
    policy_mode_contract = DIRECT_TFV_POLICY_RETURN_STEP3_CONTRACT

    def __init__(
        self,
        *,
        policy_return_model: Any,
        policy_return_normalization: Any,
        policy_return_admission: Mapping[str, Any],
        policy_return_checkpoint_sha256: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        admission = dict(policy_return_admission)
        if str(admission.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
            raise ValueError("V13 requires receding-policy-return admission")
        if admission.get("development_only") is not True:
            raise ValueError("V13 policy-return admission must be Development-only")
        if str(admission.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
            raise ValueError("V13 policy-return admission has the wrong estimand")
        if str(admission.get("policy_return_checkpoint_sha256", "")).lower() != str(
            policy_return_checkpoint_sha256
        ).lower():
            raise ValueError("V13 policy-return admission was calibrated on another critic")
        parent = str(admission.get("continuation_policy_sha256", "")).lower()
        if len(parent) != 64:
            raise ValueError("V13 policy-return admission lacks a frozen continuation policy SHA")
        if admission.get("generic_d3_floor_controls_execution") is not False:
            raise ValueError("generic D3 margin cannot control V13 execution")
        if admission.get("open_loop_first_move_margin_controls_execution") is not False:
            raise ValueError("V12 open-loop first-move margin cannot control V13 execution")
        self.policy_return_model = policy_return_model
        self.policy_return_model.eval()
        self.policy_return_normalization = policy_return_normalization
        self.policy_return_admission = admission
        self.policy_return_checkpoint_sha256 = str(policy_return_checkpoint_sha256).lower()
        self.policy_return_parent_continuation_sha256 = parent

    def _policy_return_normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.policy_return_normalization.state_mean, dtype=value.dtype, device=value.device
        )
        std = torch.as_tensor(
            self.policy_return_normalization.state_std, dtype=value.dtype, device=value.device
        ).clamp_min(1.0e-6)
        return (value - mean) / std

    def _policy_return_normalize_rainfall(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.policy_return_normalization.rainfall_mean, dtype=value.dtype, device=value.device
        )
        std = torch.as_tensor(
            self.policy_return_normalization.rainfall_std, dtype=value.dtype, device=value.device
        ).clamp_min(1.0e-6)
        return (value - mean) / std

    def _policy_return_normalize_flow(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.policy_return_normalization.flow_mean, dtype=value.dtype, device=value.device
        )
        std = torch.as_tensor(
            self.policy_return_normalization.flow_std, dtype=value.dtype, device=value.device
        ).clamp_min(1.0e-6)
        return (value - mean) / std

    def _score_policy_return_target(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
        candidate_target: torch.Tensor,
    ) -> torch.Tensor:
        """Score one executable first target using the causal rainfall scenario ensemble."""

        if rainfall.ndim != 4 or int(rainfall.shape[0]) < 2:
            raise ValueError("policy-return critic requires causal rainfall scenarios [S,H,N,R]")
        if tuple(active_target.shape) != (109,) or tuple(candidate_target.shape) != (109,):
            raise ValueError("policy-return critic requires 109-dimensional active/candidate targets")
        scenarios = int(rainfall.shape[0])
        horizon = int(self.design.prediction_horizon_steps)
        state = self._policy_return_normalize_state(current_state).expand(scenarios, -1, -1)
        rain = self._policy_return_normalize_rainfall(rainfall)
        flow = self._policy_return_normalize_flow(previous_actuator_flow).expand(scenarios, -1)
        reference = active_target.reshape(1, 1, 109).expand(scenarios, horizon, 109)
        candidate = candidate_target.reshape(1, 1, 109).expand(scenarios, horizon, 109)
        output = self.policy_return_model(
            current_state=state,
            rainfall=rain,
            reference_settings=reference,
            candidate_settings=candidate,
            previous_actuator_flow=flow,
            actuator_upstream=torch.as_tensor(
                self.graph.actuator_upstream, dtype=torch.long, device=state.device
            ),
            actuator_downstream=torch.as_tensor(
                self.graph.actuator_downstream, dtype=torch.long, device=state.device
            ),
            actuator_physics=torch.as_tensor(
                self.graph.actuator_physics, dtype=state.dtype, device=state.device
            ),
        )
        scores = output.total_delta_tfv_m3
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("policy-return critic produced non-finite values")
        return scores.mean()

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV11:
        # V12 remains the support-aware coordinated proposal/refinement mechanism.  Its own
        # first-move admission is diagnostic only after this point.
        base = super().optimize(**kwargs)
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor):
            raise ValueError("V13 requires active_target")
        candidate = getattr(base, "optimized_candidate_settings", None)
        values = dict(vars(base))
        defaults = {
            "policy_mode": self.policy_mode,
            "policy_mode_contract": self.policy_mode_contract,
            "policy_return_parent_continuation_sha256": self.policy_return_parent_continuation_sha256,
        }
        if candidate is None:
            values.update(defaults)
            return DirectTFVMPCResultV11(**values)
        target = candidate[0].detach()
        changed = int(torch.count_nonzero(torch.abs(target - active_target) > 1.0e-7).item())
        if changed == 0:
            values.update(defaults)
            values.update(
                {
                    "settings": self._hold_sequence(active_target),
                    "predicted_delta_tfv_m3": 0.0,
                    "selected_source": "LATCH_PREVIOUS_TARGET_NO_POLICY_RETURN_CHANGE",
                    "admission_margin_m3": 0.0,
                    "admission_upper_bound_m3": 0.0,
                    "admission_passed": False,
                    "calibrated_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                }
            )
            return DirectTFVMPCResultV11(**values)

        score = self._score_policy_return_target(
            current_state=kwargs["current_state"],
            rainfall=kwargs["rainfall"],
            previous_actuator_flow=kwargs["previous_actuator_flow"],
            active_target=active_target,
            candidate_target=target,
        )
        score_m3 = float(score.detach().cpu())
        margin = policy_return_margin_m3(self.policy_return_admission, changed)
        upper = score_m3 + margin
        passed = bool(upper < 0.0)
        executed = candidate.detach() if passed else self._hold_sequence(active_target).detach()
        values.update(defaults)
        values.update(
            {
                "settings": executed,
                "predicted_delta_tfv_m3": score_m3 if passed else 0.0,
                "raw_optimized_predicted_delta_tfv_m3": score_m3,
                "selected_source": (
                    "DIRECT_TFV_RECEDING_LBFGSB" if passed
                    else "LATCH_PREVIOUS_TARGET_POLICY_RETURN_UPPER_BOUND_NONNEGATIVE"
                ),
                "admission_margin_m3": float(margin),
                "admission_upper_bound_m3": float(upper),
                "admission_margin_kind": "receding_policy_return_normalized",
                "admission_passed": passed,
                "calibrated_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "refined_first_move_predicted_delta_tfv_m3": score_m3,
                "refined_first_move_margin_m3": float(margin),
                "refined_first_move_upper_bound_m3": float(upper),
                "refined_first_move_admission_passed": passed,
                "refined_first_move_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "policy_return_predicted_delta_tfv_m3": score_m3,
                "policy_return_margin_m3": float(margin),
                "policy_return_upper_bound_m3": float(upper),
                "policy_return_admission_passed": passed,
                "policy_return_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "policy_return_estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
            }
        )
        return DirectTFVMPCResultV11(**values)


__all__ = [
    "DIRECT_TFV_POLICY_RETURN_STEP3_CONTRACT",
    "DirectTFVMPCResultV11",
    "DirectTFVPolicyReturnMPCV11",
]
