"""Legacy V13 bridge: V12 direction generation plus a receding-policy-return first-action critic.

The current Practical portfolio no longer relies on this class's ``optimize`` path, but it reuses the
validated policy-return checkpoint/admission loading and target scorer. The scorer is therefore
strictly aligned to the deployed estimand: candidate occupies H10 only and the remaining H350 is
encoded as HOLD. This bridge is retained for parent-policy iteration/backward compatibility, not as
the preferred Practical online optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS
from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    encode_policy_return_action_token,
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
    """Compatibility path with an H10-action-token deployed-policy-return critic."""

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
            raise ValueError("policy-return runtime requires current receding-policy-return admission")
        if admission.get("development_only") is not True:
            raise ValueError("policy-return admission must be Development-only")
        if str(admission.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
            raise ValueError("policy-return admission has the wrong estimand")
        if str(admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
            raise ValueError("policy-return admission has the wrong H10 action encoding")
        if str(admission.get("policy_return_checkpoint_sha256", "")).lower() != str(
            policy_return_checkpoint_sha256
        ).lower():
            raise ValueError("policy-return admission was calibrated on another critic")
        parent = str(admission.get("continuation_policy_sha256", "")).lower()
        if len(parent) != 64:
            raise ValueError("policy-return admission lacks a frozen continuation policy SHA")
        if admission.get("generic_d3_floor_controls_execution") is not False:
            raise ValueError("generic D3 margin cannot control policy-return execution")
        if admission.get("open_loop_first_move_margin_controls_execution") is not False:
            raise ValueError("open-loop first-move margin cannot control policy-return execution")
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
        """Score one executable H10 intervention with only causal rainfall scenarios."""
        if rainfall.ndim != 4 or int(rainfall.shape[0]) < 2:
            raise ValueError("policy-return critic requires rainfall [scenario,H,node,feature]")
        if tuple(active_target.shape) != (109,) or tuple(candidate_target.shape) != (109,):
            raise ValueError("policy-return critic requires 109-dimensional targets")
        scenarios = int(rainfall.shape[0])
        horizon = int(self.design.prediction_horizon_steps)
        state = self._policy_return_normalize_state(current_state).expand(scenarios, -1, -1)
        rain = self._policy_return_normalize_rainfall(rainfall)
        flow = self._policy_return_normalize_flow(previous_actuator_flow).expand(scenarios, -1)
        active = active_target.reshape(1, 109).expand(scenarios, -1)
        target = candidate_target.reshape(1, 109).expand(scenarios, -1)
        reference, candidate = encode_policy_return_action_token(
            active,
            target,
            horizon_steps=horizon,
            first_action_steps=int(self.design.control_block_steps),
        )
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

    def _hold_result(
        self, base: DirectTFVMPCResultV9, active_target: torch.Tensor, source: str
    ) -> DirectTFVMPCResultV11:
        values = dict(vars(base))
        values.update(
            {
                "settings": self._hold_sequence(active_target).detach(),
                "predicted_delta_tfv_m3": 0.0,
                "selected_source": source,
                "admission_margin_m3": 0.0,
                "admission_upper_bound_m3": 0.0,
                "admission_passed": False,
                "calibrated_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "policy_return_parent_continuation_sha256": self.policy_return_parent_continuation_sha256,
            }
        )
        return DirectTFVMPCResultV11(**values)

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV11:
        """Legacy bridge only; Practical V14+ uses ``DirectTFVPolicyReturnPortfolioMPCV12``."""
        base = super().optimize(**kwargs)
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor):
            raise ValueError("policy-return bridge requires active_target")
        candidate = getattr(base, "optimized_candidate_settings", None)
        refined_count = int(getattr(base, "refined_first_move_changed_facility_count", 0))
        refined_semantics = str(getattr(base, "refined_first_move_semantics", ""))
        if candidate is None or refined_count <= 0:
            return self._hold_result(base, active_target, "LATCH_PREVIOUS_TARGET_NO_REFINED_POLICY_RETURN_QUERY")
        if refined_semantics != DIRECT_TFV_FIRST_MOVE_SEMANTICS:
            raise RuntimeError("policy-return bridge refuses a non-target-latch refined first move")

        target = candidate[0].detach()
        changed = int(torch.count_nonzero(torch.abs(target - active_target) > 1.0e-7).item())
        if changed != refined_count:
            raise RuntimeError("policy-return score/execute changed-facility count differs from refined first move")
        if changed == 0:
            return self._hold_result(base, active_target, "LATCH_PREVIOUS_TARGET_NO_POLICY_RETURN_CHANGE")

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
        values = dict(vars(base))
        values.update(
            {
                "settings": executed,
                "predicted_delta_tfv_m3": score_m3 if passed else 0.0,
                "raw_optimized_predicted_delta_tfv_m3": score_m3,
                "selected_source": (
                    "DIRECT_TFV_RECEDING_LBFGSB_LEGACY_BRIDGE"
                    if passed
                    else "LATCH_PREVIOUS_TARGET_POLICY_RETURN_UPPER_BOUND_NONNEGATIVE"
                ),
                "admission_margin_m3": float(margin),
                "admission_upper_bound_m3": float(upper),
                "admission_margin_kind": "receding_policy_return_h10_action_token",
                "admission_passed": passed,
                "calibrated_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "refined_first_move_predicted_delta_tfv_m3": score_m3,
                "refined_first_move_margin_m3": float(margin),
                "refined_first_move_upper_bound_m3": float(upper),
                "refined_first_move_admission_passed": passed,
                "refined_first_move_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "policy_return_predicted_delta_tfv_m3": score_m3,
                "policy_return_margin_m3": float(margin),
                "policy_return_upper_bound_m3": float(upper),
                "policy_return_admission_passed": passed,
                "policy_return_admission_contract": DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                "policy_return_estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
                "policy_return_parent_continuation_sha256": self.policy_return_parent_continuation_sha256,
            }
        )
        return DirectTFVMPCResultV11(**values)


__all__ = [
    "DIRECT_TFV_POLICY_RETURN_STEP3_CONTRACT",
    "DirectTFVMPCResultV11",
    "DirectTFVPolicyReturnMPCV11",
]
