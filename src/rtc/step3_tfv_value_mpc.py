"""Minimal Step3 for Project7: optimize the direct Step2 delta-TFV value model.

Step3 does not need to relearn hydraulics.  It receives the current reconstructed state, causal
rainfall forecast and current actuator target, asks Step2 for predicted delta TFV of candidate
109-facility sequences, and minimizes that value.  The existing differentiable engineering
decoder is reused so score and execution use the same bounded 12 x 109 decision variables.

This module is Development-ready infrastructure only. Production runtime remains fail-closed until
the direct value checkpoint demonstrates useful held-out ranking/top1/regret.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from .step2_tfv_value import DirectFacilityTFVValueModel
from .step2_train_response_v60 import InputNormalizationV60
from .step2_v60_contract import require_feature
from .step3_mpc_v127 import (
    ContinuousMPCDesignV127,
    decode_fractional_targets_v127,
    encode_sequence_to_fraction_v127,
)


DIRECT_TFV_STEP3_CONTRACT = "PROJECT7_DIRECT_TFV_109ACT_H120_RECEDING_MPC_V1"


@dataclass(frozen=True)
class DirectTFVMPCDesign:
    maxiter: int = 25
    ftol: float = 1.0e-7
    gtol: float = 1.0e-5
    minimum_predicted_improvement_m3: float = 0.0

    def validate(self) -> None:
        if self.maxiter <= 0 or min(self.ftol, self.gtol) <= 0.0:
            raise ValueError("direct TFV MPC solver parameters are invalid")
        if not math.isfinite(self.minimum_predicted_improvement_m3) or self.minimum_predicted_improvement_m3 < 0.0:
            raise ValueError("minimum predicted improvement must be finite and non-negative")


@dataclass(frozen=True)
class DirectTFVMPCResult:
    settings: torch.Tensor
    predicted_delta_tfv_m3: float
    selected_source: str
    optimizer_success: bool
    optimizer_steps: int
    gradient_norm: float
    scipy_message: str
    variable_count: int = 1308
    policy_mode: str = "direct_tfv_value_mpc"
    policy_mode_contract: str = DIRECT_TFV_STEP3_CONTRACT


class DirectTFVRollingMPC:
    """Directly minimize predicted delta TFV relative to the current HOLD sequence."""

    accepts_previous_requested_settings = True
    policy_mode = "direct_tfv_value_mpc"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: DirectFacilityTFVValueModel,
        graph: Any,
        normalization: InputNormalizationV60,
        mpc_design: ContinuousMPCDesignV127 = ContinuousMPCDesignV127(
            pfv_penalty_weight=0.0,
            movement_penalty_m3=0.0,
            min_improvement_vs_rbc_m3=0.0,
        ),
        solver_design: DirectTFVMPCDesign = DirectTFVMPCDesign(),
    ) -> None:
        solver_design.validate()
        mpc_design.validate()
        if len(graph.actuator_ids) != 109:
            raise ValueError("direct TFV MPC requires exactly 109 actuators")
        self.model = model
        self.graph = graph
        self.normalization = normalization
        self.mpc_design = mpc_design
        self.solver_design = solver_design
        names = tuple(graph.actuator_physics_feature_names)
        physics = np.asarray(graph.actuator_physics, dtype=np.float32)
        self.min_setting = physics[:, require_feature(names, "min_setting")].astype(np.float32)
        self.max_setting = physics[:, require_feature(names, "max_setting")].astype(np.float32)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.state_mean, dtype=state.dtype, device=state.device)
        std = torch.as_tensor(self.normalization.state_std, dtype=state.dtype, device=state.device).clamp_min(1.0e-6)
        return (state - mean) / std

    def _normalize_rainfall(self, rainfall: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.rainfall_mean, dtype=rainfall.dtype, device=rainfall.device)
        std = torch.as_tensor(self.normalization.rainfall_std, dtype=rainfall.dtype, device=rainfall.device).clamp_min(1.0e-6)
        return (rainfall - mean) / std

    def _normalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.flow_mean, dtype=flow.dtype, device=flow.device)
        std = torch.as_tensor(self.normalization.flow_std, dtype=flow.dtype, device=flow.device).clamp_min(1.0e-6)
        return (flow - mean) / std

    def _hold_sequence(self, active_target: torch.Tensor) -> torch.Tensor:
        return active_target.reshape(1, -1).expand(self.mpc_design.prediction_horizon_steps, -1)

    def score_sequence(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        sequence: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        if current_state.shape[0] != 1 or previous_actuator_flow.shape[0] != 1:
            raise ValueError("direct TFV MPC expects one current hydraulic state")
        if rainfall.ndim != 4 or rainfall.shape[1] != self.mpc_design.prediction_horizon_steps:
            raise ValueError("direct TFV MPC rainfall must be [scenario,H72,node,feature]")
        scenarios = int(rainfall.shape[0])
        reference = self._hold_sequence(active_target)
        state = self._normalize_state(current_state).expand(scenarios, -1, -1)
        rain = self._normalize_rainfall(rainfall)
        flow = self._normalize_flow(previous_actuator_flow).expand(scenarios, -1)
        output = self.model(
            current_state=state,
            rainfall=rain,
            reference_settings=reference[None].expand(scenarios, -1, -1),
            candidate_settings=sequence[None].expand(scenarios, -1, -1),
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
        # The current causal forecast contract uses one scenario. Mean keeps the interface valid
        # if multiple causal forecast scenarios are introduced later without changing the target.
        return output.total_delta_tfv_m3.mean()

    def optimize(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        current_settings: torch.Tensor,
        active_target: torch.Tensor,
    ) -> DirectTFVMPCResult:
        device = current_state.device
        dtype = current_state.dtype
        hold = self._hold_sequence(active_target)
        hold_fraction = encode_sequence_to_fraction_v127(
            hold.detach().cpu().numpy(),
            active_target=active_target.detach().cpu().numpy(),
            min_setting=self.min_setting,
            max_setting=self.max_setting,
            design=self.mpc_design,
        ).reshape(-1)
        bounds = [(0.0, 1.0)] * int(hold_fraction.size)
        last_gradient = 0.0

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal last_gradient
            fractions = torch.as_tensor(flat, dtype=dtype, device=device).reshape(
                self.mpc_design.free_control_blocks, 109
            ).requires_grad_(True)
            sequence = decode_fractional_targets_v127(
                fractions,
                active_target=active_target,
                min_setting=torch.as_tensor(self.min_setting, dtype=dtype, device=device),
                max_setting=torch.as_tensor(self.max_setting, dtype=dtype, device=device),
                design=self.mpc_design,
            )
            score = self.score_sequence(
                current_state=current_state,
                rainfall=rainfall,
                sequence=sequence,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            )
            gradient = torch.autograd.grad(score, fractions, allow_unused=False)[0]
            if not bool(torch.isfinite(score)) or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("direct TFV MPC produced non-finite objective/gradient")
            last_gradient = float(torch.linalg.vector_norm(gradient).detach().cpu())
            return float(score.detach().cpu()), gradient.detach().cpu().numpy().reshape(-1).astype(np.float64)

        result = minimize(
            objective,
            hold_fraction.astype(np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(self.solver_design.maxiter),
                "ftol": float(self.solver_design.ftol),
                "gtol": float(self.solver_design.gtol),
            },
        )
        candidate_fraction = torch.as_tensor(
            np.asarray(result.x, dtype=np.float32), dtype=dtype, device=device
        ).reshape(self.mpc_design.free_control_blocks, 109)
        candidate = decode_fractional_targets_v127(
            candidate_fraction,
            active_target=active_target,
            min_setting=torch.as_tensor(self.min_setting, dtype=dtype, device=device),
            max_setting=torch.as_tensor(self.max_setting, dtype=dtype, device=device),
            design=self.mpc_design,
        )
        with torch.no_grad():
            candidate_score = float(
                self.score_sequence(
                    current_state=current_state,
                    rainfall=rainfall,
                    sequence=candidate,
                    previous_actuator_flow=previous_actuator_flow,
                    active_target=active_target,
                ).cpu()
            )
        required = float(self.solver_design.minimum_predicted_improvement_m3)
        if not math.isfinite(candidate_score) or candidate_score >= -required:
            selected = hold
            selected_score = 0.0
            source = "HOLD_NO_PREDICTED_TFV_IMPROVEMENT"
        else:
            selected = candidate
            selected_score = candidate_score
            source = "DIRECT_TFV_LBFGSB"
        return DirectTFVMPCResult(
            settings=selected,
            predicted_delta_tfv_m3=selected_score,
            selected_source=source,
            optimizer_success=bool(result.success),
            optimizer_steps=int(getattr(result, "nit", 0)),
            gradient_norm=last_gradient,
            scipy_message=str(result.message),
        )


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCDesign",
    "DirectTFVMPCResult",
    "DirectTFVRollingMPC",
]
