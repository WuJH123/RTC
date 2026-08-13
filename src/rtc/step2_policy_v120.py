"""V12.0 value-only candidate policy for Project7 rolling RTC.

The policy searches the same low-dimensional candidate families used for targeted
D3-v2 generation. It never needs a nodewise Hydraulic rollout: the direct Value
model scores joint action sequences by signed authoritative Delta-TFV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .step2_control_basis_v60 import ControlBasisV60
from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_d3_design_v60 import D3V60DesignContract, _coefficient_specs
from .step2_v120_contract import Step2V120Contract


@dataclass(frozen=True)
class RuntimeNormalizationV120:
    state_mean: np.ndarray
    state_std: np.ndarray
    rainfall_mean: np.ndarray
    rainfall_std: np.ndarray
    flow_mean: np.ndarray
    flow_std: np.ndarray

    def validate(self) -> None:
        for name, mean, std in (
            ("state", self.state_mean, self.state_std),
            ("rainfall", self.rainfall_mean, self.rainfall_std),
            ("flow", self.flow_mean, self.flow_std),
        ):
            m = np.asarray(mean, dtype=np.float32).reshape(-1)
            s = np.asarray(std, dtype=np.float32).reshape(-1)
            if m.shape != s.shape or not m.size:
                raise ValueError(f"V120 {name} normalization shape mismatch")
            if not np.isfinite(m).all() or not np.isfinite(s).all() or np.any(s <= 0):
                raise ValueError(f"V120 {name} normalization is invalid")

    def as_payload(self) -> dict[str, list[float]]:
        self.validate()
        return {
            "state_mean": np.asarray(self.state_mean, np.float32).reshape(-1).tolist(),
            "state_std": np.asarray(self.state_std, np.float32).reshape(-1).tolist(),
            "rainfall_mean": np.asarray(self.rainfall_mean, np.float32).reshape(-1).tolist(),
            "rainfall_std": np.asarray(self.rainfall_std, np.float32).reshape(-1).tolist(),
            "flow_mean": np.asarray(self.flow_mean, np.float32).reshape(-1).tolist(),
            "flow_std": np.asarray(self.flow_std, np.float32).reshape(-1).tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "RuntimeNormalizationV120":
        result = cls(
            state_mean=np.asarray(payload["state_mean"], np.float32),
            state_std=np.asarray(payload["state_std"], np.float32),
            rainfall_mean=np.asarray(payload["rainfall_mean"], np.float32),
            rainfall_std=np.asarray(payload["rainfall_std"], np.float32),
            flow_mean=np.asarray(payload["flow_mean"], np.float32),
            flow_std=np.asarray(payload["flow_std"], np.float32),
        )
        result.validate()
        return result

    @staticmethod
    def _scale(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        m = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
        s = torch.as_tensor(std, dtype=value.dtype, device=value.device)
        return (value - m) / s

    def state(self, value: torch.Tensor) -> torch.Tensor:
        return self._scale(value, self.state_mean, self.state_std)

    def rainfall(self, value: torch.Tensor) -> torch.Tensor:
        return self._scale(value, self.rainfall_mean, self.rainfall_std)

    def flow(self, value: torch.Tensor) -> torch.Tensor:
        return self._scale(value, self.flow_mean, self.flow_std)


@dataclass(frozen=True)
class ValueOnlyPolicyResultV120:
    settings: torch.Tensor
    candidate_valid: bool
    tfv_risk_m3: float
    predicted_delta_tfv_m3: float
    selected_candidate_index: int
    candidate_count: int
    scenario_count: int
    reference_is_best: bool


def _upper_tail_cvar_per_candidate(values: torch.Tensor, alpha: float) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("V120 scenario values must be [S,C]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("V120 CVaR alpha must lie in [0,1)")
    count = max(1, int(math.ceil((1.0 - alpha) * values.shape[0])))
    return torch.topk(values, k=count, dim=0, largest=True).values.mean(dim=0)


def candidate_coefficients_v120(
    basis: ControlBasisV60,
    *,
    contract: Step2V120Contract = Step2V120Contract(),
) -> np.ndarray:
    """D3-v2-distribution candidate banks, with exact HOLD at index zero."""
    contract.validate()
    d3 = D3V60DesignContract(
        candidates_per_checkpoint=contract.candidates_per_bank,
        seed=contract.seed,
    )
    rows = [np.zeros((basis.temporal_basis_count, basis.group_count), np.float32)]
    seen = {rows[0].tobytes()}
    for bank in range(contract.candidate_banks):
        checkpoint_seed = int(contract.seed + 104729 * (bank + 1))
        for _, coeff in _coefficient_specs(
            basis, checkpoint_seed=checkpoint_seed, contract=d3
        ):
            value = np.asarray(coeff, np.float32)
            key = value.tobytes()
            if key not in seen:
                rows.append(value)
                seen.add(key)
    result = np.stack(rows)
    if result.shape[1:] != (basis.temporal_basis_count, basis.group_count):
        raise RuntimeError("V120 coefficient-bank shape drift")
    if not np.allclose(result[0], 0.0):
        raise RuntimeError("V120 candidate zero must be HOLD")
    return result


class ValueOnlyCandidatePolicyV120:
    """Robust finite-candidate policy on the frozen D3 control manifold."""

    def __init__(
        self,
        *,
        model: ControlValueSurrogateV70,
        basis: ControlBasisV60,
        prepared: PreparedStaticV60,
        normalization: RuntimeNormalizationV120,
        cvar_alpha: float = 0.90,
        min_predicted_improvement_m3: float = 0.0,
        movement_tiebreak: float = 1.0e-6,
        contract: Step2V120Contract = Step2V120Contract(),
    ) -> None:
        contract.validate()
        normalization.validate()
        basis.validate()
        self.model = model
        self.basis = basis
        self.prepared = prepared
        self.normalization = normalization
        self.cvar_alpha = float(cvar_alpha)
        self.min_predicted_improvement_m3 = max(float(min_predicted_improvement_m3), 0.0)
        self.movement_tiebreak = max(float(movement_tiebreak), 0.0)
        self.contract = contract
        self._coefficients = candidate_coefficients_v120(basis, contract=contract)

    def optimize(
        self,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        fallback_settings: torch.Tensor,
        *,
        current_settings: torch.Tensor | None = None,
        previous_actuator_flow: torch.Tensor | None = None,
        max_delta_per_update: float | torch.Tensor | None = None,
        **_controller_compatibility: object,
    ) -> ValueOnlyPolicyResultV120:
        # TorchMPCController passes the Step1 batch dimension (B=1).
        if initial_state.ndim == 3 and initial_state.shape[0] == 1:
            initial_state = initial_state[0]
        if initial_state.ndim != 2:
            raise ValueError("V120 initial_state must resolve to [N,D]")
        if rainfall_scenarios.ndim != 4:
            raise ValueError("V120 rainfall scenarios must be [S,H,N,R]")
        if fallback_settings.ndim != 3 or fallback_settings.shape[0] != 1:
            raise ValueError("V120 fallback settings must be [1,H,A]")
        if fallback_settings.shape[1] != self.basis.horizon.horizon_steps:
            raise ValueError("V120 fallback/value horizons differ")
        actuator_count = self.basis.grouping.actuator_count
        if previous_actuator_flow is None:
            previous_actuator_flow = torch.zeros(
                actuator_count, dtype=initial_state.dtype, device=initial_state.device
            )
        previous_actuator_flow = previous_actuator_flow.reshape(-1)
        if previous_actuator_flow.numel() != actuator_count:
            raise ValueError("V120 previous actuator flow count mismatch")
        scenarios = int(rainfall_scenarios.shape[0])
        coeff = torch.as_tensor(
            self._coefficients,
            dtype=fallback_settings.dtype,
            device=fallback_settings.device,
        )
        reference_one = fallback_settings
        reference_for_candidates = reference_one[:, None].expand(
            1, coeff.shape[0], -1, -1
        )
        candidate_one = self.basis.decode(reference_for_candidates, coeff[None])[0]
        if max_delta_per_update is not None:
            frozen = float(self.basis.contract.max_setting_delta_per_update)
            requested = float(torch.as_tensor(max_delta_per_update).reshape(-1).max())
            if requested > frozen + 1e-9:
                raise ValueError("V120 runtime max delta is looser than frozen basis")

        state = initial_state[None].expand(scenarios, -1, -1)
        reference = reference_one.expand(scenarios, -1, -1)
        candidate = candidate_one[None].expand(scenarios, -1, -1, -1)
        flow = previous_actuator_flow[None].expand(scenarios, -1)
        output = self.model(
            self.normalization.state(state),
            self.normalization.rainfall(rainfall_scenarios),
            reference,
            candidate,
            self.normalization.flow(flow),
            self.prepared,
        )
        predicted = output.delta_tfv_m3
        if not torch.isfinite(predicted).all():
            raise RuntimeError("V120 Value produced non-finite Delta-TFV")
        risk = _upper_tail_cvar_per_candidate(predicted, self.cvar_alpha)
        if abs(float(risk[0].detach())) > 1e-5:
            raise RuntimeError("V120 HOLD lost exact-zero value")

        block = int(self.basis.horizon.control_block_steps)
        reference_first = reference_one[0, :block].mean(dim=0)
        candidate_first = candidate_one[:, :block].mean(dim=1)
        movement = torch.mean(torch.abs(candidate_first - reference_first[None]), dim=1)
        objective = risk + self.movement_tiebreak * movement
        selected = int(torch.argmin(objective).item())
        selected_risk = float(risk[selected].detach())
        valid = bool(
            selected != 0
            and -selected_risk > self.min_predicted_improvement_m3
        )
        if not valid:
            selected = 0
            selected_risk = 0.0
        return ValueOnlyPolicyResultV120(
            settings=candidate_one[selected].detach(),
            candidate_valid=valid,
            tfv_risk_m3=selected_risk,
            predicted_delta_tfv_m3=selected_risk,
            selected_candidate_index=selected,
            candidate_count=int(candidate_one.shape[0]),
            scenario_count=scenarios,
            reference_is_best=bool(selected == 0),
        )


__all__ = [
    "RuntimeNormalizationV120",
    "ValueOnlyCandidatePolicyV120",
    "ValueOnlyPolicyResultV120",
    "candidate_coefficients_v120",
]
