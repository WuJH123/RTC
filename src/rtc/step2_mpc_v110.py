"""Production-facing V7 Value + V11 Hydraulic receding-horizon MPC adapter.

The primary objective remains the frozen V7 direct signed Delta-TFV over 360 min.
V11 is used only as a secondary hydraulic discriminator inside the V7 near-optimal
set, so short-horizon hydraulic detail cannot replace the anti-myopia value target.
The returned sequence is consumed by ``TorchMPCController``, which executes only
the first 10-min move and solves the problem again at the next decision time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .step2_control_basis_v60 import ControlBasisV60
from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_control_response_v110 import ActuatorSetHydraulicResponseV110
from .tfv_mpc import TFVFirstMPCResult

V110_MPC_RUNTIME_CONTRACT = "PROJECT7_V7_VALUE_V11_HYDRAULIC_ROLLING_MPC_V1"


@dataclass(frozen=True)
class RuntimeNormalizationV110:
    state_mean: np.ndarray
    state_std: np.ndarray
    rainfall_mean: np.ndarray
    rainfall_std: np.ndarray
    flow_mean: np.ndarray
    flow_std: np.ndarray

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeNormalizationV110":
        required = (
            "state_mean", "state_std", "rainfall_mean", "rainfall_std",
            "flow_mean", "flow_std",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"V11 runtime normalization missing: {missing}")
        result = cls(**{
            name: np.asarray(payload[name], dtype=np.float32) for name in required
        })
        result.validate()
        return result

    def validate(self) -> None:
        if self.state_mean.shape != self.state_std.shape or self.state_mean.ndim != 1:
            raise ValueError("V11 state normalization is malformed")
        if self.rainfall_mean.shape != self.rainfall_std.shape or self.rainfall_mean.ndim != 1:
            raise ValueError("V11 rainfall normalization is malformed")
        if self.flow_mean.shape != self.flow_std.shape or self.flow_mean.ndim != 1:
            raise ValueError("V11 flow normalization is malformed")
        for value in (
            self.state_mean, self.state_std, self.rainfall_mean,
            self.rainfall_std, self.flow_mean, self.flow_std,
        ):
            if not np.isfinite(value).all():
                raise ValueError("V11 runtime normalization contains non-finite values")
        if np.any(self.state_std <= 0) or np.any(self.rainfall_std <= 0) or np.any(self.flow_std <= 0):
            raise ValueError("V11 runtime normalization std must be positive")

    def as_payload(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.astype(float).tolist(),
            "state_std": self.state_std.astype(float).tolist(),
            "rainfall_mean": self.rainfall_mean.astype(float).tolist(),
            "rainfall_std": self.rainfall_std.astype(float).tolist(),
            "flow_mean": self.flow_mean.astype(float).tolist(),
            "flow_std": self.flow_std.astype(float).tolist(),
        }

    @staticmethod
    def _norm(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        m = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
        s = torch.as_tensor(std, dtype=value.dtype, device=value.device).clamp_min(1.0e-6)
        return (value - m) / s

    def state(self, value: torch.Tensor) -> torch.Tensor:
        return self._norm(value, self.state_mean, self.state_std)

    def rainfall(self, value: torch.Tensor) -> torch.Tensor:
        return self._norm(value, self.rainfall_mean, self.rainfall_std)

    def flow(self, value: torch.Tensor) -> torch.Tensor:
        return self._norm(value, self.flow_mean, self.flow_std)


class _V7V11Stack(nn.Module):
    def __init__(self, value: ControlValueSurrogateV70, hydraulic: ActuatorSetHydraulicResponseV110):
        super().__init__()
        self.value = value
        self.hydraulic = hydraulic


def _upper_tail_cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("V11 MPC cannot compute CVaR of empty tensor")
    count = max(1, int(math.ceil((1.0 - float(alpha)) * flat.numel())))
    return torch.topk(flat, k=count, largest=True).values.mean()


def _integrated_positive_flood_m3(
    delta_flood_m3s: torch.Tensor, response_minutes: torch.Tensor
) -> torch.Tensor:
    """Integrate positive local flooding deterioration; [B,C,T,N] -> [B,C,N]."""
    if delta_flood_m3s.ndim != 4:
        raise ValueError("V11 flood deterioration must be [B,C,T,N]")
    times = response_minutes.to(delta_flood_m3s) * 60.0
    times = torch.cat((times.new_zeros(1), times))
    values = torch.relu(delta_flood_m3s)
    zeros = values.new_zeros((*values.shape[:2], 1, values.shape[-1]))
    values = torch.cat((zeros, values), dim=2)
    dt = times[1:] - times[:-1]
    return (0.5 * (values[:, :, 1:] + values[:, :, :-1]) * dt[None, None, :, None]).sum(dim=2)


def _deterministic_warm_start(
    temporal_basis_count: int,
    group_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    amplitude: float = 1.0e-3,
) -> torch.Tensor:
    """Leave V7's exact-zero branch so gradient-based MPC can start moving.

    V7 intentionally forces candidate==reference to exact zero with ``torch.where``.
    Starting the optimiser at exactly zero coefficients would therefore select a
    constant-zero graph branch and can yield a zero action gradient.  This tiny,
    deterministic, sign-balanced perturbation is an optimiser initialisation only;
    it is not executed unless the final candidate passes the improvement gate.
    """
    count = int(temporal_basis_count) * int(group_count)
    if count <= 0 or amplitude <= 0:
        raise ValueError("V11 MPC warm start requires positive dimensions/amplitude")
    seed = torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
    if count == 1:
        seed = torch.ones_like(seed)
    seed = seed.reshape(1, 1, int(temporal_basis_count), int(group_count))
    return float(amplitude) * seed


class V7V11RollingMPC:
    """Lexicographic V7/V11 MPC on the frozen low-dimensional control basis.

    Stage 1 minimizes robust V7 Delta-TFV. Stage 2 may move only inside a V7
    near-optimal envelope and minimizes V11-predicted positive priority-site flood
    deterioration. This preserves the frozen TFV-first / priority-secondary
    semantics while making V11 useful to distinguish hydraulically safer
    near-equivalent control sequences.
    """

    def __init__(
        self,
        *,
        value_model: ControlValueSurrogateV70,
        hydraulic_model: ActuatorSetHydraulicResponseV110,
        basis: ControlBasisV60,
        prepared: PreparedStaticV60,
        normalization: RuntimeNormalizationV110,
        priority_indices: torch.Tensor | None = None,
        tfv_cvar_alpha: float = 0.90,
        tfv_near_opt_relative: float = 0.01,
        tfv_near_opt_absolute_m3: float = 1.0,
        near_opt_penalty: float = 1.0e4,
        movement_tiebreak: float = 1.0e-6,
        min_predicted_tfv_improvement_m3: float = 0.0,
        hydraulic_secondary_iterations: int | None = None,
    ) -> None:
        basis.validate()
        normalization.validate()
        if not 0.5 < tfv_cvar_alpha < 1.0:
            raise ValueError("V11 MPC CVaR alpha must lie in (0.5,1)")
        if tfv_near_opt_relative < 0 or tfv_near_opt_absolute_m3 < 0:
            raise ValueError("V11 MPC near-optimal tolerances must be non-negative")
        if near_opt_penalty <= 0 or movement_tiebreak < 0:
            raise ValueError("V11 MPC penalty/tiebreak values are invalid")
        if min_predicted_tfv_improvement_m3 < 0:
            raise ValueError("V11 MPC minimum improvement must be non-negative")
        if hydraulic_secondary_iterations is not None and hydraulic_secondary_iterations < 0:
            raise ValueError("V11 MPC secondary iterations must be non-negative")
        self.model = _V7V11Stack(value_model, hydraulic_model)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.basis = basis
        self.prepared = prepared
        self.normalization = normalization
        self.priority_indices = None if priority_indices is None else priority_indices.long()
        self.tfv_cvar_alpha = float(tfv_cvar_alpha)
        self.tfv_near_opt_relative = float(tfv_near_opt_relative)
        self.tfv_near_opt_absolute_m3 = float(tfv_near_opt_absolute_m3)
        self.near_opt_penalty = float(near_opt_penalty)
        self.movement_tiebreak = float(movement_tiebreak)
        self.min_predicted_tfv_improvement_m3 = float(min_predicted_tfv_improvement_m3)
        self.hydraulic_secondary_iterations = (
            None if hydraulic_secondary_iterations is None else int(hydraulic_secondary_iterations)
        )
        self.runtime_contract = V110_MPC_RUNTIME_CONTRACT

    def _expand_boundary(
        self,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        reference_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scenarios = int(rainfall_scenarios.shape[0])
        initial = initial_state
        previous = previous_actuator_flow
        reference = reference_settings
        if initial.shape[0] == 1 and scenarios > 1:
            initial = initial.expand(scenarios, -1, -1)
        if previous.shape[0] == 1 and scenarios > 1:
            previous = previous.expand(scenarios, -1)
        if reference.shape[0] == 1 and scenarios > 1:
            reference = reference.expand(scenarios, -1, -1)
        if initial.shape[0] != scenarios or previous.shape[0] != scenarios or reference.shape[0] != scenarios:
            raise ValueError("V11 MPC scenario batch dimensions are inconsistent")
        return (
            self.normalization.state(initial),
            self.normalization.rainfall(rainfall_scenarios),
            reference,
            self.normalization.flow(previous),
        )

    def _candidate_from_coefficients(
        self, reference: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        scenarios = reference.shape[0]
        coeff = coefficients.expand(scenarios, 1, -1, -1)
        return self.basis.decode(reference[:, None], coeff)

    def _value(
        self,
        initial: torch.Tensor,
        rainfall: torch.Tensor,
        reference: torch.Tensor,
        previous: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model.value(
            initial, rainfall, reference, candidate, previous, self.prepared
        )
        scenario_delta = output.delta_tfv_m3[:, 0]
        return _upper_tail_cvar(scenario_delta, self.tfv_cvar_alpha), scenario_delta

    def _hydraulic(
        self,
        initial: torch.Tensor,
        rainfall: torch.Tensor,
        reference: torch.Tensor,
        previous: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        output = self.model.hydraulic(
            initial, rainfall, reference, candidate, previous, self.prepared
        )
        node_positive = _integrated_positive_flood_m3(
            output.raw_delta_states_physical[..., 2], output.response_minutes
        )
        if self.priority_indices is None or not self.priority_indices.numel():
            robust = node_positive.new_tensor(0.0)
        else:
            p = self.priority_indices.to(node_positive.device)
            scenario_priority = node_positive.index_select(-1, p).sum(dim=-1)[:, 0]
            robust = _upper_tail_cvar(scenario_priority, self.tfv_cvar_alpha)
        return robust, output, node_positive[:, 0]

    def optimize(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        current_settings: torch.Tensor,
        fallback_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
        static_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        iterations: int = 120,
        learning_rate: float = 0.04,
        control_block_steps: int = 2,
        max_setting_delta_per_update: float | torch.Tensor | None = None,
    ) -> TFVFirstMPCResult:
        del actuator_upstream, actuator_downstream, actuator_physics, static_node_features, edge_index
        if iterations < 2 or learning_rate <= 0:
            raise ValueError("V11 MPC optimization settings are invalid")
        if control_block_steps != self.basis.horizon.control_block_steps:
            raise ValueError("V11 MPC control block differs from frozen V7 basis")
        horizon = int(rainfall_scenarios.shape[1])
        if horizon != self.basis.horizon.horizon_steps:
            raise ValueError("V11 MPC rainfall horizon differs from frozen 360-min value horizon")
        if fallback_settings.ndim != 3 or fallback_settings.shape[1:] != (
            horizon, self.basis.grouping.actuator_count
        ):
            raise ValueError("V11 MPC fallback/reference settings have wrong shape")
        if max_setting_delta_per_update is not None:
            raw = torch.as_tensor(max_setting_delta_per_update).reshape(-1)
            if raw.numel() not in (1, self.basis.grouping.actuator_count):
                raise ValueError("V11 MPC max setting delta has wrong actuator dimension")
            if float(raw.max()) > float(self.basis.contract.max_setting_delta_per_update) + 1.0e-9:
                raise ValueError("runtime max setting delta exceeds frozen V7 control basis")

        initial, rainfall, reference, previous = self._expand_boundary(
            initial_state, rainfall_scenarios, fallback_settings, previous_actuator_flow
        )
        device, dtype = initial.device, initial.dtype
        coefficients = nn.Parameter(
            _deterministic_warm_start(
                self.basis.temporal_basis_count,
                self.basis.group_count,
                device=device,
                dtype=dtype,
            )
        )
        optimizer = torch.optim.Adam([coefficients], lr=float(learning_rate))
        current = current_settings.reshape(-1).to(device=device, dtype=dtype)

        def movement(candidate: torch.Tensor) -> torch.Tensor:
            return (candidate[0, 0, 0] - current).square().mean()

        with torch.no_grad():
            fallback_candidate = reference[:, None]
            fallback_delta, _ = self._value(initial, rainfall, reference, previous, fallback_candidate)
            best_coeff: torch.Tensor | None = None
            best_primary = float(fallback_delta)

        default_primary = max(1, int(round(int(iterations) * 0.7)))
        if self.hydraulic_secondary_iterations is None:
            primary_iterations = default_primary
            secondary_iterations = max(0, int(iterations) - primary_iterations)
        else:
            secondary_iterations = min(int(iterations) - 1, self.hydraulic_secondary_iterations)
            primary_iterations = max(1, int(iterations) - secondary_iterations)
        if self.priority_indices is None or not self.priority_indices.numel():
            secondary_iterations = 0
            primary_iterations = int(iterations)

        for _ in range(primary_iterations):
            optimizer.zero_grad(set_to_none=True)
            candidate = self._candidate_from_coefficients(reference, coefficients)
            value, _ = self._value(initial, rainfall, reference, previous, candidate)
            loss = value + self.movement_tiebreak * movement(candidate)
            if not bool(torch.isfinite(loss)):
                break
            if float(value.detach()) < best_primary:
                best_primary = float(value.detach())
                best_coeff = coefficients.detach().clone()
            loss.backward()
            if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                break
            optimizer.step()
            with torch.no_grad():
                coefficients.clamp_(
                    -float(self.basis.contract.coefficient_limit),
                    float(self.basis.contract.coefficient_limit),
                )

        with torch.no_grad():
            candidate = self._candidate_from_coefficients(reference, coefficients)
            tail_value, _ = self._value(initial, rainfall, reference, previous, candidate)
            if float(tail_value) < best_primary:
                best_primary = float(tail_value)
                best_coeff = coefficients.detach().clone()
            if best_coeff is None:
                # No strictly better candidate was discovered. The final
                # admissibility gate will therefore fall back instead of executing
                # the deterministic warm-start perturbation.
                primary_candidate = reference[:, None]
                primary_value = fallback_delta
            else:
                coefficients.copy_(best_coeff)
                primary_candidate = self._candidate_from_coefficients(reference, coefficients)
                primary_value, _ = self._value(
                    initial, rainfall, reference, previous, primary_candidate
                )
            allowed = (
                primary_value
                + self.tfv_near_opt_absolute_m3
                + self.tfv_near_opt_relative * primary_value.abs()
            )
            primary_hydraulic, _, _ = self._hydraulic(
                initial, rainfall, reference, previous, primary_candidate
            )
            best_secondary_coeff = None if best_coeff is None else best_coeff.clone()
            best_secondary_hydraulic = float(primary_hydraulic)
            best_secondary_value = float(primary_value)

        if best_coeff is not None:
            for _ in range(secondary_iterations):
                optimizer.zero_grad(set_to_none=True)
                candidate = self._candidate_from_coefficients(reference, coefficients)
                value, _ = self._value(initial, rainfall, reference, previous, candidate)
                hydraulic, _, _ = self._hydraulic(
                    initial, rainfall, reference, previous, candidate
                )
                loss = (
                    hydraulic
                    + self.near_opt_penalty * torch.relu(value - allowed).square()
                    + self.movement_tiebreak * movement(candidate)
                )
                if not bool(torch.isfinite(loss)):
                    break
                if float(value.detach()) <= float(allowed) + 1.0e-6:
                    h = float(hydraulic.detach())
                    v = float(value.detach())
                    if h < best_secondary_hydraulic - 1.0e-9 or (
                        abs(h - best_secondary_hydraulic) <= 1.0e-9 and v < best_secondary_value
                    ):
                        best_secondary_hydraulic = h
                        best_secondary_value = v
                        best_secondary_coeff = coefficients.detach().clone()
                loss.backward()
                if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                    break
                optimizer.step()
                with torch.no_grad():
                    coefficients.clamp_(
                        -float(self.basis.contract.coefficient_limit),
                        float(self.basis.contract.coefficient_limit),
                    )

        with torch.no_grad():
            if best_secondary_coeff is None:
                candidate = reference[:, None]
            else:
                coefficients.copy_(best_secondary_coeff)
                candidate = self._candidate_from_coefficients(reference, coefficients)
            final_value, _ = self._value(initial, rainfall, reference, previous, candidate)
            _, hydraulic_output, node_positive = self._hydraulic(
                initial, rainfall, reference, previous, candidate
            )
            settings = candidate[0, 0]
            if not bool(torch.isfinite(settings).all()):
                raise RuntimeError("V11 MPC produced non-finite settings")
            if torch.any(settings < -1.0e-6) or torch.any(settings > 1.0 + 1.0e-6):
                raise RuntimeError("V11 MPC produced settings outside [0,1]")

            predicted_improvement = float(fallback_delta - final_value)
            numeric_margin = max(
                1.0e-6,
                1.0e-9 * max(abs(float(fallback_delta)), 1.0),
            )
            required_improvement = self.min_predicted_tfv_improvement_m3
            candidate_valid = bool(
                best_secondary_coeff is not None
                and torch.isfinite(final_value).item()
                and predicted_improvement >= required_improvement + numeric_margin
            )
            p_positive = 0.0
            if self.priority_indices is not None and self.priority_indices.numel():
                p = self.priority_indices.to(node_positive.device)
                p_positive = float(node_positive.index_select(-1, p).sum(dim=-1).max())
            site_positive = node_positive.max(dim=0).values
            worst_site_flood = float(site_positive.max())
            depth_delta = torch.relu(hydraulic_output.raw_delta_states_physical[..., 0])
            worst_depth = float(depth_delta.max())
            near_excess = max(0.0, float(final_value - allowed))

        return TFVFirstMPCResult(
            settings=settings,
            candidate_valid=candidate_valid,
            tfv_risk_m3=float(final_value),
            primary_tfv_reference_m3=float(fallback_delta),
            priority_positive_flood_deterioration_m3=p_positive,
            worst_site_flood_deterioration_m3=worst_site_flood,
            worst_site_depth_deterioration_m=worst_depth,
            tfv_near_opt_excess_m3=near_excess,
        )


__all__ = [
    "RuntimeNormalizationV110",
    "V110_MPC_RUNTIME_CONTRACT",
    "V7V11RollingMPC",
    "_deterministic_warm_start",
    "_integrated_positive_flood_m3",
]
