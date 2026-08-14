"""Project7 V127 bounded continuous differentiable rolling MPC.

RBC provides safety; the differentiable surrogate provides optimization. Sparse-RBC is
one warm start and the fail-safe, never the Step2 reference or action-space ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from .step2_differentiable_v127 import ControlOrientedDifferentiableSurrogateV127
from .step2_v60_contract import require_feature
from .step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123

V127_STEP3_CONTRACT = "PROJECT7_V127_109ACT_H120_LBFGSB_RECEDING_HORIZON_MPC_V2_SMOOTH_GRAD_HARD_ACCEPT"


@dataclass(frozen=True)
class Step2GradientEvidenceV127:
    holdout_rank: float
    holdout_top1: float
    d2_gradient_sign_accuracy: float
    d2_gradient_cosine_similarity: float
    d5_gradient_sign_accuracy: float
    d5_gradient_cosine_similarity: float
    causal_step1_state_verified: bool
    causal_rainfall_verified: bool

    def validate(self) -> None:
        values = (
            self.holdout_rank, self.holdout_top1, self.d2_gradient_sign_accuracy,
            self.d2_gradient_cosine_similarity, self.d5_gradient_sign_accuracy,
            self.d5_gradient_cosine_similarity,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("V127 continuous gate contains non-finite evidence")
        failures = []
        for label, value, threshold in (
            ("rank", self.holdout_rank, 0.70),
            ("top1", self.holdout_top1, 0.50),
            ("D2 gradient sign", self.d2_gradient_sign_accuracy, 0.70),
            ("D2 gradient cosine", self.d2_gradient_cosine_similarity, 0.60),
            ("D5 gradient sign", self.d5_gradient_sign_accuracy, 0.70),
            ("D5 gradient cosine", self.d5_gradient_cosine_similarity, 0.60),
        ):
            if float(value) < threshold:
                failures.append(f"{label}={float(value):.4f}<{threshold:.2f}")
        if not self.causal_step1_state_verified:
            failures.append("causal Step1 state not verified")
        if not self.causal_rainfall_verified:
            failures.append("causal rainfall not verified")
        if failures:
            raise ValueError("V127 continuous MPC blocked: " + "; ".join(failures))


@dataclass(frozen=True)
class ContinuousMPCDesignV127:
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    free_control_blocks: int = 12
    max_setting_delta_per_update: float = 0.50
    lbfgsb_maxiter: int = 30
    lbfgsb_ftol: float = 1.0e-7
    lbfgsb_gtol: float = 1.0e-5
    cvar_alpha: float = 0.90
    pfv_soft_margin_m3: float = 100.0
    pfv_penalty_weight: float = 1.0
    movement_penalty_m3: float = 1.0
    min_improvement_vs_rbc_m3: float = 1.0

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V127 MPC requires 300-s model / 600-s control")
        if (self.prediction_horizon_steps, self.free_control_blocks) != (72, 12):
            raise ValueError("V127 MPC requires H360 prediction / H120 free control")
        if not 0 < self.max_setting_delta_per_update <= 0.5:
            raise ValueError("V127 max target movement must be in (0,0.5]")
        if self.lbfgsb_maxiter <= 0 or self.lbfgsb_ftol <= 0 or self.lbfgsb_gtol <= 0:
            raise ValueError("V127 L-BFGS-B parameters are invalid")
        if not 0 <= self.cvar_alpha < 1:
            raise ValueError("V127 CVaR alpha must lie in [0,1)")
        if min(self.pfv_soft_margin_m3, self.pfv_penalty_weight, self.movement_penalty_m3, self.min_improvement_vs_rbc_m3) < 0:
            raise ValueError("V127 secondary/regularization parameters must be non-negative")


@dataclass(frozen=True)
class ContinuousMPCResultV127:
    settings: torch.Tensor
    candidate_valid: bool
    predicted_delta_tfv_m3: float
    selected_group_score_m3: float
    selected_candidate_index: int
    candidate_count: int
    first_move_group_count: int
    scenario_count: int
    tail_only_noop_candidate_count: int
    optimisation_steps: int
    selected_source: str
    continuous_optimizer_success: bool
    continuous_objective_m3: float
    continuous_tfv_risk_m3: float
    continuous_pfv_risk_m3: float
    continuous_optimization_tfv_m3: float
    continuous_optimization_pfv_m3: float
    rbc_objective_m3: float
    rbc_tfv_risk_m3: float
    rbc_pfv_risk_m3: float
    rbc_optimization_tfv_m3: float
    rbc_optimization_pfv_m3: float
    gradient_norm: float
    scipy_message: str
    variable_count: int
    policy_mode: str = "continuous_differentiable_mpc"
    policy_mode_contract: str = V127_STEP3_CONTRACT


def _cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("V127 CVaR expects [scenario]")
    count = max(1, int(math.ceil((1.0 - float(alpha)) * values.numel())))
    return torch.topk(values, k=count, largest=True).values.mean()


def decode_fractional_targets_v127(
    fractions: torch.Tensor, *, active_target: torch.Tensor, min_setting: torch.Tensor,
    max_setting: torch.Tensor, design: ContinuousMPCDesignV127,
) -> torch.Tensor:
    design.validate()
    if fractions.ndim != 2 or fractions.shape[0] != design.free_control_blocks:
        raise ValueError("V127 fractions must be [12,actuator]")
    active, lo, hi = active_target.reshape(-1).to(fractions), min_setting.reshape(-1).to(fractions), max_setting.reshape(-1).to(fractions)
    if fractions.shape[1] != active.numel() or lo.numel() != active.numel() or hi.numel() != active.numel():
        raise ValueError("V127 actuator dimensions do not align")
    if bool(torch.any(lo > hi)) or bool(torch.any((fractions < 0) | (fractions > 1))):
        raise ValueError("V127 fraction/bounds invalid")
    previous = active; free = []
    for k in range(design.free_control_blocks):
        lower = torch.maximum(lo, previous - design.max_setting_delta_per_update)
        upper = torch.minimum(hi, previous + design.max_setting_delta_per_update)
        target = lower + fractions[k] * (upper - lower)
        free.append(target); previous = target
    blocks = torch.stack(free)
    total_blocks = design.prediction_horizon_steps // design.control_block_steps
    if total_blocks > len(free):
        blocks = torch.cat((blocks, blocks[-1:].expand(total_blocks - len(free), -1)), dim=0)
    return blocks.repeat_interleave(design.control_block_steps, dim=0)


def encode_sequence_to_fraction_v127(
    sequence: np.ndarray, *, active_target: np.ndarray, min_setting: np.ndarray,
    max_setting: np.ndarray, design: ContinuousMPCDesignV127,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=float)
    if values.shape != (design.prediction_horizon_steps, len(active_target)):
        raise ValueError("V127 seed sequence shape mismatch")
    targets = values[::design.control_block_steps][:design.free_control_blocks]
    previous = np.asarray(active_target, dtype=float).reshape(-1)
    lo, hi = np.asarray(min_setting, dtype=float), np.asarray(max_setting, dtype=float)
    result = []
    for target in targets:
        lower = np.maximum(lo, previous - design.max_setting_delta_per_update)
        upper = np.minimum(hi, previous + design.max_setting_delta_per_update)
        width = upper - lower
        result.append(np.clip(np.where(width > 1e-12, (target - lower) / width, 0.5), 0, 1))
        previous = np.clip(target, lower, upper)
    return np.stack(result).astype(np.float64)


class DifferentiableRollingMPCV127:
    accepts_previous_requested_settings = True
    policy_mode = "continuous_differentiable_mpc"
    policy_mode_contract = V127_STEP3_CONTRACT

    def __init__(self, *, model: ControlOrientedDifferentiableSurrogateV127, graph: Any,
                 priority_indices: torch.Tensor, evidence: Step2GradientEvidenceV127,
                 flood_rate_index: int, design: ContinuousMPCDesignV127 = ContinuousMPCDesignV127()) -> None:
        evidence.validate(); design.validate()
        if len(graph.actuator_ids) != 109:
            raise ValueError("V127 requires 109 actuators")
        self.model, self.graph, self.priority_indices = model, graph, priority_indices.long()
        self.evidence, self.flood_rate_index, self.design = evidence, int(flood_rate_index), design
        names = tuple(graph.actuator_physics_feature_names); physics = np.asarray(graph.actuator_physics, dtype=np.float32)
        self.min_setting = physics[:, require_feature(names, "min_setting")].astype(np.float32)
        self.max_setting = physics[:, require_feature(names, "max_setting")].astype(np.float32)
        self._last_fraction: np.ndarray | None = None

    def _rbc_sequence(self, initial_state, current_settings, active_target) -> torch.Tensor:
        fallback = active_target.reshape(1, -1).expand(self.design.prediction_horizon_steps, -1)
        raw = build_sparse_state_auto_rbc_anchor_v123(
            initial_state[0].detach().cpu().numpy(), current_settings.detach().cpu().numpy(),
            fallback.detach().cpu().numpy(), self.graph,
            control_block_steps=self.design.control_block_steps,
            max_delta_per_update=self.design.max_setting_delta_per_update,
        )
        sequence = torch.as_tensor(raw, dtype=initial_state.dtype, device=initial_state.device)
        blocks = sequence[::self.design.control_block_steps].clone()
        blocks[self.design.free_control_blocks:] = blocks[self.design.free_control_blocks - 1]
        return blocks.repeat_interleave(self.design.control_block_steps, dim=0)

    def _score(self, *, initial_state, rainfall, sequence, flow, active_target, rbc_smooth_pfv):
        scenarios = int(rainfall.shape[0]); state = initial_state.expand(scenarios, -1, -1)
        output = self.model.objective_rollout(
            initial_state=state, rainfall=rainfall, settings=sequence[None].expand(scenarios, -1, -1),
            previous_actuator_flow=flow.expand(scenarios, -1),
            actuator_upstream=torch.as_tensor(self.graph.actuator_upstream, dtype=torch.long, device=state.device),
            actuator_downstream=torch.as_tensor(self.graph.actuator_downstream, dtype=torch.long, device=state.device),
            actuator_physics=torch.as_tensor(self.graph.actuator_physics, dtype=state.dtype, device=state.device),
            static_node_features=torch.as_tensor(self.graph.static_node_features, dtype=state.dtype, device=state.device),
            edge_index=torch.as_tensor(self.graph.edge_index, dtype=torch.long, device=state.device),
            flood_rate_index=self.flood_rate_index, priority_indices=self.priority_indices,
            dt_seconds=float(self.design.model_step_seconds),
        )
        hard_tfv, hard_pfv = _cvar(output.tfv_m3, self.design.cvar_alpha), _cvar(output.pfv_m3, self.design.cvar_alpha)
        smooth_tfv, smooth_pfv = _cvar(output.optimization_tfv_m3, self.design.cvar_alpha), _cvar(output.optimization_pfv_m3, self.design.cvar_alpha)
        pfv_penalty = smooth_tfv.new_zeros(())
        if rbc_smooth_pfv is not None:
            pfv_penalty = self.design.pfv_penalty_weight * torch.relu(smooth_pfv - rbc_smooth_pfv - self.design.pfv_soft_margin_m3)
        first = sequence[:self.design.control_block_steps].mean(0)
        movement = self.design.movement_penalty_m3 * torch.mean(torch.square(first - active_target))
        return smooth_tfv + pfv_penalty + movement, hard_tfv, hard_pfv, smooth_tfv, smooth_pfv

    def optimize(self, initial_state: torch.Tensor, rainfall_scenarios: torch.Tensor,
                 fallback_settings: torch.Tensor, *, current_settings=None,
                 previous_requested_settings=None, previous_actuator_flow=None,
                 max_delta_per_update=None, **_: object) -> ContinuousMPCResultV127:
        if current_settings is None or previous_requested_settings is None or previous_actuator_flow is None:
            raise ValueError("V127 requires current/target/flow readback")
        if initial_state.ndim == 2:
            initial_state = initial_state[None]
        active = previous_requested_settings.reshape(-1).to(initial_state); current = current_settings.reshape(-1).to(initial_state)
        flow = previous_actuator_flow.reshape(1, -1).to(initial_state)
        if active.numel() != 109 or current.numel() != 109 or flow.shape[-1] != 109:
            raise ValueError("V127 actuator readback count mismatch")
        if max_delta_per_update is not None and float(torch.as_tensor(max_delta_per_update).max()) > 0.500000001:
            raise ValueError("V127 runtime max delta is looser than frozen contract")
        lo = torch.as_tensor(self.min_setting, dtype=initial_state.dtype, device=initial_state.device)
        hi = torch.as_tensor(self.max_setting, dtype=initial_state.dtype, device=initial_state.device)
        rbc = self._rbc_sequence(initial_state, current, active)
        with torch.no_grad():
            rbc_score, rbc_hard_tfv, rbc_hard_pfv, rbc_smooth_tfv, rbc_smooth_pfv = self._score(
                initial_state=initial_state, rainfall=rainfall_scenarios, sequence=rbc,
                flow=flow, active_target=active, rbc_smooth_pfv=None)
        hold = active[None].expand(72, -1).clone()
        starts = [
            encode_sequence_to_fraction_v127(hold.cpu().numpy(), active_target=active.cpu().numpy(), min_setting=self.min_setting, max_setting=self.max_setting, design=self.design),
            encode_sequence_to_fraction_v127(rbc.cpu().numpy(), active_target=active.cpu().numpy(), min_setting=self.min_setting, max_setting=self.max_setting, design=self.design),
        ]
        if self._last_fraction is not None and self._last_fraction.shape == starts[0].shape:
            starts.append(np.vstack((self._last_fraction[1:], self._last_fraction[-1:])))
        best = None; total_iterations = 0; last_grad = float("nan"); messages = []; variable_count = 12 * 109

        def objective(flat):
            nonlocal last_grad
            fractions = torch.as_tensor(flat.reshape(12, 109), dtype=initial_state.dtype, device=initial_state.device).requires_grad_(True)
            sequence = decode_fractional_targets_v127(fractions, active_target=active, min_setting=lo, max_setting=hi, design=self.design)
            score, _, _, _, _ = self._score(initial_state=initial_state, rainfall=rainfall_scenarios,
                                             sequence=sequence, flow=flow, active_target=active,
                                             rbc_smooth_pfv=rbc_smooth_pfv.detach())
            if not bool(torch.isfinite(score)):
                raise FloatingPointError("V127 smooth MPC objective non-finite")
            grad = torch.autograd.grad(score, fractions)[0]
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("V127 smooth MPC gradient non-finite")
            last_grad = float(torch.linalg.vector_norm(grad).detach())
            return float(score.detach()), grad.detach().cpu().numpy().astype(np.float64).reshape(-1)

        for i, start in enumerate(starts):
            try:
                result = minimize(objective, np.clip(start, 0, 1).reshape(-1), method="L-BFGS-B", jac=True,
                                  bounds=[(0.0, 1.0)] * variable_count,
                                  options={"maxiter": self.design.lbfgsb_maxiter, "ftol": self.design.lbfgsb_ftol,
                                           "gtol": self.design.lbfgsb_gtol, "maxls": 20})
                total_iterations += int(result.nit); messages.append(f"start{i}:{result.message}")
                fraction = np.clip(np.asarray(result.x).reshape(12, 109), 0, 1)
                with torch.no_grad():
                    sequence = decode_fractional_targets_v127(torch.as_tensor(fraction, dtype=initial_state.dtype, device=initial_state.device),
                                                              active_target=active, min_setting=lo, max_setting=hi, design=self.design)
                    score, hard_tfv, hard_pfv, smooth_tfv, smooth_pfv = self._score(
                        initial_state=initial_state, rainfall=rainfall_scenarios, sequence=sequence,
                        flow=flow, active_target=active, rbc_smooth_pfv=rbc_smooth_pfv)
                record = dict(score=float(score), hard_tfv=float(hard_tfv), hard_pfv=float(hard_pfv),
                              smooth_tfv=float(smooth_tfv), smooth_pfv=float(smooth_pfv), sequence=sequence,
                              fraction=fraction, success=bool(result.success), message=str(result.message), start=i)
                if best is None or record["score"] < best["score"]:
                    best = record
            except Exception as exc:
                messages.append(f"start{i}:{type(exc).__name__}:{str(exc)[:180]}")

        use_continuous = bool(best is not None and best["success"] and math.isfinite(best["hard_tfv"])
                              and best["hard_tfv"] < float(rbc_hard_tfv) - self.design.min_improvement_vs_rbc_m3
                              and best["score"] < float(rbc_score))
        if use_continuous:
            selected = best; source = "continuous_lbfgsb"; self._last_fraction = np.asarray(best["fraction"])
        else:
            selected = dict(score=float(rbc_score), hard_tfv=float(rbc_hard_tfv), hard_pfv=float(rbc_hard_pfv),
                            smooth_tfv=float(rbc_smooth_tfv), smooth_pfv=float(rbc_smooth_pfv), sequence=rbc,
                            fraction=starts[1], success=False, message="RBC safety fallback", start=1)
            source = "rbc_safety_fallback"; self._last_fraction = starts[1].copy()
        sequence = selected["sequence"]
        if not bool(torch.allclose(sequence[0], sequence[1], rtol=0.0, atol=1e-7)):
            raise RuntimeError("V127 selected first 10-min target is not constant")
        return ContinuousMPCResultV127(
            settings=sequence.detach(), candidate_valid=True,
            predicted_delta_tfv_m3=float(selected["hard_tfv"] - float(rbc_hard_tfv)),
            selected_group_score_m3=float(selected["score"]), selected_candidate_index=int(selected["start"]),
            candidate_count=len(starts), first_move_group_count=len(starts), scenario_count=int(rainfall_scenarios.shape[0]),
            tail_only_noop_candidate_count=0, optimisation_steps=total_iterations, selected_source=source,
            continuous_optimizer_success=use_continuous, continuous_objective_m3=float(selected["score"]),
            continuous_tfv_risk_m3=float(selected["hard_tfv"]), continuous_pfv_risk_m3=float(selected["hard_pfv"]),
            continuous_optimization_tfv_m3=float(selected["smooth_tfv"]), continuous_optimization_pfv_m3=float(selected["smooth_pfv"]),
            rbc_objective_m3=float(rbc_score), rbc_tfv_risk_m3=float(rbc_hard_tfv), rbc_pfv_risk_m3=float(rbc_hard_pfv),
            rbc_optimization_tfv_m3=float(rbc_smooth_tfv), rbc_optimization_pfv_m3=float(rbc_smooth_pfv),
            gradient_norm=float(last_grad), scipy_message=" | ".join(messages)[:2000], variable_count=variable_count)


__all__ = ["ContinuousMPCDesignV127", "ContinuousMPCResultV127", "DifferentiableRollingMPCV127",
           "Step2GradientEvidenceV127", "V127_STEP3_CONTRACT", "decode_fractional_targets_v127",
           "encode_sequence_to_fraction_v127"]
