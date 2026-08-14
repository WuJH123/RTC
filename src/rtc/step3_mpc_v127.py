"""Project7 V127 bounded continuous differentiable rolling MPC.

RBC provides safety; differentiable MPC provides optimization. Sparse-RBC is one warm
start and the fail-safe, never the Step2 reference or action-space ceiling. The optimizer
is explicitly deadline-bounded inside objective/gradient evaluations so a stalled solve
cannot consume the next 10-minute control period.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from .step2_differentiable_v127 import ControlOrientedDifferentiableSurrogateV127
from .step2_v60_contract import require_feature
from .step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123

V127_STEP3_CONTRACT = "PROJECT7_V127_109ACT_H120_LBFGSB_RECEDING_HORIZON_MPC_V3_DEADLINE"


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
            self.holdout_rank,
            self.holdout_top1,
            self.d2_gradient_sign_accuracy,
            self.d2_gradient_cosine_similarity,
            self.d5_gradient_sign_accuracy,
            self.d5_gradient_cosine_similarity,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("V127 continuous gate contains non-finite evidence")
        failures: list[str] = []
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
    optimizer_deadline_seconds: float = 480.0
    cvar_alpha: float = 0.90
    pfv_soft_margin_m3: float = 100.0
    pfv_penalty_weight: float = 1.0
    movement_penalty_m3: float = 1.0
    min_improvement_vs_rbc_m3: float = 1.0

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    @property
    def variable_count(self) -> int:
        return self.free_control_blocks * 109

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V127 MPC requires 300-s model / 600-s control")
        if (self.prediction_horizon_steps, self.free_control_blocks) != (72, 12):
            raise ValueError("V127 MPC requires H360 prediction / H120 free control")
        if not 0.0 < self.max_setting_delta_per_update <= 0.5:
            raise ValueError("V127 max target movement must be in (0,0.5]")
        if self.lbfgsb_maxiter <= 0 or self.lbfgsb_ftol <= 0 or self.lbfgsb_gtol <= 0:
            raise ValueError("V127 L-BFGS-B parameters are invalid")
        if not 0.0 < self.optimizer_deadline_seconds < self.control_update_seconds:
            raise ValueError("V127 optimizer deadline must be inside one control period")
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("V127 CVaR alpha must lie in [0,1)")
        if min(
            self.pfv_soft_margin_m3,
            self.pfv_penalty_weight,
            self.movement_penalty_m3,
            self.min_improvement_vs_rbc_m3,
        ) < 0.0:
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
    optimizer_elapsed_seconds: float
    deadline_exceeded: bool
    policy_mode: str = "continuous_differentiable_mpc"
    policy_mode_contract: str = V127_STEP3_CONTRACT


def _cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("V127 CVaR expects one value per rainfall scenario")
    count = max(1, int(math.ceil((1.0 - float(alpha)) * values.numel())))
    return torch.topk(values, k=count, largest=True).values.mean()


def decode_fractional_targets_v127(
    fractions: torch.Tensor,
    *,
    active_target: torch.Tensor,
    min_setting: torch.Tensor,
    max_setting: torch.Tensor,
    design: ContinuousMPCDesignV127,
) -> torch.Tensor:
    design.validate()
    if fractions.ndim != 2 or fractions.shape[0] != design.free_control_blocks:
        raise ValueError("V127 fractions must be [12, actuator]")
    active = active_target.reshape(-1).to(fractions)
    lo = min_setting.reshape(-1).to(fractions)
    hi = max_setting.reshape(-1).to(fractions)
    if fractions.shape[1] != active.numel() or lo.numel() != active.numel() or hi.numel() != active.numel():
        raise ValueError("V127 actuator dimensions do not align")
    if bool(torch.any(lo > hi)) or bool(torch.any((fractions < 0.0) | (fractions > 1.0))):
        raise ValueError("V127 fractional targets or physical bounds are invalid")
    previous = active
    free: list[torch.Tensor] = []
    for k in range(design.free_control_blocks):
        lower = torch.maximum(lo, previous - design.max_setting_delta_per_update)
        upper = torch.minimum(hi, previous + design.max_setting_delta_per_update)
        target = lower + fractions[k] * (upper - lower)
        free.append(target)
        previous = target
    blocks = torch.stack(free)
    total_blocks = design.prediction_horizon_steps // design.control_block_steps
    if total_blocks < design.free_control_blocks:
        raise RuntimeError("V127 free-control horizon exceeds prediction horizon")
    if total_blocks > design.free_control_blocks:
        blocks = torch.cat(
            (blocks, blocks[-1:].expand(total_blocks - design.free_control_blocks, -1)), dim=0
        )
    return blocks.repeat_interleave(design.control_block_steps, dim=0)


def encode_sequence_to_fraction_v127(
    sequence: np.ndarray,
    *,
    active_target: np.ndarray,
    min_setting: np.ndarray,
    max_setting: np.ndarray,
    design: ContinuousMPCDesignV127,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.shape != (design.prediction_horizon_steps, len(active_target)):
        raise ValueError("V127 seed sequence shape mismatch")
    targets = values[:: design.control_block_steps][: design.free_control_blocks]
    previous = np.asarray(active_target, dtype=np.float64).reshape(-1)
    lo = np.asarray(min_setting, dtype=np.float64).reshape(-1)
    hi = np.asarray(max_setting, dtype=np.float64).reshape(-1)
    result: list[np.ndarray] = []
    for target in targets:
        lower = np.maximum(lo, previous - design.max_setting_delta_per_update)
        upper = np.minimum(hi, previous + design.max_setting_delta_per_update)
        width = upper - lower
        fraction = np.where(width > 1e-12, (target - lower) / width, 0.5)
        result.append(np.clip(fraction, 0.0, 1.0))
        previous = np.clip(target, lower, upper)
    return np.stack(result).astype(np.float64)


class DifferentiableRollingMPCV127:
    accepts_previous_requested_settings = True
    policy_mode = "continuous_differentiable_mpc"
    policy_mode_contract = V127_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: ControlOrientedDifferentiableSurrogateV127,
        graph: Any,
        priority_indices: torch.Tensor,
        evidence: Step2GradientEvidenceV127,
        flood_rate_index: int,
        design: ContinuousMPCDesignV127 = ContinuousMPCDesignV127(),
    ) -> None:
        evidence.validate()
        design.validate()
        if len(graph.actuator_ids) != 109:
            raise ValueError("V127 continuous MPC requires exactly 109 actuators")
        self.model = model
        self.graph = graph
        self.priority_indices = priority_indices.long()
        self.evidence = evidence
        self.flood_rate_index = int(flood_rate_index)
        self.design = design
        names = tuple(graph.actuator_physics_feature_names)
        physics = np.asarray(graph.actuator_physics, dtype=np.float32)
        self.min_setting = physics[:, require_feature(names, "min_setting")].astype(np.float32)
        self.max_setting = physics[:, require_feature(names, "max_setting")].astype(np.float32)
        self._last_fraction: np.ndarray | None = None

    def _rbc_sequence(
        self,
        initial_state: torch.Tensor,
        current_settings: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        fallback = active_target.reshape(1, -1).expand(self.design.prediction_horizon_steps, -1)
        raw = build_sparse_state_auto_rbc_anchor_v123(
            initial_state[0].detach().cpu().numpy(),
            current_settings.detach().cpu().numpy(),
            fallback.detach().cpu().numpy(),
            self.graph,
            control_block_steps=self.design.control_block_steps,
            max_delta_per_update=self.design.max_setting_delta_per_update,
        )
        sequence = torch.as_tensor(raw, dtype=initial_state.dtype, device=initial_state.device)
        blocks = sequence[:: self.design.control_block_steps].clone()
        blocks[self.design.free_control_blocks :] = blocks[self.design.free_control_blocks - 1]
        return blocks.repeat_interleave(self.design.control_block_steps, dim=0)

    def _score(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        sequence: torch.Tensor,
        flow: torch.Tensor,
        active_target: torch.Tensor,
        rbc_smooth_pfv: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scenarios = int(rainfall.shape[0])
        state = initial_state.expand(scenarios, -1, -1)
        output = self.model.objective_rollout(
            initial_state=state,
            rainfall=rainfall,
            settings=sequence[None].expand(scenarios, -1, -1),
            previous_actuator_flow=flow.expand(scenarios, -1),
            actuator_upstream=torch.as_tensor(
                self.graph.actuator_upstream, dtype=torch.long, device=state.device
            ),
            actuator_downstream=torch.as_tensor(
                self.graph.actuator_downstream, dtype=torch.long, device=state.device
            ),
            actuator_physics=torch.as_tensor(
                self.graph.actuator_physics, dtype=state.dtype, device=state.device
            ),
            static_node_features=torch.as_tensor(
                self.graph.static_node_features, dtype=state.dtype, device=state.device
            ),
            edge_index=torch.as_tensor(self.graph.edge_index, dtype=torch.long, device=state.device),
            flood_rate_index=self.flood_rate_index,
            priority_indices=self.priority_indices,
            dt_seconds=float(self.design.model_step_seconds),
        )
        hard_tfv = _cvar(output.tfv_m3, self.design.cvar_alpha)
        hard_pfv = _cvar(output.pfv_m3, self.design.cvar_alpha)
        smooth_tfv = _cvar(output.optimization_tfv_m3, self.design.cvar_alpha)
        smooth_pfv = _cvar(output.optimization_pfv_m3, self.design.cvar_alpha)
        pfv_penalty = smooth_tfv.new_zeros(())
        if rbc_smooth_pfv is not None:
            pfv_penalty = self.design.pfv_penalty_weight * torch.relu(
                smooth_pfv - rbc_smooth_pfv - self.design.pfv_soft_margin_m3
            )
        first = sequence[: self.design.control_block_steps].mean(dim=0)
        movement = self.design.movement_penalty_m3 * torch.mean(torch.square(first - active_target))
        return smooth_tfv + pfv_penalty + movement, hard_tfv, hard_pfv, smooth_tfv, smooth_pfv

    def optimize(
        self,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        fallback_settings: torch.Tensor,
        *,
        current_settings: torch.Tensor | None = None,
        previous_requested_settings: torch.Tensor | None = None,
        previous_actuator_flow: torch.Tensor | None = None,
        max_delta_per_update: float | torch.Tensor | None = None,
        **_: object,
    ) -> ContinuousMPCResultV127:
        del fallback_settings
        if current_settings is None or previous_requested_settings is None or previous_actuator_flow is None:
            raise ValueError("V127 MPC requires current/target/flow readback")
        if initial_state.ndim == 2:
            initial_state = initial_state[None]
        if initial_state.ndim != 3 or rainfall_scenarios.ndim != 4:
            raise ValueError("V127 initial state/rainfall shape mismatch")
        active = previous_requested_settings.reshape(-1).to(initial_state)
        current = current_settings.reshape(-1).to(initial_state)
        flow = previous_actuator_flow.reshape(1, -1).to(initial_state)
        if active.numel() != 109 or current.numel() != 109 or flow.shape[-1] != 109:
            raise ValueError("V127 actuator readback count mismatch")
        if max_delta_per_update is not None:
            runtime_delta = float(torch.as_tensor(max_delta_per_update).max())
            if runtime_delta > self.design.max_setting_delta_per_update + 1e-9:
                raise ValueError("V127 runtime max delta is looser than frozen contract")

        lo = torch.as_tensor(self.min_setting, dtype=initial_state.dtype, device=initial_state.device)
        hi = torch.as_tensor(self.max_setting, dtype=initial_state.dtype, device=initial_state.device)
        rbc = self._rbc_sequence(initial_state, current, active)
        with torch.no_grad():
            rbc_score, rbc_hard_tfv, rbc_hard_pfv, rbc_smooth_tfv, rbc_smooth_pfv = self._score(
                initial_state=initial_state,
                rainfall=rainfall_scenarios,
                sequence=rbc,
                flow=flow,
                active_target=active,
                rbc_smooth_pfv=None,
            )
        hold = active[None].expand(self.design.prediction_horizon_steps, -1).clone()
        starts = [
            encode_sequence_to_fraction_v127(
                hold.detach().cpu().numpy(),
                active_target=active.detach().cpu().numpy(),
                min_setting=self.min_setting,
                max_setting=self.max_setting,
                design=self.design,
            ),
            encode_sequence_to_fraction_v127(
                rbc.detach().cpu().numpy(),
                active_target=active.detach().cpu().numpy(),
                min_setting=self.min_setting,
                max_setting=self.max_setting,
                design=self.design,
            ),
        ]
        if self._last_fraction is not None and self._last_fraction.shape == starts[0].shape:
            starts.append(np.vstack((self._last_fraction[1:], self._last_fraction[-1:])))

        started = time.perf_counter()
        deadline = started + self.design.optimizer_deadline_seconds
        deadline_exceeded = False
        best: dict[str, Any] | None = None
        total_iterations = 0
        last_gradient_norm = float("nan")
        messages: list[str] = []

        def require_time() -> None:
            nonlocal deadline_exceeded
            if time.perf_counter() >= deadline:
                deadline_exceeded = True
                raise TimeoutError("V127 continuous optimizer deadline exceeded")

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal last_gradient_norm
            require_time()
            fractions = torch.as_tensor(
                flat.reshape(self.design.free_control_blocks, 109),
                dtype=initial_state.dtype,
                device=initial_state.device,
            ).requires_grad_(True)
            sequence = decode_fractional_targets_v127(
                fractions,
                active_target=active,
                min_setting=lo,
                max_setting=hi,
                design=self.design,
            )
            score, _, _, _, _ = self._score(
                initial_state=initial_state,
                rainfall=rainfall_scenarios,
                sequence=sequence,
                flow=flow,
                active_target=active,
                rbc_smooth_pfv=rbc_smooth_pfv.detach(),
            )
            require_time()
            if not bool(torch.isfinite(score)):
                raise FloatingPointError("V127 smooth MPC objective became non-finite")
            gradient = torch.autograd.grad(score, fractions)[0]
            require_time()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("V127 smooth MPC gradient became non-finite")
            last_gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
            return float(score.detach()), gradient.detach().cpu().numpy().astype(np.float64).reshape(-1)

        for start_index, start in enumerate(starts):
            if time.perf_counter() >= deadline:
                deadline_exceeded = True
                messages.append(f"start{start_index}:deadline-before-start")
                break
            try:
                result = minimize(
                    objective,
                    np.clip(start, 0.0, 1.0).reshape(-1),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0.0, 1.0)] * self.design.variable_count,
                    options={
                        "maxiter": int(self.design.lbfgsb_maxiter),
                        "ftol": float(self.design.lbfgsb_ftol),
                        "gtol": float(self.design.lbfgsb_gtol),
                        "maxls": 20,
                    },
                )
                total_iterations += int(getattr(result, "nit", 0))
                messages.append(f"start{start_index}:{result.message}")
                require_time()
                fraction = np.clip(
                    np.asarray(result.x).reshape(self.design.free_control_blocks, 109),
                    0.0,
                    1.0,
                )
                with torch.no_grad():
                    sequence = decode_fractional_targets_v127(
                        torch.as_tensor(
                            fraction, dtype=initial_state.dtype, device=initial_state.device
                        ),
                        active_target=active,
                        min_setting=lo,
                        max_setting=hi,
                        design=self.design,
                    )
                    score, hard_tfv, hard_pfv, smooth_tfv, smooth_pfv = self._score(
                        initial_state=initial_state,
                        rainfall=rainfall_scenarios,
                        sequence=sequence,
                        flow=flow,
                        active_target=active,
                        rbc_smooth_pfv=rbc_smooth_pfv,
                    )
                record = {
                    "score": float(score),
                    "hard_tfv": float(hard_tfv),
                    "hard_pfv": float(hard_pfv),
                    "smooth_tfv": float(smooth_tfv),
                    "smooth_pfv": float(smooth_pfv),
                    "sequence": sequence,
                    "fraction": fraction,
                    "success": bool(result.success),
                    "message": str(result.message),
                    "start_index": start_index,
                }
                if best is None or float(record["score"]) < float(best["score"]):
                    best = record
            except TimeoutError as exc:
                deadline_exceeded = True
                messages.append(f"start{start_index}:TimeoutError:{exc}")
                break
            except Exception as exc:
                messages.append(f"start{start_index}:{type(exc).__name__}:{str(exc)[:180]}")

        use_continuous = bool(
            not deadline_exceeded
            and best is not None
            and bool(best["success"])
            and math.isfinite(float(best["hard_tfv"]))
            and float(best["hard_tfv"])
            < float(rbc_hard_tfv) - self.design.min_improvement_vs_rbc_m3
            and float(best["score"]) < float(rbc_score)
        )
        if use_continuous:
            assert best is not None
            selected = best
            source = "continuous_lbfgsb"
            self._last_fraction = np.asarray(best["fraction"], dtype=np.float64)
        else:
            selected = {
                "score": float(rbc_score),
                "hard_tfv": float(rbc_hard_tfv),
                "hard_pfv": float(rbc_hard_pfv),
                "smooth_tfv": float(rbc_smooth_tfv),
                "smooth_pfv": float(rbc_smooth_pfv),
                "sequence": rbc,
                "fraction": starts[1],
                "success": False,
                "message": "RBC safety fallback",
                "start_index": 1,
            }
            source = "rbc_safety_fallback"
            self._last_fraction = starts[1].copy()

        sequence = selected["sequence"]
        if not bool(torch.allclose(sequence[0], sequence[1], rtol=0.0, atol=1e-7)):
            raise RuntimeError("V127 selected first 10-min target is not constant")
        elapsed = float(time.perf_counter() - started)
        return ContinuousMPCResultV127(
            settings=sequence.detach(),
            candidate_valid=True,
            predicted_delta_tfv_m3=float(selected["hard_tfv"] - float(rbc_hard_tfv)),
            selected_group_score_m3=float(selected["score"]),
            selected_candidate_index=int(selected["start_index"]),
            candidate_count=len(starts),
            first_move_group_count=len(starts),
            scenario_count=int(rainfall_scenarios.shape[0]),
            tail_only_noop_candidate_count=0,
            optimisation_steps=total_iterations,
            selected_source=source,
            continuous_optimizer_success=use_continuous,
            continuous_objective_m3=float(selected["score"]),
            continuous_tfv_risk_m3=float(selected["hard_tfv"]),
            continuous_pfv_risk_m3=float(selected["hard_pfv"]),
            continuous_optimization_tfv_m3=float(selected["smooth_tfv"]),
            continuous_optimization_pfv_m3=float(selected["smooth_pfv"]),
            rbc_objective_m3=float(rbc_score),
            rbc_tfv_risk_m3=float(rbc_hard_tfv),
            rbc_pfv_risk_m3=float(rbc_hard_pfv),
            rbc_optimization_tfv_m3=float(rbc_smooth_tfv),
            rbc_optimization_pfv_m3=float(rbc_smooth_pfv),
            gradient_norm=float(last_gradient_norm),
            scipy_message=" | ".join(messages)[:2000],
            variable_count=self.design.variable_count,
            optimizer_elapsed_seconds=elapsed,
            deadline_exceeded=deadline_exceeded,
        )


__all__ = [
    "ContinuousMPCDesignV127",
    "ContinuousMPCResultV127",
    "DifferentiableRollingMPCV127",
    "Step2GradientEvidenceV127",
    "V127_STEP3_CONTRACT",
    "decode_fractional_targets_v127",
    "encode_sequence_to_fraction_v127",
]
