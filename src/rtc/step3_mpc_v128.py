"""Project7 V128 engineering-envelope-aware continuous rolling MPC.

V128 retains the V127 scientific objective and 12x109 L-BFGS-B receding horizon, but
makes the physical target decoder explicitly per-actuator.  Bounds/rates are applied
*before* surrogate scoring, so runtime never needs to project a scored action.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from .engineering_v128 import V128EngineeringEnvelope
from .step2_differentiable_v128 import TypedActuatorMessageSurrogateV128
from .step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123
from .step3_mpc_v127 import Step2GradientEvidenceV127, _cvar

V128_STEP3_CONTRACT = (
    "PROJECT7_V128_109ACT_H120_PER_ACTUATOR_ENVELOPE_LBFGSB_MPC_V1_ANYTIME"
)


@dataclass(frozen=True)
class ContinuousMPCDesignV128:
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    free_control_blocks: int = 12
    lbfgsb_maxiter: int = 30
    lbfgsb_ftol: float = 1.0e-7
    lbfgsb_gtol: float = 1.0e-5
    optimizer_deadline_seconds: float = 480.0
    cvar_alpha: float = 0.90
    pfv_soft_margin_m3: float = 100.0
    pfv_penalty_weight: float = 1.0
    movement_penalty_m3: float = 0.0
    min_improvement_vs_rbc_m3: float = 0.0

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    @property
    def variable_count(self) -> int:
        return self.free_control_blocks * 109

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V128 MPC requires 300-s model / 600-s control")
        if (self.prediction_horizon_steps, self.free_control_blocks) != (72, 12):
            raise ValueError("V128 MPC requires H360 prediction / H120 free control")
        if self.lbfgsb_maxiter <= 0 or self.lbfgsb_ftol <= 0 or self.lbfgsb_gtol <= 0:
            raise ValueError("V128 L-BFGS-B parameters are invalid")
        if not 0.0 < self.optimizer_deadline_seconds < self.control_update_seconds:
            raise ValueError("V128 optimizer deadline must lie inside the control period")
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("V128 CVaR alpha must lie in [0,1)")
        if min(
            self.pfv_soft_margin_m3,
            self.pfv_penalty_weight,
            self.movement_penalty_m3,
            self.min_improvement_vs_rbc_m3,
        ) < 0.0:
            raise ValueError("V128 objective/regularization parameters must be non-negative")


@dataclass(frozen=True)
class ContinuousMPCResultV128:
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
    engineering_envelope_sha256: str
    policy_mode: str = "continuous_differentiable_mpc_v128"
    policy_mode_contract: str = V128_STEP3_CONTRACT


def _vectors(
    envelope: V128EngineeringEnvelope,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lo = torch.as_tensor(envelope.min_setting, dtype=like.dtype, device=like.device)
    hi = torch.as_tensor(envelope.max_setting, dtype=like.dtype, device=like.device)
    delta = torch.as_tensor(
        envelope.max_delta_per_10min, dtype=like.dtype, device=like.device
    )
    return lo, hi, delta


def decode_fractional_targets_v128(
    fractions: torch.Tensor,
    *,
    active_target: torch.Tensor,
    envelope: V128EngineeringEnvelope,
    design: ContinuousMPCDesignV128,
) -> torch.Tensor:
    design.validate()
    envelope.validate()
    if fractions.ndim != 2 or fractions.shape != (
        design.free_control_blocks,
        len(envelope.actuator_ids),
    ):
        raise ValueError("V128 fractions must be [12, actuator]")
    if not bool(torch.isfinite(fractions).all()) or bool(
        torch.any((fractions < 0.0) | (fractions > 1.0))
    ):
        raise ValueError("V128 fractions must be finite in [0,1]")
    active = active_target.reshape(-1).to(fractions)
    if active.numel() != len(envelope.actuator_ids):
        raise ValueError("V128 active target count differs from engineering envelope")
    lo, hi, delta = _vectors(envelope, fractions)
    if bool(torch.any(active < lo - 1.0e-6) or torch.any(active > hi + 1.0e-6)):
        raise ValueError("V128 active supervisory target is outside frozen engineering bounds")

    previous = active
    free: list[torch.Tensor] = []
    for k in range(design.free_control_blocks):
        lower = torch.maximum(lo, previous - delta)
        upper = torch.minimum(hi, previous + delta)
        target = lower + fractions[k] * (upper - lower)
        free.append(target)
        previous = target
    blocks = torch.stack(free)
    total_blocks = design.prediction_horizon_steps // design.control_block_steps
    if total_blocks > design.free_control_blocks:
        blocks = torch.cat(
            (
                blocks,
                blocks[-1:].expand(total_blocks - design.free_control_blocks, -1),
            ),
            dim=0,
        )
    return blocks.repeat_interleave(design.control_block_steps, dim=0)


def encode_sequence_to_fraction_v128(
    sequence: np.ndarray,
    *,
    active_target: np.ndarray,
    envelope: V128EngineeringEnvelope,
    design: ContinuousMPCDesignV128,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    n = len(envelope.actuator_ids)
    if values.shape != (design.prediction_horizon_steps, n):
        raise ValueError("V128 seed sequence shape mismatch")
    previous = np.asarray(active_target, dtype=np.float64).reshape(-1)
    lo = np.asarray(envelope.min_setting, dtype=np.float64).reshape(-1)
    hi = np.asarray(envelope.max_setting, dtype=np.float64).reshape(-1)
    delta = np.asarray(envelope.max_delta_per_10min, dtype=np.float64).reshape(-1)
    targets = values[:: design.control_block_steps][: design.free_control_blocks]
    result: list[np.ndarray] = []
    for target in targets:
        lower = np.maximum(lo, previous - delta)
        upper = np.minimum(hi, previous + delta)
        clipped = np.clip(target, lower, upper)
        width = upper - lower
        fraction = np.where(width > 1.0e-12, (clipped - lower) / width, 0.5)
        result.append(np.clip(fraction, 0.0, 1.0))
        previous = clipped
    return np.stack(result).astype(np.float64)


class DifferentiableRollingMPCV128:
    accepts_previous_requested_settings = True
    policy_mode = "continuous_differentiable_mpc_v128"
    policy_mode_contract = V128_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: TypedActuatorMessageSurrogateV128,
        graph: Any,
        priority_indices: torch.Tensor,
        evidence: Step2GradientEvidenceV127,
        flood_rate_index: int,
        engineering_envelope: V128EngineeringEnvelope,
        design: ContinuousMPCDesignV128 = ContinuousMPCDesignV128(),
    ) -> None:
        evidence.validate()
        design.validate()
        engineering_envelope.assert_graph_order(graph)
        if len(graph.actuator_ids) != 109:
            raise ValueError("V128 continuous MPC requires exactly 109 actuators")
        self.model = model
        self.graph = graph
        self.priority_indices = priority_indices.long()
        self.evidence = evidence
        self.flood_rate_index = int(flood_rate_index)
        self.engineering_envelope = engineering_envelope
        self.design = design
        self._last_free_targets: np.ndarray | None = None

    def _rbc_sequence(
        self,
        initial_state: torch.Tensor,
        current_settings: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        n = len(self.engineering_envelope.actuator_ids)
        fallback = active_target.reshape(1, -1).expand(
            self.design.prediction_horizon_steps, -1
        )
        # Generate hydraulic feedback without pretending the physical current setting is
        # the supervisory rate anchor. Then project the *target* once to the same envelope
        # used by MPC, before the RBC sequence is scored.
        raw = build_sparse_state_auto_rbc_anchor_v123(
            initial_state[0].detach().cpu().numpy(),
            current_settings.detach().cpu().numpy(),
            fallback.detach().cpu().numpy(),
            self.graph,
            control_block_steps=self.design.control_block_steps,
            max_delta_per_update=1.0,
        )
        raw_target = np.asarray(raw[0], dtype=np.float64).reshape(n)
        active = active_target.detach().cpu().numpy().astype(np.float64).reshape(n)
        lo = np.asarray(self.engineering_envelope.min_setting, dtype=np.float64)
        hi = np.asarray(self.engineering_envelope.max_setting, dtype=np.float64)
        delta = np.asarray(
            self.engineering_envelope.max_delta_per_10min, dtype=np.float64
        )
        lower = np.maximum(lo, active - delta)
        upper = np.minimum(hi, active + delta)
        target = np.clip(raw_target, lower, upper)
        sequence = np.repeat(
            target[None, :], self.design.prediction_horizon_steps, axis=0
        )
        return torch.as_tensor(
            sequence, dtype=initial_state.dtype, device=initial_state.device
        )

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
            edge_index=torch.as_tensor(
                self.graph.edge_index, dtype=torch.long, device=state.device
            ),
            flood_rate_index=self.flood_rate_index,
            priority_indices=self.priority_indices,
            dt_seconds=float(self.design.model_step_seconds),
        )
        hard_tfv = _cvar(output.tfv_m3, self.design.cvar_alpha)
        hard_pfv = _cvar(output.pfv_m3, self.design.cvar_alpha)
        smooth_tfv = _cvar(output.optimization_tfv_m3, self.design.cvar_alpha)
        smooth_pfv = _cvar(output.optimization_pfv_m3, self.design.cvar_alpha)
        pfv_penalty = smooth_tfv.new_zeros(())
        if rbc_smooth_pfv is not None and self.design.pfv_penalty_weight > 0.0:
            pfv_penalty = self.design.pfv_penalty_weight * torch.relu(
                smooth_pfv - rbc_smooth_pfv - self.design.pfv_soft_margin_m3
            )
        first = sequence[: self.design.control_block_steps].mean(dim=0)
        movement = self.design.movement_penalty_m3 * torch.mean(
            torch.square(first - active_target)
        )
        return (
            smooth_tfv + pfv_penalty + movement,
            hard_tfv,
            hard_pfv,
            smooth_tfv,
            smooth_pfv,
        )

    def _shifted_previous_start(self, active: torch.Tensor) -> np.ndarray | None:
        shape = (self.design.free_control_blocks, len(self.engineering_envelope.actuator_ids))
        if self._last_free_targets is None or self._last_free_targets.shape != shape:
            return None
        shifted = np.vstack((self._last_free_targets[1:], self._last_free_targets[-1:]))
        total_blocks = self.design.prediction_horizon_steps // self.design.control_block_steps
        full_blocks = np.vstack(
            (
                shifted,
                np.repeat(
                    shifted[-1:], total_blocks - self.design.free_control_blocks, axis=0
                ),
            )
        )
        sequence = np.repeat(full_blocks, self.design.control_block_steps, axis=0)
        return encode_sequence_to_fraction_v128(
            sequence,
            active_target=active.detach().cpu().numpy(),
            envelope=self.engineering_envelope,
            design=self.design,
        )

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
    ) -> ContinuousMPCResultV128:
        del fallback_settings
        if (
            current_settings is None
            or previous_requested_settings is None
            or previous_actuator_flow is None
        ):
            raise ValueError("V128 MPC requires current/target/flow readback")
        if initial_state.ndim == 2:
            initial_state = initial_state[None]
        if initial_state.ndim != 3 or rainfall_scenarios.ndim != 4:
            raise ValueError("V128 initial state/rainfall shape mismatch")
        active = previous_requested_settings.reshape(-1).to(initial_state)
        current = current_settings.reshape(-1).to(initial_state)
        flow = previous_actuator_flow.reshape(1, -1).to(initial_state)
        n = len(self.engineering_envelope.actuator_ids)
        if active.numel() != n or current.numel() != n or flow.shape[-1] != n:
            raise ValueError("V128 actuator readback count mismatch")
        if not bool(
            torch.isfinite(active).all()
            and torch.isfinite(current).all()
            and torch.isfinite(flow).all()
        ):
            raise ValueError("V128 actuator readback contains non-finite values")
        if max_delta_per_update is not None:
            runtime_delta = torch.as_tensor(
                max_delta_per_update, dtype=active.dtype, device=active.device
            )
            runtime_delta = torch.broadcast_to(runtime_delta, active.shape)
            expected_delta = torch.as_tensor(
                self.engineering_envelope.max_delta_per_10min,
                dtype=active.dtype,
                device=active.device,
            )
            if not bool(torch.allclose(runtime_delta, expected_delta, rtol=0.0, atol=1e-7)):
                raise ValueError("V128 runtime target-delta envelope differs from scored MPC envelope")

        rbc = self._rbc_sequence(initial_state, current, active)
        with torch.no_grad():
            (
                rbc_score,
                rbc_hard_tfv,
                rbc_hard_pfv,
                rbc_smooth_tfv,
                rbc_smooth_pfv,
            ) = self._score(
                initial_state=initial_state,
                rainfall=rainfall_scenarios,
                sequence=rbc,
                flow=flow,
                active_target=active,
                rbc_smooth_pfv=None,
            )
        hold = active[None].expand(self.design.prediction_horizon_steps, -1).clone()
        starts = [
            encode_sequence_to_fraction_v128(
                hold.detach().cpu().numpy(),
                active_target=active.detach().cpu().numpy(),
                envelope=self.engineering_envelope,
                design=self.design,
            ),
            encode_sequence_to_fraction_v128(
                rbc.detach().cpu().numpy(),
                active_target=active.detach().cpu().numpy(),
                envelope=self.engineering_envelope,
                design=self.design,
            ),
        ]
        shifted = self._shifted_previous_start(active)
        if shifted is not None:
            starts.append(shifted)

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
                raise TimeoutError("V128 continuous optimizer deadline exceeded")

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal last_gradient_norm
            require_time()
            fractions = torch.as_tensor(
                flat.reshape(self.design.free_control_blocks, n),
                dtype=initial_state.dtype,
                device=initial_state.device,
            ).requires_grad_(True)
            sequence = decode_fractional_targets_v128(
                fractions,
                active_target=active,
                envelope=self.engineering_envelope,
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
                raise FloatingPointError("V128 smooth MPC objective became non-finite")
            gradient = torch.autograd.grad(score, fractions)[0]
            require_time()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("V128 smooth MPC gradient became non-finite")
            last_gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
            return (
                float(score.detach()),
                gradient.detach().cpu().numpy().astype(np.float64).reshape(-1),
            )

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
                messages.append(
                    f"start{start_index}:success={bool(result.success)}:{result.message}"
                )
                require_time()
                fraction = np.clip(
                    np.asarray(result.x).reshape(self.design.free_control_blocks, n),
                    0.0,
                    1.0,
                )
                with torch.no_grad():
                    sequence = decode_fractional_targets_v128(
                        torch.as_tensor(
                            fraction,
                            dtype=initial_state.dtype,
                            device=initial_state.device,
                        ),
                        active_target=active,
                        envelope=self.engineering_envelope,
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
                values = [score, hard_tfv, hard_pfv, smooth_tfv, smooth_pfv]
                if not all(math.isfinite(float(value)) for value in values):
                    messages.append(f"start{start_index}:nonfinite-final-score")
                    continue
                record = {
                    "score": float(score),
                    "hard_tfv": float(hard_tfv),
                    "hard_pfv": float(hard_pfv),
                    "smooth_tfv": float(smooth_tfv),
                    "smooth_pfv": float(smooth_pfv),
                    "sequence": sequence,
                    "fraction": fraction,
                    "solver_success": bool(result.success),
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
                messages.append(
                    f"start{start_index}:{type(exc).__name__}:{str(exc)[:180]}"
                )

        numerical_eps = 1.0e-6
        use_continuous = bool(
            best is not None
            and math.isfinite(float(best["hard_tfv"]))
            and float(best["hard_tfv"])
            < float(rbc_hard_tfv)
            - max(self.design.min_improvement_vs_rbc_m3, numerical_eps)
            and float(best["score"]) <= float(rbc_score) + numerical_eps
        )
        if use_continuous:
            assert best is not None
            selected = best
            source = "continuous_lbfgsb_v128"
        else:
            selected = {
                "score": float(rbc_score),
                "hard_tfv": float(rbc_hard_tfv),
                "hard_pfv": float(rbc_hard_pfv),
                "smooth_tfv": float(rbc_smooth_tfv),
                "smooth_pfv": float(rbc_smooth_pfv),
                "sequence": rbc,
                "fraction": starts[1],
                "solver_success": False,
                "message": "RBC safety fallback",
                "start_index": 1,
            }
            source = "rbc_safety_fallback_v128"

        sequence = selected["sequence"]
        if not bool(torch.allclose(sequence[0], sequence[1], rtol=0.0, atol=1e-7)):
            raise RuntimeError("V128 selected first 10-min target is not constant")
        blocks = sequence[:: self.design.control_block_steps]
        self._last_free_targets = (
            blocks[: self.design.free_control_blocks]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        elapsed = float(time.perf_counter() - started)
        return ContinuousMPCResultV128(
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
            engineering_envelope_sha256=self.engineering_envelope.semantic_sha256,
        )


__all__ = [
    "ContinuousMPCDesignV128",
    "ContinuousMPCResultV128",
    "DifferentiableRollingMPCV128",
    "V128_STEP3_CONTRACT",
    "decode_fractional_targets_v128",
    "encode_sequence_to_fraction_v128",
]
