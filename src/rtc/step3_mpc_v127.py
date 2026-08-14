"""Project7 V127 continuous differentiable rolling MPC.

RBC provides safety; the differentiable surrogate provides optimisation.  Sparse-RBC is
used only as one warm start and as the fail-safe sequence if the continuous solve is
invalid or cannot beat the safety seed by the frozen predicted-improvement margin.  It
is never the Step2 Value reference and never limits the continuous action space.

The free decision vector contains 12 x 109 continuous target fractions (H120 at 10-min
cadence).  A differentiable sequential transform maps fractions in [0,1] to exact
per-actuator bounds and max +/-0.5 target changes.  The final free target is held through
the remaining H360 prediction horizon.  SciPy L-BFGS-B receives objective gradients from
PyTorch autograd, matching the optimisation pattern established in recent differentiable
urban-drainage MPC research while preserving Project7's target-latch execution contract.
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

V127_STEP3_CONTRACT = "PROJECT7_V127_109ACT_H120_LBFGSB_RECEDING_HORIZON_MPC_V1"


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
        if self.holdout_rank < 0.70:
            failures.append(f"rank={self.holdout_rank:.4f}<0.70")
        if self.holdout_top1 < 0.50:
            failures.append(f"top1={self.holdout_top1:.4f}<0.50")
        if self.d2_gradient_sign_accuracy < 0.70:
            failures.append(f"D2 gradient sign={self.d2_gradient_sign_accuracy:.4f}<0.70")
        if self.d2_gradient_cosine_similarity < 0.60:
            failures.append(f"D2 gradient cosine={self.d2_gradient_cosine_similarity:.4f}<0.60")
        if self.d5_gradient_sign_accuracy < 0.70:
            failures.append(f"D5 gradient sign={self.d5_gradient_sign_accuracy:.4f}<0.70")
        if self.d5_gradient_cosine_similarity < 0.60:
            failures.append(f"D5 gradient cosine={self.d5_gradient_cosine_similarity:.4f}<0.60")
        if not self.causal_step1_state_verified:
            failures.append("causal Step1 state store not verified")
        if not self.causal_rainfall_verified:
            failures.append("causal rainfall forecast not verified")
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

    @property
    def variable_count_per_actuator(self) -> int:
        return self.free_control_blocks

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V127 MPC requires 300-s model / 600-s control cadence")
        if self.prediction_horizon_steps != 72 or self.free_control_blocks != 12:
            raise ValueError("V127 MPC requires H360 prediction / H120 free control")
        if not 0.0 < self.max_setting_delta_per_update <= 0.5:
            raise ValueError("V127 MPC max target movement must lie in (0,0.5]")
        if self.lbfgsb_maxiter <= 0 or self.lbfgsb_ftol <= 0 or self.lbfgsb_gtol <= 0:
            raise ValueError("V127 L-BFGS-B tolerances are invalid")
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("V127 CVaR alpha must lie in [0,1)")
        if self.pfv_soft_margin_m3 < 0 or self.pfv_penalty_weight < 0:
            raise ValueError("V127 PFV soft protection is invalid")
        if self.movement_penalty_m3 < 0 or self.min_improvement_vs_rbc_m3 < 0:
            raise ValueError("V127 regularisation/improvement margins must be non-negative")


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
    rbc_objective_m3: float
    rbc_tfv_risk_m3: float
    rbc_pfv_risk_m3: float
    gradient_norm: float
    scipy_message: str
    variable_count: int
    policy_mode: str = "continuous_differentiable_mpc"
    policy_mode_contract: str = V127_STEP3_CONTRACT


def _upper_tail_cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
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
    """Map [B,A] fractions to exact sequentially feasible H72 targets."""
    design.validate()
    if fractions.ndim != 2:
        raise ValueError("V127 fractions must be [free_block,actuator]")
    blocks, actuators = fractions.shape
    if blocks != design.free_control_blocks:
        raise ValueError("V127 free fraction horizon mismatch")
    active = active_target.reshape(-1).to(fractions)
    lo = min_setting.reshape(-1).to(fractions)
    hi = max_setting.reshape(-1).to(fractions)
    if active.numel() != actuators or lo.numel() != actuators or hi.numel() != actuators:
        raise ValueError("V127 target bounds/active latch actuator mismatch")
    if bool(torch.any(lo > hi)) or bool(torch.any((fractions < 0.0) | (fractions > 1.0))):
        raise ValueError("V127 fractional targets or physical bounds are invalid")
    previous = active
    free: list[torch.Tensor] = []
    delta = float(design.max_setting_delta_per_update)
    for k in range(blocks):
        lower = torch.maximum(lo, previous - delta)
        upper = torch.minimum(hi, previous + delta)
        target = lower + fractions[k] * (upper - lower)
        free.append(target)
        previous = target
    control = torch.stack(free, dim=0)
    all_blocks = design.prediction_horizon_steps // design.control_block_steps
    if all_blocks < blocks:
        raise RuntimeError("V127 prediction horizon shorter than free control horizon")
    if all_blocks > blocks:
        control = torch.cat((control, control[-1:].expand(all_blocks - blocks, -1)), dim=0)
    return control.repeat_interleave(design.control_block_steps, dim=0)


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
    blocks = values[:: design.control_block_steps][: design.free_control_blocks]
    previous = np.asarray(active_target, dtype=np.float64).reshape(-1)
    lo = np.asarray(min_setting, dtype=np.float64).reshape(-1)
    hi = np.asarray(max_setting, dtype=np.float64).reshape(-1)
    result: list[np.ndarray] = []
    for target in blocks:
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
            raise ValueError("V127 continuous MPC requires the frozen 109-actuator graph")
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
        *,
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
        # Freeze H120 free-RBC warm start then hold its last target; this makes all starts
        # share the exact V127 optimisation horizon semantics.
        blocks = sequence[:: self.design.control_block_steps].clone()
        blocks[self.design.free_control_blocks :] = blocks[self.design.free_control_blocks - 1]
        return blocks.repeat_interleave(self.design.control_block_steps, dim=0)

    def _score_sequence(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        sequence: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        rbc_pfv_reference_m3: torch.Tensor | None,
        active_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scenarios = int(rainfall_scenarios.shape[0])
        state = initial_state.expand(scenarios, -1, -1)
        flow = previous_actuator_flow.expand(scenarios, -1)
        settings = sequence[None].expand(scenarios, -1, -1)
        physics = torch.as_tensor(self.graph.actuator_physics, dtype=state.dtype, device=state.device)
        static = torch.as_tensor(self.graph.static_node_features, dtype=state.dtype, device=state.device)
        up = torch.as_tensor(self.graph.actuator_upstream, dtype=torch.long, device=state.device)
        down = torch.as_tensor(self.graph.actuator_downstream, dtype=torch.long, device=state.device)
        edges = torch.as_tensor(self.graph.edge_index, dtype=torch.long, device=state.device)
        output = self.model.objective_rollout(
            initial_state=state,
            rainfall=rainfall_scenarios,
            settings=settings,
            previous_actuator_flow=flow,
            actuator_upstream=up,
            actuator_downstream=down,
            actuator_physics=physics,
            static_node_features=static,
            edge_index=edges,
            flood_rate_index=self.flood_rate_index,
            priority_indices=self.priority_indices,
            dt_seconds=float(self.design.model_step_seconds),
        )
        tfv_risk = _upper_tail_cvar(output.tfv_m3, self.design.cvar_alpha)
        pfv_risk = _upper_tail_cvar(output.pfv_m3, self.design.cvar_alpha)
        pfv_penalty = output.tfv_m3.new_zeros(())
        if rbc_pfv_reference_m3 is not None:
            pfv_penalty = float(self.design.pfv_penalty_weight) * torch.relu(
                pfv_risk - rbc_pfv_reference_m3 - float(self.design.pfv_soft_margin_m3)
            )
        first = sequence[: self.design.control_block_steps].mean(dim=0)
        movement = float(self.design.movement_penalty_m3) * torch.mean(
            torch.square(first - active_target)
        )
        return tfv_risk + pfv_penalty + movement, tfv_risk, pfv_risk

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
        if current_settings is None or previous_requested_settings is None or previous_actuator_flow is None:
            raise ValueError("V127 MPC requires current/target/flow readback")
        if initial_state.ndim == 2:
            initial_state = initial_state[None]
        if initial_state.ndim != 3 or rainfall_scenarios.ndim != 4:
            raise ValueError("V127 initial state/rainfall shape mismatch")
        active_target = previous_requested_settings.reshape(-1).to(initial_state)
        current = current_settings.reshape(-1).to(initial_state)
        flow = previous_actuator_flow.reshape(1, -1).to(initial_state)
        if active_target.numel() != 109 or current.numel() != 109 or flow.shape[-1] != 109:
            raise ValueError("V127 actuator readback count mismatch")
        if max_delta_per_update is not None:
            observed_delta = float(torch.as_tensor(max_delta_per_update).max())
            if observed_delta > self.design.max_setting_delta_per_update + 1e-9:
                raise ValueError("V127 runtime max delta is looser than frozen contract")

        lo = torch.as_tensor(self.min_setting, dtype=initial_state.dtype, device=initial_state.device)
        hi = torch.as_tensor(self.max_setting, dtype=initial_state.dtype, device=initial_state.device)
        rbc = self._rbc_sequence(
            initial_state=initial_state,
            current_settings=current,
            active_target=active_target,
        )
        with torch.no_grad():
            rbc_objective, rbc_tfv, rbc_pfv = self._score_sequence(
                initial_state=initial_state,
                rainfall_scenarios=rainfall_scenarios,
                sequence=rbc,
                previous_actuator_flow=flow,
                rbc_pfv_reference_m3=None,
                active_target=active_target,
            )
        rbc_pfv_reference = rbc_pfv.detach()

        hold = active_target[None].expand(self.design.prediction_horizon_steps, -1).clone()
        starts = [
            encode_sequence_to_fraction_v127(
                hold.detach().cpu().numpy(), active_target=active_target.detach().cpu().numpy(),
                min_setting=self.min_setting, max_setting=self.max_setting, design=self.design,
            ),
            encode_sequence_to_fraction_v127(
                rbc.detach().cpu().numpy(), active_target=active_target.detach().cpu().numpy(),
                min_setting=self.min_setting, max_setting=self.max_setting, design=self.design,
            ),
        ]
        if self._last_fraction is not None and self._last_fraction.shape == starts[0].shape:
            shifted = np.vstack((self._last_fraction[1:], self._last_fraction[-1:]))
            starts.append(np.clip(shifted, 0.0, 1.0))

        best: dict[str, object] | None = None
        total_iterations = 0
        last_gradient_norm = float("nan")
        messages: list[str] = []
        variable_count = self.design.free_control_blocks * 109

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal last_gradient_norm
            fractions = torch.as_tensor(
                flat.reshape(self.design.free_control_blocks, 109),
                dtype=initial_state.dtype,
                device=initial_state.device,
            ).requires_grad_(True)
            sequence = decode_fractional_targets_v127(
                fractions,
                active_target=active_target,
                min_setting=lo,
                max_setting=hi,
                design=self.design,
            )
            score, _, _ = self._score_sequence(
                initial_state=initial_state,
                rainfall_scenarios=rainfall_scenarios,
                sequence=sequence,
                previous_actuator_flow=flow,
                rbc_pfv_reference_m3=rbc_pfv_reference,
                active_target=active_target,
            )
            if not bool(torch.isfinite(score)):
                raise FloatingPointError("V127 differentiable objective became non-finite")
            gradient = torch.autograd.grad(score, fractions, create_graph=False)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("V127 differentiable objective gradient became non-finite")
            last_gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
            return float(score.detach()), gradient.detach().cpu().numpy().astype(np.float64).reshape(-1)

        for start_index, start in enumerate(starts):
            try:
                result = minimize(
                    objective,
                    np.asarray(start, dtype=np.float64).reshape(-1),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0.0, 1.0)] * variable_count,
                    options={
                        "maxiter": int(self.design.lbfgsb_maxiter),
                        "ftol": float(self.design.lbfgsb_ftol),
                        "gtol": float(self.design.lbfgsb_gtol),
                        "maxls": 20,
                    },
                )
                total_iterations += int(getattr(result, "nit", 0))
                messages.append(f"start{start_index}:{str(result.message)}")
                fraction = np.clip(np.asarray(result.x).reshape(self.design.free_control_blocks, 109), 0.0, 1.0)
                with torch.no_grad():
                    sequence = decode_fractional_targets_v127(
                        torch.as_tensor(fraction, dtype=initial_state.dtype, device=initial_state.device),
                        active_target=active_target, min_setting=lo, max_setting=hi, design=self.design,
                    )
                    score, tfv, pfv = self._score_sequence(
                        initial_state=initial_state,
                        rainfall_scenarios=rainfall_scenarios,
                        sequence=sequence,
                        previous_actuator_flow=flow,
                        rbc_pfv_reference_m3=rbc_pfv_reference,
                        active_target=active_target,
                    )
                record = {
                    "score": float(score), "tfv": float(tfv), "pfv": float(pfv),
                    "sequence": sequence.detach(), "fraction": fraction,
                    "success": bool(result.success), "message": str(result.message),
                    "start_index": start_index,
                }
                if best is None or float(record["score"]) < float(best["score"]):
                    best = record
            except Exception as exc:
                messages.append(f"start{start_index}:{type(exc).__name__}:{str(exc)[:200]}")

        use_continuous = bool(
            best is not None
            and math.isfinite(float(best["tfv"]))
            and float(best["tfv"]) < float(rbc_tfv) - float(self.design.min_improvement_vs_rbc_m3)
        )
        if use_continuous:
            assert best is not None
            selected = best["sequence"]
            self._last_fraction = np.asarray(best["fraction"], dtype=np.float64)
            source = "continuous_lbfgsb"
            objective_value = float(best["score"])
            tfv_value = float(best["tfv"])
            pfv_value = float(best["pfv"])
            success = bool(best["success"])
            start_index = int(best["start_index"])
        else:
            selected = rbc.detach()
            self._last_fraction = starts[1].copy()
            source = "rbc_safety_fallback"
            objective_value = float(rbc_objective)
            tfv_value = float(rbc_tfv)
            pfv_value = float(rbc_pfv)
            success = False
            start_index = 1

        first = selected[: self.design.control_block_steps]
        if not bool(torch.allclose(first, first[0:1].expand_as(first), rtol=0.0, atol=1e-7)):
            raise RuntimeError("V127 selected first 10-min target is not constant")
        predicted_delta = tfv_value - float(rbc_tfv)
        return ContinuousMPCResultV127(
            settings=selected,
            candidate_valid=True,
            predicted_delta_tfv_m3=float(predicted_delta),
            selected_group_score_m3=float(objective_value),
            selected_candidate_index=start_index,
            candidate_count=len(starts) + (1 if best is not None else 0),
            first_move_group_count=len(starts) + (1 if best is not None else 0),
            scenario_count=int(rainfall_scenarios.shape[0]),
            tail_only_noop_candidate_count=0,
            optimisation_steps=total_iterations,
            selected_source=source,
            continuous_optimizer_success=bool(use_continuous and success),
            continuous_objective_m3=float(objective_value),
            continuous_tfv_risk_m3=float(tfv_value),
            continuous_pfv_risk_m3=float(pfv_value),
            rbc_objective_m3=float(rbc_objective),
            rbc_tfv_risk_m3=float(rbc_tfv),
            rbc_pfv_risk_m3=float(rbc_pfv),
            gradient_norm=float(last_gradient_norm),
            scipy_message=" | ".join(messages)[:2000],
            variable_count=variable_count,
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
