"""Research-grade rolling MPC for Project7 V12.2.

The V12.2 contract separates four concepts that earlier Project7 controllers mixed:

1. ``current_setting`` is a realised physical state and is an input to state estimation.
2. ``target_setting`` is the supervisory command latch.  Keeping the active target is
   the exact PASSIVE / no-new-command action.
3. The prediction horizon can be long enough to observe delayed flooding response,
   while only a shorter control horizon contains free future commands.  The remaining
   tail is held at the last planned target, so an unexecuted late tail cannot create a
   fictitious first-move benefit.
4. Baseline controllers are *comparators*, not the Proposed action space.  This module
   optimises joint multi-actuator actions on the frozen low-dimensional control
   manifold.  Baseline trajectories may be supplied as optional initial seeds by a
   higher-level caller, but they never cap the optimum.

Continuous optimisation is deliberately fail-closed behind a stronger Step2 evidence
contract.  A Value model that can merely rank a finite candidate bank is not assumed to
have trustworthy gradients between candidates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .step2_control_basis_v60 import ControlBasisV60
from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_policy_v120 import RuntimeNormalizationV120, candidate_coefficients_v120
from .step2_v120_contract import Step2V120Contract

V122_STEP3_CONTRACT = "PROJECT7_V122_PASSIVE_TARGET_LATCH_RECEDING_HORIZON_MPC_V1"


@dataclass(frozen=True)
class Step2GradientEvidenceV122:
    """Minimum evidence required before interpolating with surrogate gradients."""

    rank_correlation: float
    top1_rate: float
    gradient_sign_accuracy: float
    gradient_cosine_similarity: float
    passive_reference_semantics_verified: bool

    def validate(self) -> None:
        values = (
            self.rank_correlation,
            self.top1_rate,
            self.gradient_sign_accuracy,
            self.gradient_cosine_similarity,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("V122 Step2 gradient evidence contains non-finite values")
        reasons: list[str] = []
        if self.rank_correlation < 0.70:
            reasons.append(f"rank={self.rank_correlation:.4f}<0.70")
        if self.top1_rate < 0.50:
            reasons.append(f"top1={self.top1_rate:.4f}<0.50")
        if self.gradient_sign_accuracy < 0.70:
            reasons.append(f"gradient_sign={self.gradient_sign_accuracy:.4f}<0.70")
        if self.gradient_cosine_similarity < 0.60:
            reasons.append(f"gradient_cosine={self.gradient_cosine_similarity:.4f}<0.60")
        if not self.passive_reference_semantics_verified:
            reasons.append("passive-reference semantics not verified")
        if reasons:
            raise ValueError(
                "V122 continuous MPC is blocked by Step2 evidence: " + "; ".join(reasons)
            )


@dataclass(frozen=True)
class RollingMPCDesignV122:
    """Frozen temporal and optimisation design.

    The Step2 Value model retains a 360-min (72 x 5-min) prediction window.  Free
    supervisory targets are restricted to the first 120 min (12 x 10-min updates).
    Afterwards the final free target is held constant.  Only the first 10-min target
    is written to SWMM before the whole problem is solved again.
    """

    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    control_horizon_steps: int = 24
    max_setting_delta_per_update: float = 0.50
    seed_candidate_count: int = 25  # PASSIVE + one D3-style 24-candidate bank
    optimizer_steps: int = 40
    optimizer_learning_rate: float = 0.04
    coefficient_trust_radius: float = 0.25
    cvar_alpha: float = 0.90
    min_predicted_improvement_m3: float = 0.0
    first_move_round_decimals: int = 6

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    @property
    def control_horizon_blocks(self) -> int:
        return self.control_horizon_steps // self.control_block_steps

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V122 requires the frozen 300-s model / 600-s control clock")
        if self.control_update_seconds % self.model_step_seconds:
            raise ValueError("control update must be an integer multiple of model step")
        if self.prediction_horizon_steps != 72:
            raise ValueError("V122 Value prediction horizon is frozen at H72 / 360 min")
        if not 0 < self.control_horizon_steps <= self.prediction_horizon_steps:
            raise ValueError("invalid V122 control horizon")
        if self.control_horizon_steps % self.control_block_steps:
            raise ValueError("V122 control horizon must align to 10-min blocks")
        if not 0.0 < self.max_setting_delta_per_update <= 1.0:
            raise ValueError("invalid per-update target-setting bound")
        if self.seed_candidate_count != 25:
            raise ValueError("V122 keeps one distribution-matched D3 bank: HOLD + 24")
        if self.optimizer_steps <= 0 or self.optimizer_learning_rate <= 0:
            raise ValueError("invalid V122 continuous optimiser settings")
        if not 0.0 < self.coefficient_trust_radius <= 1.0:
            raise ValueError("invalid V122 coefficient trust radius")
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("invalid V122 CVaR alpha")
        if self.min_predicted_improvement_m3 < 0:
            raise ValueError("V122 minimum improvement cannot be negative")


@dataclass(frozen=True)
class RollingMPCResultV122:
    settings: torch.Tensor
    candidate_valid: bool
    predicted_delta_tfv_m3: float
    selected_group_score_m3: float
    selected_candidate_index: int
    candidate_count: int
    first_move_group_count: int
    scenario_count: int
    passive_selected: bool
    tail_only_noop_candidate_count: int
    optimisation_steps: int


def passive_target_sequence_v122(
    active_target_setting: torch.Tensor, horizon_steps: int
) -> torch.Tensor:
    """Return the no-new-supervisory-command reference sequence."""

    target = active_target_setting.reshape(-1)
    if horizon_steps <= 0 or not bool(torch.isfinite(target).all()):
        raise ValueError("invalid passive target sequence input")
    return target[None, :].expand(int(horizon_steps), -1).clone()


def _target_latch_project_v122(
    candidate_settings: torch.Tensor,
    *,
    active_target_setting: torch.Tensor,
    min_setting: torch.Tensor,
    max_setting: torch.Tensor,
    design: RollingMPCDesignV122,
) -> torch.Tensor:
    """Project planned *target commands*; realised current setting is not a command anchor."""

    design.validate()
    if candidate_settings.ndim != 3:
        raise ValueError("V122 candidate settings must be [C,H,A]")
    candidates, horizon, actuators = candidate_settings.shape
    if horizon != design.prediction_horizon_steps:
        raise ValueError("V122 candidate/prediction horizon mismatch")
    active = active_target_setting.reshape(-1).to(candidate_settings)
    lo = min_setting.reshape(-1).to(candidate_settings)
    hi = max_setting.reshape(-1).to(candidate_settings)
    if active.numel() != actuators or lo.numel() != actuators or hi.numel() != actuators:
        raise ValueError("V122 target-latch projection actuator mismatch")
    if bool(torch.any(lo > hi)) or not bool(torch.isfinite(active).all()):
        raise ValueError("invalid V122 actuator bounds/active target")

    block_steps = design.control_block_steps
    blocks = candidate_settings[:, ::block_steps, :]
    free_blocks = design.control_horizon_blocks
    if blocks.shape[1] * block_steps != horizon or free_blocks > blocks.shape[1]:
        raise ValueError("V122 block grid mismatch")

    delta = float(design.max_setting_delta_per_update)
    previous = active[None, :].expand(candidates, -1)
    projected: list[torch.Tensor] = []
    for block in range(free_blocks):
        desired = blocks[:, block, :]
        lower = torch.maximum(lo[None, :], previous - delta)
        upper = torch.minimum(hi[None, :], previous + delta)
        value = torch.maximum(torch.minimum(desired, upper), lower)
        projected.append(value)
        previous = value

    if not projected:
        raise RuntimeError("V122 free control horizon is empty")
    terminal = projected[-1]
    for _ in range(free_blocks, blocks.shape[1]):
        projected.append(terminal)
    result = torch.stack(projected, dim=1).repeat_interleave(block_steps, dim=1)
    if result.shape != candidate_settings.shape:
        raise RuntimeError("V122 target-latch projection changed tensor shape")
    return result


def _upper_tail_cvar_v122(values: torch.Tensor, alpha: float) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("V122 scenario values must be [S,C]")
    count = max(1, int(math.ceil((1.0 - float(alpha)) * values.shape[0])))
    return torch.topk(values, k=count, dim=0, largest=True).values.mean(dim=0)


def _first_move_keys_v122(
    settings: torch.Tensor,
    passive: torch.Tensor,
    *,
    design: RollingMPCDesignV122,
) -> tuple[list[bytes], np.ndarray, int]:
    block = design.control_block_steps
    first = settings[:, :block, :].mean(dim=1).detach().cpu().numpy()
    passive_first = passive[:block].mean(dim=0).detach().cpu().numpy()
    rounded = np.round(first, design.first_move_round_decimals)
    passive_rounded = np.round(passive_first, design.first_move_round_decimals)
    passive_mask = np.all(rounded == passive_rounded[None, :], axis=1)
    keys = [row.tobytes() for row in rounded]
    return keys, passive_mask, int(np.sum(passive_mask))


def _select_first_move_group_v122(
    settings: torch.Tensor,
    sequence_risk_m3: torch.Tensor,
    passive: torch.Tensor,
    *,
    design: RollingMPCDesignV122,
) -> tuple[int, float, int, int]:
    """Aggregate continuation uncertainty by executable first move.

    The passive first-move group is structurally assigned zero value.  Therefore a
    sequence whose first 10-min command is passive cannot win merely because its later
    (currently unexecuted) commands look beneficial.
    """

    keys, passive_mask, passive_count = _first_move_keys_v122(
        settings, passive, design=design
    )
    groups: dict[bytes, list[int]] = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)

    best_idx = 0
    best_score = 0.0
    found_nonpassive = False
    for indices in groups.values():
        if any(bool(passive_mask[i]) for i in indices):
            score = 0.0
            representative = next(i for i in indices if passive_mask[i])
        else:
            values = sequence_risk_m3[indices]
            score_tensor = values.median()
            score = float(score_tensor.detach())
            distance = torch.abs(values - score_tensor)
            representative = indices[int(torch.argmin(distance).item())]
        if score < best_score and score < -float(design.min_predicted_improvement_m3):
            best_idx = int(representative)
            best_score = float(score)
            found_nonpassive = True

    if not found_nonpassive:
        passive_indices = np.flatnonzero(passive_mask)
        best_idx = int(passive_indices[0]) if passive_indices.size else 0
        best_score = 0.0
    return best_idx, best_score, len(groups), passive_count


class DifferentiableRollingMPCV122:
    """Multi-start, trust-region, low-dimensional receding-horizon MPC.

    The optimisation variables are the frozen V6/V7 group x temporal-basis
    coefficients, not baseline-policy IDs.  Each D3-style seed is allowed to refine only
    actuator groups already active in that seed; this keeps interpolation inside the
    mechanism support demonstrated by D3 instead of turning a 24-candidate experiment
    into an unconstrained 109-actuator extrapolation.
    """

    accepts_previous_requested_settings = True

    def __init__(
        self,
        *,
        model: ControlValueSurrogateV70,
        basis: ControlBasisV60,
        prepared: PreparedStaticV60,
        normalization: RuntimeNormalizationV120,
        evidence: Step2GradientEvidenceV122,
        design: RollingMPCDesignV122 = RollingMPCDesignV122(),
    ) -> None:
        evidence.validate()
        design.validate()
        basis.validate()
        normalization.validate()
        if basis.horizon.horizon_steps != design.prediction_horizon_steps:
            raise ValueError("V122 basis/Value prediction horizon mismatch")
        if basis.horizon.control_block_steps != design.control_block_steps:
            raise ValueError("V122 basis/control cadence mismatch")
        self.model = model
        self.basis = basis
        self.prepared = prepared
        self.normalization = normalization
        self.evidence = evidence
        self.design = design
        seed_contract = Step2V120Contract(candidate_banks=1)
        seeds = candidate_coefficients_v120(basis, contract=seed_contract)
        if seeds.shape[0] != design.seed_candidate_count:
            raise RuntimeError("V122 D3 seed-bank size drift")
        self._seed_coefficients = seeds

    def _score(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        passive: torch.Tensor,
        coefficients: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target_setting: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scenarios = int(rainfall_scenarios.shape[0])
        reference_for_candidates = passive[None, None].expand(
            1, coefficients.shape[0], -1, -1
        )
        decoded = self.basis.decode(reference_for_candidates, coefficients[None])[0]
        settings = _target_latch_project_v122(
            decoded,
            active_target_setting=active_target_setting,
            min_setting=torch.as_tensor(
                self.basis.min_setting, dtype=decoded.dtype, device=decoded.device
            ),
            max_setting=torch.as_tensor(
                self.basis.max_setting, dtype=decoded.dtype, device=decoded.device
            ),
            design=self.design,
        )
        state = initial_state[None].expand(scenarios, -1, -1)
        reference = passive[None].expand(scenarios, -1, -1)
        candidate = settings[None].expand(scenarios, -1, -1, -1)
        flow = previous_actuator_flow.reshape(1, -1).expand(scenarios, -1)
        output = self.model(
            self.normalization.state(state),
            self.normalization.rainfall(rainfall_scenarios),
            reference,
            candidate,
            self.normalization.flow(flow),
            self.prepared,
        )
        risk = _upper_tail_cvar_v122(output.delta_tfv_m3, self.design.cvar_alpha)
        if not bool(torch.isfinite(risk).all()):
            raise RuntimeError("V122 Value produced non-finite TFV risk")
        return settings, risk

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
        **_controller_compatibility: object,
    ) -> RollingMPCResultV122:
        # ``fallback_settings`` remains in the protocol for controller compatibility, but
        # PASSIVE is defined from the active SWMM target latch, not realised current.
        del fallback_settings, current_settings
        if previous_requested_settings is None:
            raise ValueError("V122 requires active target_setting readback")
        active_target = previous_requested_settings.reshape(-1)
        if active_target.numel() != self.basis.grouping.actuator_count:
            raise ValueError("V122 active target actuator count mismatch")
        if initial_state.ndim == 3 and initial_state.shape[0] == 1:
            initial_state = initial_state[0]
        if initial_state.ndim != 2:
            raise ValueError("V122 initial state must resolve to [N,D]")
        if rainfall_scenarios.ndim != 4:
            raise ValueError("V122 rainfall scenarios must be [S,H,N,R]")
        if rainfall_scenarios.shape[1] != self.design.prediction_horizon_steps:
            raise ValueError("V122 rainfall/prediction horizon mismatch")
        if previous_actuator_flow is None:
            previous_actuator_flow = torch.zeros_like(active_target)
        previous_actuator_flow = previous_actuator_flow.reshape(-1)
        if previous_actuator_flow.numel() != active_target.numel():
            raise ValueError("V122 actuator-flow vector mismatch")
        if max_delta_per_update is not None:
            raw = torch.as_tensor(max_delta_per_update, dtype=torch.float32).reshape(-1)
            if not bool(torch.isfinite(raw).all()):
                raise ValueError("V122 runtime max delta is non-finite")
            if float(raw.max()) > self.design.max_setting_delta_per_update + 1e-9:
                raise ValueError("runtime target-setting bound is looser than V122 contract")

        passive = passive_target_sequence_v122(
            active_target, self.design.prediction_horizon_steps
        ).to(initial_state)
        seed = torch.as_tensor(
            self._seed_coefficients,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        group_mask = torch.any(torch.abs(seed) > 1.0e-8, dim=1)
        coordinate_mask = group_mask[:, None, :].expand_as(seed)

        # Preserve an untouched seed bank so optimisation can never be worse than its
        # distribution-matched starting points.  Candidate zero stays exact PASSIVE.
        with torch.no_grad():
            seed_settings, seed_risk = self._score(
                initial_state=initial_state,
                rainfall_scenarios=rainfall_scenarios,
                passive=passive,
                coefficients=seed,
                previous_actuator_flow=previous_actuator_flow,
                active_target_setting=active_target,
            )

        coeff = seed.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([coeff], lr=self.design.optimizer_learning_rate)
        radius = float(self.design.coefficient_trust_radius)
        for _ in range(self.design.optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            _, risk = self._score(
                initial_state=initial_state,
                rainfall_scenarios=rainfall_scenarios,
                passive=passive,
                coefficients=coeff,
                previous_actuator_flow=previous_actuator_flow,
                active_target_setting=active_target,
            )
            # Passive has no active coordinates.  Optimise all supported non-passive
            # starts together; hard projection below enforces the trust region.
            loss = risk[1:].mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V122 optimisation loss is non-finite")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                lower = torch.clamp(seed - radius, -1.0, 1.0)
                upper = torch.clamp(seed + radius, -1.0, 1.0)
                coeff.copy_(torch.maximum(torch.minimum(coeff, upper), lower))
                coeff.mul_(coordinate_mask)
                coeff[0].zero_()

        with torch.no_grad():
            opt_settings, opt_risk = self._score(
                initial_state=initial_state,
                rainfall_scenarios=rainfall_scenarios,
                passive=passive,
                coefficients=coeff.detach(),
                previous_actuator_flow=previous_actuator_flow,
                active_target_setting=active_target,
            )
            all_settings = torch.cat((seed_settings, opt_settings[1:]), dim=0)
            all_risk = torch.cat((seed_risk, opt_risk[1:]), dim=0)
            selected, group_score, group_count, passive_like = _select_first_move_group_v122(
                all_settings, all_risk, passive, design=self.design
            )
            block = self.design.control_block_steps
            passive_first = passive[:block].mean(dim=0)
            selected_first = all_settings[selected, :block].mean(dim=0)
            passive_selected = bool(
                torch.allclose(selected_first, passive_first, rtol=0.0, atol=10.0 ** -self.design.first_move_round_decimals)
            )
            valid = bool(
                not passive_selected
                and group_score < -float(self.design.min_predicted_improvement_m3)
            )
            if not valid:
                # Seed zero is exact passive by construction.
                selected = 0
                group_score = 0.0
                passive_selected = True

        return RollingMPCResultV122(
            settings=all_settings[selected].detach(),
            candidate_valid=valid,
            predicted_delta_tfv_m3=float(group_score),
            selected_group_score_m3=float(group_score),
            selected_candidate_index=int(selected),
            candidate_count=int(all_settings.shape[0]),
            first_move_group_count=int(group_count),
            scenario_count=int(rainfall_scenarios.shape[0]),
            passive_selected=passive_selected,
            tail_only_noop_candidate_count=max(int(passive_like) - 1, 0),
            optimisation_steps=int(self.design.optimizer_steps),
        )


__all__ = [
    "DifferentiableRollingMPCV122",
    "RollingMPCDesignV122",
    "RollingMPCResultV122",
    "Step2GradientEvidenceV122",
    "V122_STEP3_CONTRACT",
    "passive_target_sequence_v122",
]
