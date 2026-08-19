"""Hybrid Practical H10 proposal portfolio for Project7 RTC.

The historical Project7 optimizer searched a 12 x 109 control plan with L-BFGS-B. Development
replays showed that such high-dimensional optimization could exploit surrogate extrema outside the
reliable training-support geometry. The current paper method keeps the finite H10 probe/hydraulic
candidates, but adds exactly one *first-action-only* differentiable proposal:

    SUPPORT_CONSTRAINED_GRADIENT_H10

Only the current 109-dimensional H10 supervisory target is optimized. Every projected-gradient trial
is immediately projected to the frozen first-move q95 radius, 0.5 target slew, actuator bounds and
q95 changed-facility ceiling. The differentiable objective is the frozen base-Step2 H10 action-effect
score with the exact ``H10 candidate -> H350 HOLD`` token. The projected-gradient proposal is only a
candidate proposer: the separately trained receding-policy-return critic remains the final ranker and
one-sided admission gate.

No future realised rainfall, online SWMM candidate search, PFV/peak objective, rule-baseline warm
start or action penalty is introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .direct_tfv_policy_return import encode_policy_return_action_token
from .direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    LearnedH10ProbeProposal,
    PolicyReturnPortfolioCandidate,
    _bounded_supported_target,
    _normalization_tensors,
    build_learned_h10_probe_proposal,
    build_policy_return_candidate_portfolio,
    score_h10_first_action_targets,
)


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_CAUSAL_CANDIDATE_PORTFOLIO_V4_H10_HYBRID_GRADIENT"
)
DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT = (
    "PROJECT7_DIRECT_TFV_SUPPORTED_109D_H10_PROJECTED_GRADIENT_GENERATOR_V1"
)
PROJECTED_GRADIENT_SOURCE = "SUPPORT_CONSTRAINED_GRADIENT_H10"


@dataclass(frozen=True)
class ProjectedGradientH10Proposal:
    target: torch.Tensor | None
    start_score_m3: float
    best_score_m3: float
    attempted_steps: int
    accepted_improvement_steps: int
    final_gradient_l2: float
    produced_nonhold_candidate: bool
    generator_contract: str = DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT


@dataclass(frozen=True)
class HybridPolicyReturnPortfolio:
    candidates: tuple[PolicyReturnPortfolioCandidate, ...]
    learned_probe: LearnedH10ProbeProposal
    projected_gradient: ProjectedGradientH10Proposal


def _score_single_h10_target_with_grad(
    *,
    model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    candidate_target: torch.Tensor,
) -> torch.Tensor:
    """Differentiable frozen-Step2 mean score for one H10 target under causal rain scenarios."""
    if current_state.ndim != 3 or int(current_state.shape[0]) != 1:
        raise ValueError("projected-gradient scorer expects current_state [1,node,state]")
    if rainfall_scenarios.ndim != 4 or int(rainfall_scenarios.shape[0]) < 1:
        raise ValueError("projected-gradient scorer expects rainfall [scenario,H,node,feature]")
    if previous_actuator_flow.shape != (1, 109):
        raise ValueError("projected-gradient scorer expects previous flow [1,109]")
    if active_target.shape != (109,) or candidate_target.shape != (109,):
        raise ValueError("projected-gradient scorer expects 109-dimensional targets")

    device = current_state.device
    dtype = current_state.dtype
    scenarios, horizon, nodes, rain_features = rainfall_scenarios.shape
    norm = _normalization_tensors(normalization, dtype=dtype, device=device)
    state = ((current_state - norm["state_mean"]) / norm["state_std"]).expand(
        scenarios, -1, -1
    )
    rainfall = (rainfall_scenarios - norm["rain_mean"]) / norm["rain_std"]
    if tuple(rainfall.shape[1:]) != (horizon, nodes, rain_features):
        raise ValueError("projected-gradient rainfall geometry changed unexpectedly")
    flow = ((previous_actuator_flow - norm["flow_mean"]) / norm["flow_std"]).expand(
        scenarios, -1
    )
    active = active_target.reshape(1, 109).expand(scenarios, -1)
    target = candidate_target.reshape(1, 109).expand(scenarios, -1)
    reference, candidate = encode_policy_return_action_token(
        active,
        target,
        horizon_steps=int(horizon),
        first_action_steps=2,
    )
    output = model(
        current_state=state,
        rainfall=rainfall,
        reference_settings=reference,
        candidate_settings=candidate,
        previous_actuator_flow=flow,
        actuator_upstream=torch.as_tensor(
            graph.actuator_upstream, dtype=torch.long, device=device
        ),
        actuator_downstream=torch.as_tensor(
            graph.actuator_downstream, dtype=torch.long, device=device
        ),
        actuator_physics=torch.as_tensor(
            graph.actuator_physics, dtype=dtype, device=device
        ),
    )
    score = output.total_delta_tfv_m3.mean()
    if not bool(torch.isfinite(score)):
        raise RuntimeError("projected-gradient base Step2 score is non-finite")
    return score


def build_projected_gradient_h10_proposal(
    *,
    model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    max_delta_per_update: float = 0.5,
    gradient_steps: int = 6,
    step_fraction: float = 0.25,
) -> ProjectedGradientH10Proposal:
    """Generate one bounded 109-D H10 proposal using deterministic projected gradient descent.

    Gradients are normalized before stepping, three fixed backtracking fractions are tested per
    iteration, and every trial is projected relative to the active supervisory target. The projection
    itself handles one-sided physical headroom at bounds, so an actuator at setting 1 can still move
    downward and an actuator at setting 0 can still move upward. Even when the base Step2 score does
    not improve, the best finite non-HOLD projected trial is retained as an exploratory proposal; the
    policy-return critic, not this proposer, decides whether it is beneficial enough to execute.
    """
    if int(gradient_steps) <= 0:
        raise ValueError("gradient_steps must be positive")
    if not 0.0 < float(step_fraction) <= 1.0:
        raise ValueError("step_fraction must lie in (0,1]")
    if not 1 <= int(max_changed_facilities) <= 109:
        raise ValueError("projected-gradient changed-facility ceiling must lie in [1,109]")
    if active_target.shape != (109,):
        raise ValueError("projected-gradient active target must contain 109 settings")

    active_np = active_target.detach().cpu().numpy().astype(np.float64)
    radius = np.asarray(first_radius, dtype=np.float64).reshape(-1)
    if radius.shape != (109,) or not np.isfinite(radius).all():
        raise ValueError("projected-gradient first-move radius must contain 109 finite entries")
    allowed = np.minimum(np.maximum(radius, 0.0), float(max_delta_per_update))
    if int(np.count_nonzero(allowed > 1.0e-7)) == 0:
        return ProjectedGradientH10Proposal(
            target=None,
            start_score_m3=0.0,
            best_score_m3=0.0,
            attempted_steps=0,
            accepted_improvement_steps=0,
            final_gradient_l2=0.0,
            produced_nonhold_candidate=False,
        )

    start_score = float(
        score_h10_first_action_targets(
            model=model,
            normalization=normalization,
            graph=graph,
            current_state=current_state,
            rainfall_scenarios=rainfall_scenarios,
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
            candidate_targets=active_target.reshape(1, 109),
            probe_chunk_size=1,
        )[0].detach().cpu()
    )
    current = active_target.detach().clone()
    current_score = start_score
    best_nonhold_target: torch.Tensor | None = None
    best_nonhold_score = float("inf")
    accepted = 0
    attempted = 0
    final_gradient_l2 = 0.0

    model_was_training = bool(model.training)
    parameter_flags = [bool(parameter.requires_grad) for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    try:
        for _ in range(int(gradient_steps)):
            attempted += 1
            variable = current.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                objective = _score_single_h10_target_with_grad(
                    model=model,
                    normalization=normalization,
                    graph=graph,
                    current_state=current_state,
                    rainfall_scenarios=rainfall_scenarios,
                    previous_actuator_flow=previous_actuator_flow,
                    active_target=active_target,
                    candidate_target=variable,
                )
                gradient = torch.autograd.grad(
                    objective, variable, retain_graph=False, create_graph=False
                )[0]
            if not bool(torch.isfinite(gradient).all()):
                break
            final_gradient_l2 = float(torch.linalg.vector_norm(gradient).detach().cpu())
            grad_scale = float(torch.max(torch.abs(gradient)).detach().cpu())
            if not np.isfinite(grad_scale) or grad_scale <= 1.0e-12:
                break
            direction = -gradient.detach().cpu().numpy().astype(np.float64) / grad_scale
            trial_targets: list[np.ndarray] = []
            for backtrack in (1.0, 0.5, 0.25):
                raw = (
                    current.detach().cpu().numpy().astype(np.float64)
                    + float(step_fraction) * float(backtrack) * allowed * direction
                )
                projected = _bounded_supported_target(
                    active_target=active_np,
                    raw_delta=raw - active_np,
                    graph=graph,
                    first_radius=radius,
                    max_changed_facilities=int(max_changed_facilities),
                    max_delta_per_update=float(max_delta_per_update),
                )
                if np.count_nonzero(np.abs(projected.astype(np.float64) - active_np) > 1.0e-7) <= 0:
                    continue
                if any(np.array_equal(projected, existing) for existing in trial_targets):
                    continue
                trial_targets.append(projected)
            if not trial_targets:
                break
            tensor = torch.as_tensor(
                np.stack(trial_targets),
                dtype=active_target.dtype,
                device=active_target.device,
            )
            scores = score_h10_first_action_targets(
                model=model,
                normalization=normalization,
                graph=graph,
                current_state=current_state,
                rainfall_scenarios=rainfall_scenarios,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
                candidate_targets=tensor,
                probe_chunk_size=len(trial_targets),
            )
            index = int(torch.argmin(scores).item())
            trial_score = float(scores[index].detach().cpu())
            trial = tensor[index].detach()
            if trial_score < best_nonhold_score:
                best_nonhold_score = trial_score
                best_nonhold_target = trial.clone()
            if trial_score < current_score - 1.0e-6:
                current = trial.clone()
                current_score = trial_score
                accepted += 1
            else:
                break
    finally:
        for parameter, flag in zip(model.parameters(), parameter_flags, strict=True):
            parameter.requires_grad_(flag)
        model.train(model_was_training)

    target = best_nonhold_target
    produced = target is not None
    return ProjectedGradientH10Proposal(
        target=target,
        start_score_m3=start_score,
        best_score_m3=(best_nonhold_score if produced else start_score),
        attempted_steps=attempted,
        accepted_improvement_steps=accepted,
        final_gradient_l2=final_gradient_l2,
        produced_nonhold_candidate=produced,
    )


def build_hybrid_policy_return_portfolio(
    *,
    model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    first_radius: np.ndarray,
    max_changed_facilities: int,
    max_delta_per_update: float = 0.5,
    probe_chunk_size: int = 24,
    gradient_steps: int = 6,
    gradient_step_fraction: float = 0.25,
) -> HybridPolicyReturnPortfolio:
    """Build the current at-most-four-candidate Practical H10 portfolio."""
    learned = build_learned_h10_probe_proposal(
        model=model,
        normalization=normalization,
        graph=graph,
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        first_radius=first_radius,
        max_changed_facilities=int(max_changed_facilities),
        max_delta_per_update=float(max_delta_per_update),
        probe_chunk_size=int(probe_chunk_size),
    )
    deterministic = list(
        build_policy_return_candidate_portfolio(
            current_state=current_state,
            rainfall_scenarios=rainfall_scenarios,
            active_target=active_target,
            learned_target=learned.target,
            graph=graph,
            first_radius=first_radius,
            max_changed_facilities=int(max_changed_facilities),
            max_delta_per_update=float(max_delta_per_update),
        )
    )
    gradient = build_projected_gradient_h10_proposal(
        model=model,
        normalization=normalization,
        graph=graph,
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        first_radius=first_radius,
        max_changed_facilities=int(max_changed_facilities),
        max_delta_per_update=float(max_delta_per_update),
        gradient_steps=int(gradient_steps),
        step_fraction=float(gradient_step_fraction),
    )
    candidates = list(deterministic)
    if gradient.target is not None:
        key = gradient.target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
        seen = {
            row.target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            for row in candidates
        }
        if key not in seen:
            changed = int(
                torch.count_nonzero(torch.abs(gradient.target - active_target) > 1.0e-7).item()
            )
            if changed > 0:
                candidates.append(
                    PolicyReturnPortfolioCandidate(
                        source=PROJECTED_GRADIENT_SOURCE,
                        target=gradient.target.detach(),
                        changed_facility_count=changed,
                    )
                )
    return HybridPolicyReturnPortfolio(
        candidates=tuple(candidates[:4]),
        learned_probe=learned,
        projected_gradient=gradient,
    )


__all__ = [
    "DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT",
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT",
    "DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT",
    "HybridPolicyReturnPortfolio",
    "PROJECTED_GRADIENT_SOURCE",
    "ProjectedGradientH10Proposal",
    "build_hybrid_policy_return_portfolio",
    "build_projected_gradient_h10_proposal",
]
