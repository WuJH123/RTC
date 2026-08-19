"""Target-latch-consistent refinement of the Direct-TFV action actually executed next.

The upstream V6/V7 optimizer remains useful for discovering a coordinated H120 direction inside the
q95 D3-HOLD support geometry. The online controller, however, writes a *target latch*: once a new
10-minute target is accepted it remains the supervisory command until a later decision changes it.
Therefore the correct no-further-command counterfactual is not ``H10 new -> old target H350``. It is
``write the refined first target now -> keep that new target latched through H360``.

This module refines the V6 first-move direction with one shrink factor per changed facility in [0,1].
Unchanged facilities remain exactly at the previous target latch. After continuous refinement, an
objective-consistent backward pruning pass removes any facility whose return to the previous target
does not worsen predicted TFV. The pruning introduces no action penalty or tuned sparsity weight:
actual changed-facility count is selected by the learned TFV objective itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize


DIRECT_TFV_FIRST_MOVE_SEMANTICS = (
    "WRITE_REFINED_H10_TARGET_THEN_LATCH_NEW_TARGET_UNTIL_NEXT_COMMAND_H360"
)
_PRUNE_NUMERICAL_TOLERANCE_M3 = 1.0e-5


@dataclass(frozen=True)
class DirectTFVFirstMoveRefinement:
    sequence: torch.Tensor
    predicted_delta_tfv_m3: float
    base_prefix_predicted_delta_tfv_m3: float
    gain_vs_base_prefix_m3: float
    changed_facility_count: int
    changed_facility_ids: tuple[str, ...]
    pre_prune_changed_facility_count: int
    pruned_facility_count: int
    optimizer_success: bool
    optimizer_steps: int
    optimizer_starts: int
    elapsed_seconds: float
    scipy_message: str


def _latched_target_sequence(target: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    return target[None].expand(int(horizon_steps), -1).clone()


def _latched_first_move_sequence(
    *,
    mpc: Any,
    active_target: torch.Tensor,
    base_delta: torch.Tensor,
    changed_indices: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Repeat the newly written target through H360 under no-further-command semantics."""

    target = active_target.clone()
    target[changed_indices] = (
        active_target[changed_indices] + base_delta[changed_indices] * scales
    )
    sequence = _latched_target_sequence(target, int(mpc.design.prediction_horizon_steps))
    # The base first move is q95-supported. Shrink-only refinement cannot increase any per-facility
    # displacement, but retain the joint contraction as a fail-closed check on first-block/H120 mass.
    if hasattr(mpc, "_contract_to_joint_sequence_support"):
        sequence = mpc._contract_to_joint_sequence_support(sequence, active_target)
    return sequence


def _score_sequence_batch(
    *,
    mpc: Any,
    sequences: torch.Tensor,
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
) -> torch.Tensor:
    if hasattr(mpc, "_score_sequences"):
        return mpc._score_sequences(
            current_state=current_state,
            rainfall=rainfall,
            sequences=sequences,
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
        )
    return torch.stack(
        [
            mpc.score_sequence(
                current_state=current_state,
                rainfall=rainfall,
                sequence=sequence,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            )
            for sequence in sequences
        ]
    )


def _prune_non_beneficial_facilities(
    *,
    mpc: Any,
    sequence: torch.Tensor,
    score_m3: float,
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
) -> tuple[torch.Tensor, float, int, int]:
    """Greedily remove target changes that do not improve the predicted TFV objective.

    Reverting one facility to the previous latch only shrinks the action, so every pruned candidate
    remains inside the support envelope of the already-contracted sequence. Candidate removals are
    scored as one GPU batch per pruning round. A facility is removed only when its removal is no worse
    than the current candidate up to a fixed floating-point tolerance.
    """

    target = sequence[0].detach().clone()
    active = torch.nonzero(
        torch.abs(target - active_target) > 1.0e-7, as_tuple=False
    ).reshape(-1)
    before = int(active.numel())
    if before == 0:
        return sequence.detach(), float(score_m3), 0, 0

    current_score = float(score_m3)
    current_target = target
    while int(active.numel()) > 0:
        candidates: list[torch.Tensor] = []
        for index in active.detach().cpu().tolist():
            candidate_target = current_target.clone()
            candidate_target[int(index)] = active_target[int(index)]
            candidates.append(
                _latched_target_sequence(
                    candidate_target, int(mpc.design.prediction_horizon_steps)
                )
            )
        with torch.no_grad():
            scores = _score_sequence_batch(
                mpc=mpc,
                sequences=torch.stack(candidates),
                current_state=current_state,
                rainfall=rainfall,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            ).detach()
        best_position = int(torch.argmin(scores).item())
        best_score = float(scores[best_position].cpu())
        if best_score > current_score + _PRUNE_NUMERICAL_TOLERANCE_M3:
            break
        removed_index = int(active[best_position].item())
        current_target = current_target.clone()
        current_target[removed_index] = active_target[removed_index]
        current_score = best_score
        active = torch.nonzero(
            torch.abs(current_target - active_target) > 1.0e-7, as_tuple=False
        ).reshape(-1)

    final_sequence = _latched_target_sequence(
        current_target, int(mpc.design.prediction_horizon_steps)
    )
    after = int(active.numel())
    return final_sequence.detach(), float(current_score), before, before - after


def refine_supported_first_move(
    *,
    mpc: Any,
    base_candidate: torch.Tensor,
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    maxiter: int = 12,
    deadline_seconds: float = 30.0,
) -> DirectTFVFirstMoveRefinement:
    """Refine, then objective-prune, the supported V6/V7 first target."""

    if tuple(base_candidate.shape) != (int(mpc.design.prediction_horizon_steps), 109):
        raise ValueError("first-move refinement requires a [H72,109] base candidate")
    if tuple(active_target.shape) != (109,):
        raise ValueError("first-move refinement requires 109 active-target settings")
    if maxiter <= 0 or not 0.0 < float(deadline_seconds) < 600.0:
        raise ValueError("invalid first-move refinement solver budget")

    block_steps = int(mpc.design.control_block_steps)
    base_target = base_candidate[:block_steps].mean(dim=0)
    base_delta = base_target - active_target
    changed = torch.nonzero(torch.abs(base_delta) > 1.0e-7, as_tuple=False).reshape(-1)
    hold = mpc._hold_sequence(active_target)
    if int(changed.numel()) == 0:
        return DirectTFVFirstMoveRefinement(
            sequence=hold,
            predicted_delta_tfv_m3=0.0,
            base_prefix_predicted_delta_tfv_m3=0.0,
            gain_vs_base_prefix_m3=0.0,
            changed_facility_count=0,
            changed_facility_ids=(),
            pre_prune_changed_facility_count=0,
            pruned_facility_count=0,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            elapsed_seconds=0.0,
            scipy_message="NO_EXECUTABLE_FIRST_MOVE",
        )

    with torch.no_grad():
        base_prefix = _latched_first_move_sequence(
            mpc=mpc,
            active_target=active_target,
            base_delta=base_delta,
            changed_indices=changed,
            scales=torch.ones(
                int(changed.numel()), dtype=active_target.dtype, device=active_target.device
            ),
        )
        base_score = float(
            mpc.score_sequence(
                current_state=current_state,
                rainfall=rainfall,
                sequence=base_prefix,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            ).detach().cpu()
        )

    started = time.perf_counter()
    best_score = base_score
    best_sequence = base_prefix.detach()
    best_result = None
    starts = (
        np.ones(int(changed.numel()), dtype=np.float64),
        np.full(int(changed.numel()), 0.5, dtype=np.float64),
    )

    for start in starts:
        if time.perf_counter() - started >= float(deadline_seconds):
            break

        def objective(raw: np.ndarray) -> tuple[float, np.ndarray]:
            scales = torch.tensor(
                np.asarray(raw, dtype=np.float64),
                dtype=active_target.dtype,
                device=active_target.device,
                requires_grad=True,
            )
            sequence = _latched_first_move_sequence(
                mpc=mpc,
                active_target=active_target,
                base_delta=base_delta,
                changed_indices=changed,
                scales=scales,
            )
            score = mpc.score_sequence(
                current_state=current_state,
                rainfall=rainfall,
                sequence=sequence,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            )
            gradient = torch.autograd.grad(score, scales, retain_graph=False, create_graph=False)[0]
            return (
                float(score.detach().cpu()),
                gradient.detach().cpu().numpy().astype(np.float64),
            )

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * int(changed.numel()),
            options={"maxiter": int(maxiter), "ftol": 1.0e-7, "gtol": 1.0e-5},
        )
        raw = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        scales = torch.as_tensor(raw, dtype=active_target.dtype, device=active_target.device)
        sequence = _latched_first_move_sequence(
            mpc=mpc,
            active_target=active_target,
            base_delta=base_delta,
            changed_indices=changed,
            scales=scales,
        )
        with torch.no_grad():
            score_value = float(
                mpc.score_sequence(
                    current_state=current_state,
                    rainfall=rainfall,
                    sequence=sequence,
                    previous_actuator_flow=previous_actuator_flow,
                    active_target=active_target,
                ).detach().cpu()
            )
        if score_value < best_score:
            best_score = score_value
            best_sequence = sequence.detach()
            best_result = result

    best_sequence, best_score, pre_prune_count, pruned_count = _prune_non_beneficial_facilities(
        mpc=mpc,
        sequence=best_sequence,
        score_m3=best_score,
        current_state=current_state,
        rainfall=rainfall,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
    )
    first_target = best_sequence[0]
    final_changed = torch.nonzero(
        torch.abs(first_target - active_target) > 1.0e-7, as_tuple=False
    ).reshape(-1)
    actuator_ids = tuple(str(value) for value in mpc.graph.actuator_ids)
    changed_ids = tuple(actuator_ids[int(i)] for i in final_changed.detach().cpu().tolist())
    elapsed = time.perf_counter() - started
    chosen = best_result
    return DirectTFVFirstMoveRefinement(
        sequence=best_sequence,
        predicted_delta_tfv_m3=float(best_score),
        base_prefix_predicted_delta_tfv_m3=float(base_score),
        gain_vs_base_prefix_m3=float(base_score - best_score),
        changed_facility_count=int(final_changed.numel()),
        changed_facility_ids=changed_ids,
        pre_prune_changed_facility_count=int(pre_prune_count),
        pruned_facility_count=int(pruned_count),
        optimizer_success=True if chosen is None else bool(chosen.success),
        optimizer_steps=0 if chosen is None else int(getattr(chosen, "nit", 0)),
        optimizer_starts=len(starts),
        elapsed_seconds=float(elapsed),
        scipy_message=(
            "BASE_LATCHED_FIRST_MOVE_RETAINED"
            if chosen is None
            else str(getattr(chosen, "message", ""))[:1000]
        ),
    )


__all__ = [
    "DIRECT_TFV_FIRST_MOVE_SEMANTICS",
    "DirectTFVFirstMoveRefinement",
    "refine_supported_first_move",
]
