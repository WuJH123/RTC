"""Target-latch-consistent refinement of the Direct-TFV action actually executed next.

The upstream V6/V7 optimizer remains useful for discovering a coordinated H120 direction inside the
q95 D3-HOLD support geometry. The online controller, however, writes a *target latch*: once a new
10-minute target is accepted it remains the supervisory command until a later decision changes it.
Therefore the correct no-further-command counterfactual is not ``H10 new -> old target H350``. It is
``write the refined first target now -> keep that new target latched through H360``.

This module refines the V6 first-move direction with one shrink factor per changed facility in [0,1].
Unchanged facilities remain exactly at the previous target latch. Because refinement can only shrink
the supported V6 first displacement, it never expands the first move outside the upstream support
envelope. After the continuous refinement, a deterministic TFV-only backward elimination removes a
facility only when returning that facility exactly to HOLD is numerically non-worse for predicted TFV.
This makes the *actual* changed-facility subset a model result rather than merely inheriting the
upstream active set, without adding a sparsity penalty or changing the TFV objective. The resulting
target is repeated through H360 solely as the no-further-command value counterfactual; the real
controller still re-observes and may issue a new target after 10 minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize


DIRECT_TFV_FIRST_MOVE_SEMANTICS = (
    "WRITE_REFINED_H10_TARGET_THEN_LATCH_NEW_TARGET_UNTIL_NEXT_COMMAND_H360"
)
DIRECT_TFV_FIRST_MOVE_PRUNING_RULE = (
    "TFV_ONLY_BACKWARD_ELIMINATION_NUMERICAL_NONWORSENING_TO_HOLD"
)


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
    pruned_facility_ids: tuple[str, ...]
    pruning_evaluations: int
    pruning_gain_m3: float
    optimizer_success: bool
    optimizer_steps: int
    optimizer_starts: int
    elapsed_seconds: float
    scipy_message: str


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
    sequence = target[None].expand(int(mpc.design.prediction_horizon_steps), -1).clone()
    # The base first move is q95-supported. Shrink-only refinement cannot increase any per-facility
    # displacement, but retain the joint contraction as a fail-closed check on first-block/H120 mass.
    if hasattr(mpc, "_contract_to_joint_sequence_support"):
        sequence = mpc._contract_to_joint_sequence_support(sequence, active_target)
    return sequence


def _score_value(
    *,
    mpc: Any,
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    sequence: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
) -> float:
    with torch.no_grad():
        score = mpc.score_sequence(
            current_state=current_state,
            rainfall=rainfall,
            sequence=sequence,
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
        )
    value = float(score.detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError("Direct-TFV first-move refinement produced a non-finite TFV score")
    return value


def _score_tie_tolerance_m3(score_m3: float) -> float:
    """Machine-resolution tie tolerance; this is not a performance threshold."""

    magnitude = np.float32(abs(float(score_m3)))
    probe = magnitude if magnitude > 0.0 else np.float32(1.0)
    spacing = abs(float(np.spacing(probe)))
    return max(1.0e-6, 8.0 * spacing)


def _backward_prune_first_move(
    *,
    mpc: Any,
    sequence: torch.Tensor,
    score_m3: float,
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    deadline_at: float,
) -> tuple[torch.Tensor, float, tuple[int, ...], int]:
    """Greedily remove TFV-redundant first-move facilities without changing the objective.

    At each round every currently changed facility is tested by setting exactly that facility back to
    the previous target latch. The numerically best leave-one-out candidate is accepted only if its
    predicted TFV is no worse than the current candidate within float32 score resolution. Because each
    accepted move only removes displacement, q95 per-facility and joint-sequence support cannot expand.
    """

    current = sequence.detach().clone()
    current_score = float(score_m3)
    pruned: list[int] = []
    evaluations = 0

    while time.perf_counter() < deadline_at:
        first = current[0]
        changed = torch.nonzero(
            torch.abs(first - active_target) > 1.0e-7, as_tuple=False
        ).reshape(-1)
        if int(changed.numel()) == 0:
            break

        best_sequence: torch.Tensor | None = None
        best_score = float("inf")
        best_index: int | None = None
        for raw_index in changed.detach().cpu().tolist():
            if time.perf_counter() >= deadline_at:
                return current, current_score, tuple(pruned), evaluations
            index = int(raw_index)
            candidate = current.clone()
            candidate[:, index] = active_target[index]
            if hasattr(mpc, "_contract_to_joint_sequence_support"):
                candidate = mpc._contract_to_joint_sequence_support(candidate, active_target)
            candidate_score = _score_value(
                mpc=mpc,
                current_state=current_state,
                rainfall=rainfall,
                sequence=candidate,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            )
            evaluations += 1
            if candidate_score < best_score:
                best_score = candidate_score
                best_sequence = candidate
                best_index = index

        if best_sequence is None or best_index is None:
            break
        tolerance = _score_tie_tolerance_m3(current_score)
        if best_score > current_score + tolerance:
            break
        current = best_sequence.detach()
        current_score = float(best_score)
        pruned.append(best_index)

    return current, current_score, tuple(pruned), evaluations


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
    """Refine the supported V6/V7 first target while preserving direction and latch semantics."""

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
            pruned_facility_ids=(),
            pruning_evaluations=0,
            pruning_gain_m3=0.0,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            elapsed_seconds=0.0,
            scipy_message="NO_EXECUTABLE_FIRST_MOVE",
        )

    base_prefix = _latched_first_move_sequence(
        mpc=mpc,
        active_target=active_target,
        base_delta=base_delta,
        changed_indices=changed,
        scales=torch.ones(
            int(changed.numel()), dtype=active_target.dtype, device=active_target.device
        ),
    )
    base_score = _score_value(
        mpc=mpc,
        current_state=current_state,
        rainfall=rainfall,
        sequence=base_prefix,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
    )

    started = time.perf_counter()
    deadline_at = started + float(deadline_seconds)
    best_score = base_score
    best_sequence = base_prefix.detach()
    best_result = None
    starts = (
        np.ones(int(changed.numel()), dtype=np.float64),
        np.full(int(changed.numel()), 0.5, dtype=np.float64),
    )
    attempted_starts = 0
    total_steps = 0
    messages: list[str] = []

    for start in starts:
        if time.perf_counter() >= deadline_at:
            messages.append("deadline reached before next continuous-refinement start")
            break
        attempted_starts += 1

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
            if not bool(torch.isfinite(score)) or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(
                    "Direct-TFV first-move refinement produced non-finite objective/gradient"
                )
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
        total_steps += int(getattr(result, "nit", 0))
        messages.append(str(getattr(result, "message", "")))
        raw = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        scales = torch.as_tensor(raw, dtype=active_target.dtype, device=active_target.device)
        sequence = _latched_first_move_sequence(
            mpc=mpc,
            active_target=active_target,
            base_delta=base_delta,
            changed_indices=changed,
            scales=scales,
        )
        score_value = _score_value(
            mpc=mpc,
            current_state=current_state,
            rainfall=rainfall,
            sequence=sequence,
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
        )
        if score_value < best_score:
            best_score = score_value
            best_sequence = sequence.detach()
            best_result = result

    pre_prune_sequence = best_sequence.detach()
    pre_prune_score = float(best_score)
    pre_prune_changed = torch.nonzero(
        torch.abs(pre_prune_sequence[0] - active_target) > 1.0e-7, as_tuple=False
    ).reshape(-1)
    best_sequence, best_score, pruned_indices, pruning_evaluations = _backward_prune_first_move(
        mpc=mpc,
        sequence=pre_prune_sequence,
        score_m3=pre_prune_score,
        current_state=current_state,
        rainfall=rainfall,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        deadline_at=deadline_at,
    )

    first_target = best_sequence[0]
    final_changed = torch.nonzero(
        torch.abs(first_target - active_target) > 1.0e-7, as_tuple=False
    ).reshape(-1)
    actuator_ids = tuple(str(value) for value in mpc.graph.actuator_ids)
    changed_ids = tuple(actuator_ids[int(i)] for i in final_changed.detach().cpu().tolist())
    pruned_ids = tuple(actuator_ids[int(i)] for i in pruned_indices)
    elapsed = time.perf_counter() - started
    chosen = best_result
    if pruned_ids:
        messages.append(
            f"{DIRECT_TFV_FIRST_MOVE_PRUNING_RULE}: pruned={len(pruned_ids)} evaluations={pruning_evaluations}"
        )
    return DirectTFVFirstMoveRefinement(
        sequence=best_sequence,
        predicted_delta_tfv_m3=float(best_score),
        base_prefix_predicted_delta_tfv_m3=float(base_score),
        gain_vs_base_prefix_m3=float(base_score - best_score),
        changed_facility_count=int(final_changed.numel()),
        changed_facility_ids=changed_ids,
        pre_prune_changed_facility_count=int(pre_prune_changed.numel()),
        pruned_facility_count=len(pruned_ids),
        pruned_facility_ids=pruned_ids,
        pruning_evaluations=int(pruning_evaluations),
        pruning_gain_m3=float(pre_prune_score - best_score),
        optimizer_success=True if chosen is None else bool(chosen.success),
        optimizer_steps=int(total_steps),
        optimizer_starts=int(attempted_starts),
        elapsed_seconds=float(elapsed),
        scipy_message=(
            "BASE_LATCHED_FIRST_MOVE_RETAINED"
            if not messages
            else " | ".join(messages)[:2000]
        ),
    )


__all__ = [
    "DIRECT_TFV_FIRST_MOVE_PRUNING_RULE",
    "DIRECT_TFV_FIRST_MOVE_SEMANTICS",
    "DirectTFVFirstMoveRefinement",
    "refine_supported_first_move",
]
