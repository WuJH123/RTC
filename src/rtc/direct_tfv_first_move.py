"""Receding-consistent refinement of the only Direct-TFV action actually executed.

The upstream V6/V7 optimizer remains useful for discovering a coordinated H120 direction inside the
q95 D3-HOLD support geometry.  However, the controller commits only the first 10-minute block before
observing the network again.  This module therefore refines that first move *within* the already
supported V6 direction: each changed facility receives an independent shrink factor in [0,1], the
remaining H350 is HOLD_ACTIVE_TARGET, and Step2 scores the resulting H360 consequence.

Because the refinement can only shrink the supported V6 first move, it never expands the action
outside the upstream q95 query geometry.  It is a small K-dimensional differentiable optimization
(K <= the frozen active-set ceiling), not another 12 x K H120 search.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize


DIRECT_TFV_FIRST_MOVE_SEMANTICS = "REFINE_SUPPORTED_V6_FIRST_MOVE_THEN_HOLD_ACTIVE_TARGET_H350"


@dataclass(frozen=True)
class DirectTFVFirstMoveRefinement:
    sequence: torch.Tensor
    predicted_delta_tfv_m3: float
    base_prefix_predicted_delta_tfv_m3: float
    gain_vs_base_prefix_m3: float
    changed_facility_count: int
    changed_facility_ids: tuple[str, ...]
    optimizer_success: bool
    optimizer_steps: int
    optimizer_starts: int
    elapsed_seconds: float
    scipy_message: str


def _prefix_sequence(
    *,
    mpc: Any,
    active_target: torch.Tensor,
    base_delta: torch.Tensor,
    changed_indices: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    hold = mpc._hold_sequence(active_target).clone()
    target = active_target.clone()
    target[changed_indices] = (
        active_target[changed_indices] + base_delta[changed_indices] * scales
    )
    hold[: int(mpc.design.control_block_steps)] = target[None]
    # The base V6 first move is already q95-supported.  Independent shrink factors cannot exceed
    # its per-facility displacement, but the radial contraction is retained as a fail-closed check
    # on joint first-block/H120/TV geometry.
    if hasattr(mpc, "_contract_to_joint_sequence_support"):
        hold = mpc._contract_to_joint_sequence_support(hold, active_target)
    return hold


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
    """Refine the V6/V7 first move while preserving its direction and support envelope."""

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
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            elapsed_seconds=0.0,
            scipy_message="NO_EXECUTABLE_FIRST_MOVE",
        )

    with torch.no_grad():
        base_prefix = _prefix_sequence(
            mpc=mpc,
            active_target=active_target,
            base_delta=base_delta,
            changed_indices=changed,
            scales=torch.ones(int(changed.numel()), dtype=active_target.dtype, device=active_target.device),
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
            sequence = _prefix_sequence(
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

        remaining = max(1, int(maxiter))
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * int(changed.numel()),
            options={"maxiter": remaining, "ftol": 1.0e-7, "gtol": 1.0e-5},
        )
        raw = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        scales = torch.as_tensor(raw, dtype=active_target.dtype, device=active_target.device)
        sequence = _prefix_sequence(
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

    first_target = best_sequence[:block_steps].mean(dim=0)
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
        optimizer_success=True if chosen is None else bool(chosen.success),
        optimizer_steps=0 if chosen is None else int(getattr(chosen, "nit", 0)),
        optimizer_starts=len(starts),
        elapsed_seconds=float(elapsed),
        scipy_message=(
            "BASE_PREFIX_RETAINED"
            if chosen is None
            else str(getattr(chosen, "message", ""))[:1000]
        ),
    )


__all__ = [
    "DIRECT_TFV_FIRST_MOVE_SEMANTICS",
    "DirectTFVFirstMoveRefinement",
    "refine_supported_first_move",
]
