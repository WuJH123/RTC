"""V12.1 Step3: first-move-conditioned robust selection on the frozen V120 Value model.

V120 predicts the value of an entire 360-minute candidate sequence while the rolling
controller executes only the first 10-minute block.  A sequence whose first block is
identical to HOLD must therefore never win merely because its unexecuted tail looks
favourable.  V121 turns the sequence-value model into an action-value selector:

1. generate and project the exact V120/D3-v2 candidate bank;
2. score every complete sequence with the frozen V120 model;
3. group sequences by their *executable first move*;
4. assign the HOLD/no-op group exactly zero value;
5. aggregate continuation values within every non-HOLD first-move group by the median;
6. select the first move lexicographically by robust Delta-TFV, then movement.

No V120 weights, normalization, rainfall information budget, action bounds or candidate
coefficients are changed.  The median is a fixed robust aggregation operator, not a
Validation-tuned hyperparameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .step2_d3_design_v60 import D3V60DesignContract, _coefficient_specs
from .step2_policy_v120 import (
    ValueOnlyCandidatePolicyV120,
    _project_executable_sequences_v120,
)

V121_STEP3_CONTRACT = "PROJECT7_V121_FIRST_MOVE_CONDITIONED_ROBUST_TFV_V1"
FIRST_MOVE_GROUP_ATOL = 1.0e-7


@dataclass(frozen=True)
class FirstMoveRobustResultV121:
    settings: torch.Tensor
    candidate_valid: bool
    tfv_risk_m3: float
    predicted_delta_tfv_m3: float
    selected_candidate_index: int
    candidate_count: int
    scenario_count: int
    reference_is_best: bool
    scoring_projection_applied: bool
    scoring_projection_max: float
    selected_first_move_group: int
    first_move_group_count: int
    selected_group_size: int
    hold_group_size: int
    selected_candidate_family: str
    selected_sequence_delta_tfv_m3: float
    robust_group_delta_tfv_m3: float
    best_vs_hold_improvement_m3: float
    first_move_mean_abs_delta: float
    selected_group_tail_spread_m3: float
    tail_only_noop_candidates: int


def _families_for_v120_bank(policy: ValueOnlyCandidatePolicyV120) -> tuple[str, ...]:
    """Reconstruct family labels without changing the frozen V120 coefficient bank."""

    basis = policy.basis
    contract = policy.contract
    d3 = D3V60DesignContract(
        candidates_per_checkpoint=contract.candidates_per_bank,
        seed=contract.seed,
    )
    labels = ["hold"]
    coefficients = [np.zeros((basis.temporal_basis_count, basis.group_count), np.float32)]
    seen = {coefficients[0].tobytes()}
    for bank in range(contract.candidate_banks):
        checkpoint_seed = int(contract.seed + 104729 * (bank + 1))
        for family, coeff in _coefficient_specs(
            basis, checkpoint_seed=checkpoint_seed, contract=d3
        ):
            value = np.asarray(coeff, np.float32)
            key = value.tobytes()
            if key in seen:
                continue
            seen.add(key)
            coefficients.append(value)
            labels.append(str(family))
    expected = np.asarray(policy._coefficients, dtype=np.float32)  # frozen V120 bank
    actual = np.stack(coefficients)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError("V121 family reconstruction differs from frozen V120 candidates")
    return tuple(labels)


def _first_move_groups(
    first_moves: np.ndarray,
    *,
    atol: float = FIRST_MOVE_GROUP_ATOL,
) -> tuple[tuple[int, ...], ...]:
    """Deterministically group candidates that imply the same executable first move."""

    values = np.asarray(first_moves, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("V121 first moves must be finite [candidate, actuator]")
    groups: list[list[int]] = []
    representatives: list[np.ndarray] = []
    for index, value in enumerate(values):
        matched = None
        for group_index, representative in enumerate(representatives):
            if np.allclose(value, representative, rtol=0.0, atol=float(atol)):
                matched = group_index
                break
        if matched is None:
            representatives.append(value.copy())
            groups.append([index])
        else:
            groups[matched].append(index)
    return tuple(tuple(group) for group in groups)


class FirstMoveRobustCandidatePolicyV121:
    """Use frozen V120 sequence values as robust values of executable first moves."""

    accepts_previous_requested_settings = True

    def __init__(
        self,
        base: ValueOnlyCandidatePolicyV120,
        *,
        first_move_group_atol: float = FIRST_MOVE_GROUP_ATOL,
    ) -> None:
        if not 0.0 < float(first_move_group_atol) <= 1.0e-4:
            raise ValueError("V121 first-move grouping tolerance must be numerical only")
        self.base = base
        self.model = base.model
        self.basis = base.basis
        self.prepared = base.prepared
        self.normalization = base.normalization
        self.contract = base.contract
        self.first_move_group_atol = float(first_move_group_atol)
        self._families = _families_for_v120_bank(base)
        self.last_result: FirstMoveRobustResultV121 | None = None

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
    ) -> FirstMoveRobustResultV121:
        if initial_state.ndim == 3 and initial_state.shape[0] == 1:
            initial_state = initial_state[0]
        if initial_state.ndim != 2:
            raise ValueError("V121 initial_state must resolve to [N,D]")
        if rainfall_scenarios.ndim != 4:
            raise ValueError("V121 rainfall scenarios must be [S,H,N,R]")
        if fallback_settings.ndim != 3 or fallback_settings.shape[0] != 1:
            raise ValueError("V121 fallback settings must be [1,H,A]")
        if fallback_settings.shape[1] != self.basis.horizon.horizon_steps:
            raise ValueError("V121 fallback/value horizons differ")
        if current_settings is None:
            raise ValueError("V121 requires physical current-setting readback")

        current = current_settings.reshape(-1)
        actuator_count = self.basis.grouping.actuator_count
        if current.numel() != actuator_count:
            raise ValueError("V121 current actuator setting count mismatch")
        if previous_actuator_flow is None:
            previous_actuator_flow = torch.zeros(
                actuator_count, dtype=initial_state.dtype, device=initial_state.device
            )
        previous_flow = previous_actuator_flow.reshape(-1)
        if previous_flow.numel() != actuator_count:
            raise ValueError("V121 previous actuator flow count mismatch")

        coefficients = torch.as_tensor(
            self.base._coefficients,
            dtype=fallback_settings.dtype,
            device=fallback_settings.device,
        )
        reference_one = fallback_settings
        reference_for_candidates = reference_one[:, None].expand(
            1, coefficients.shape[0], -1, -1
        )
        candidate_one = self.basis.decode(reference_for_candidates, coefficients[None])[0]

        frozen_delta = float(self.basis.contract.max_setting_delta_per_update)
        requested_delta = frozen_delta
        if max_delta_per_update is not None:
            raw_delta = torch.as_tensor(max_delta_per_update, dtype=torch.float32).reshape(-1)
            if not bool(torch.isfinite(raw_delta).all()):
                raise ValueError("V121 runtime max delta is non-finite")
            requested_delta = float(raw_delta.max())
            if requested_delta > frozen_delta + 1.0e-9:
                raise ValueError("V121 runtime max delta is looser than frozen basis")

        candidate_one, projection_max = _project_executable_sequences_v120(
            candidate_one,
            current_settings=current,
            previous_requested_settings=previous_requested_settings,
            min_settings=torch.as_tensor(
                self.basis.min_setting,
                dtype=candidate_one.dtype,
                device=candidate_one.device,
            ),
            max_settings=torch.as_tensor(
                self.basis.max_setting,
                dtype=candidate_one.dtype,
                device=candidate_one.device,
            ),
            max_delta_per_update=requested_delta,
            control_block_steps=int(self.basis.horizon.control_block_steps),
        )
        if not torch.allclose(candidate_one[0], reference_one[0], rtol=0.0, atol=1.0e-7):
            raise RuntimeError("V121 executable projection changed the HOLD reference")

        scenario_count = int(rainfall_scenarios.shape[0])
        state = initial_state[None].expand(scenario_count, -1, -1)
        reference = reference_one.expand(scenario_count, -1, -1)
        candidates = candidate_one[None].expand(scenario_count, -1, -1, -1)
        flow = previous_flow[None].expand(scenario_count, -1)
        output = self.model(
            self.normalization.state(state),
            self.normalization.rainfall(rainfall_scenarios),
            reference,
            candidates,
            self.normalization.flow(flow),
            self.prepared,
        )
        predicted = output.delta_tfv_m3
        if not torch.isfinite(predicted).all():
            raise RuntimeError("V121 frozen V120 Value produced non-finite Delta-TFV")
        # Runtime V120 uses one causal rainfall scenario.  Mean preserves that exact value
        # and remains well-defined if a future pre-registered V121 audit adds scenarios.
        sequence_risk = predicted.mean(dim=0)
        if abs(float(sequence_risk[0].detach())) > 1.0e-5:
            raise RuntimeError("V121 HOLD lost exact-zero Value")

        block = int(self.basis.horizon.control_block_steps)
        reference_first = reference_one[0, :block].mean(dim=0)
        candidate_first = candidate_one[:, :block].mean(dim=1)
        first_np = candidate_first.detach().cpu().numpy()
        reference_np = reference_first.detach().cpu().numpy()
        groups = _first_move_groups(first_np, atol=self.first_move_group_atol)
        hold_group = next(i for i, members in enumerate(groups) if 0 in members)

        risk_np = sequence_risk.detach().cpu().numpy().astype(np.float64)
        group_scores: list[float] = []
        group_movements: list[float] = []
        group_spreads: list[float] = []
        noop_members = 0
        for members in groups:
            member = np.asarray(members, dtype=int)
            first = first_np[int(member[0])]
            movement = float(np.mean(np.abs(first - reference_np)))
            is_noop = bool(
                np.allclose(first, reference_np, rtol=0.0, atol=self.first_move_group_atol)
            )
            if is_noop:
                # The controller will execute the same first move as HOLD.  Any predicted
                # benefit that exists only in an unexecuted future tail is non-actionable.
                score = 0.0
                noop_members += int(len(member))
            else:
                score = float(np.median(risk_np[member]))
            spread = float(np.ptp(risk_np[member])) if len(member) > 1 else 0.0
            group_scores.append(score)
            group_movements.append(movement)
            group_spreads.append(spread)

        # Lexicographic selection avoids a new tunable weighted objective.
        selected_group = min(
            range(len(groups)),
            key=lambda i: (group_scores[i], group_movements[i], i),
        )
        selected_score = float(group_scores[selected_group])
        valid = bool(
            selected_group != hold_group
            and -selected_score > float(self.base.min_predicted_improvement_m3)
        )
        if not valid:
            selected_group = hold_group
            selected_score = 0.0
            selected_index = 0
        else:
            members = np.asarray(groups[selected_group], dtype=int)
            selected_index = int(
                members[np.argmin(np.abs(risk_np[members] - selected_score))]
            )

        selected_members = groups[selected_group]
        first_move_delta = float(
            np.mean(np.abs(first_np[selected_index] - reference_np))
        )
        result = FirstMoveRobustResultV121(
            settings=candidate_one[selected_index].detach(),
            candidate_valid=valid,
            tfv_risk_m3=selected_score,
            predicted_delta_tfv_m3=selected_score,
            selected_candidate_index=selected_index,
            candidate_count=int(candidate_one.shape[0]),
            scenario_count=scenario_count,
            reference_is_best=bool(selected_group == hold_group),
            scoring_projection_applied=bool(projection_max > 1.0e-9),
            scoring_projection_max=float(projection_max),
            selected_first_move_group=int(selected_group),
            first_move_group_count=int(len(groups)),
            selected_group_size=int(len(selected_members)),
            hold_group_size=int(len(groups[hold_group])),
            selected_candidate_family=str(self._families[selected_index]),
            selected_sequence_delta_tfv_m3=float(risk_np[selected_index]),
            robust_group_delta_tfv_m3=selected_score,
            best_vs_hold_improvement_m3=max(0.0, -selected_score),
            first_move_mean_abs_delta=first_move_delta,
            selected_group_tail_spread_m3=float(group_spreads[selected_group]),
            tail_only_noop_candidates=max(0, int(noop_members - 1)),
        )
        self.last_result = result
        return result


__all__ = [
    "FIRST_MOVE_GROUP_ATOL",
    "FirstMoveRobustCandidatePolicyV121",
    "FirstMoveRobustResultV121",
    "V121_STEP3_CONTRACT",
    "_first_move_groups",
]
