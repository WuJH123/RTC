"""V12.2 finite-candidate policy adapter.

This module keeps the frozen V120 implementation untouched for lineage loading and
adds only the V122 first-move grouping rule.  Candidate sequences are projected before
scoring; the selected executable first block is never projected after scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .step2_policy_v120 import (
    ValueOnlyCandidatePolicyV120,
    ValueOnlyPolicyResultV120,
    _project_executable_sequences_v120,
    _upper_tail_cvar_per_candidate,
)


@dataclass(frozen=True)
class FirstMovePolicyResultV122:
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
    selected_group_score_m3: float
    first_move_group_count: int
    tail_only_noop_candidate_count: int


class FirstMoveFinitePolicyV122(ValueOnlyCandidatePolicyV120):
    """Reuse a loaded V120 policy while aggregating risk by executable first move."""

    def __init__(self, base: ValueOnlyCandidatePolicyV120, *, first_move_group_atol: float = 1.0e-7):
        self.__dict__.update(base.__dict__)
        self.first_move_group_atol = float(first_move_group_atol)
        if not torch.isfinite(torch.as_tensor(self.first_move_group_atol)) or self.first_move_group_atol <= 0:
            raise ValueError("V122 first-move grouping tolerance must be positive")

    def optimize(self, initial_state: torch.Tensor, rainfall_scenarios: torch.Tensor,
                 fallback_settings: torch.Tensor, *, current_settings: torch.Tensor | None = None,
                 previous_requested_settings: torch.Tensor | None = None,
                 previous_actuator_flow: torch.Tensor | None = None,
                 max_delta_per_update: float | torch.Tensor | None = None,
                 **_controller_compatibility: object) -> FirstMovePolicyResultV122:
        if initial_state.ndim == 3 and initial_state.shape[0] == 1:
            initial_state = initial_state[0]
        if initial_state.ndim != 2 or rainfall_scenarios.ndim != 4:
            raise ValueError("V122 finite policy received incompatible initial/rainfall shape")
        if fallback_settings.ndim != 3 or fallback_settings.shape[0] != 1:
            raise ValueError("V122 fallback settings must be [1,H,A]")
        if fallback_settings.shape[1] != self.basis.horizon.horizon_steps:
            raise ValueError("V122 fallback/value horizons differ")
        if current_settings is None:
            raise ValueError("V122 finite policy requires current-setting readback")
        actuator_count = self.basis.grouping.actuator_count
        current_settings = current_settings.reshape(-1)
        if current_settings.numel() != actuator_count:
            raise ValueError("V122 current actuator setting count mismatch")
        if previous_actuator_flow is None:
            previous_actuator_flow = torch.zeros(
                actuator_count, dtype=initial_state.dtype, device=initial_state.device
            )
        previous_actuator_flow = previous_actuator_flow.reshape(-1)
        scenarios = int(rainfall_scenarios.shape[0])
        coeff = torch.as_tensor(self._coefficients, dtype=fallback_settings.dtype, device=fallback_settings.device)
        reference_one = fallback_settings
        reference_for_candidates = reference_one[:, None].expand(1, coeff.shape[0], -1, -1)
        candidate_one = self.basis.decode(reference_for_candidates, coeff[None])[0]
        frozen = float(self.basis.contract.max_setting_delta_per_update)
        requested = frozen
        if max_delta_per_update is not None:
            raw_delta = torch.as_tensor(max_delta_per_update, dtype=torch.float32).reshape(-1)
            if not bool(torch.isfinite(raw_delta).all()):
                raise ValueError("V122 runtime max delta is non-finite")
            requested = float(raw_delta.max())
            if requested > frozen + 1.0e-9:
                raise ValueError("V122 runtime max delta is looser than frozen basis")
        candidate_one, projection_max = _project_executable_sequences_v120(
            candidate_one,
            current_settings=current_settings,
            previous_requested_settings=previous_requested_settings,
            min_settings=torch.as_tensor(self.basis.min_setting, dtype=candidate_one.dtype, device=candidate_one.device),
            max_settings=torch.as_tensor(self.basis.max_setting, dtype=candidate_one.dtype, device=candidate_one.device),
            max_delta_per_update=requested,
            control_block_steps=int(self.basis.horizon.control_block_steps),
        )
        if not torch.allclose(candidate_one[0], reference_one[0], rtol=0.0, atol=1.0e-7):
            raise RuntimeError("V122 executable projection changed HOLD reference")
        state = initial_state[None].expand(scenarios, -1, -1)
        reference = reference_one.expand(scenarios, -1, -1)
        candidate = candidate_one[None].expand(scenarios, -1, -1, -1)
        flow = previous_actuator_flow[None].expand(scenarios, -1)
        output = self.model(
            self.normalization.state(state), self.normalization.rainfall(rainfall_scenarios),
            reference, candidate, self.normalization.flow(flow), self.prepared,
        )
        predicted = output.delta_tfv_m3
        if not bool(torch.isfinite(predicted).all()):
            raise RuntimeError("V122 Value produced non-finite Delta-TFV")
        risk = _upper_tail_cvar_per_candidate(predicted, self.cvar_alpha)
        if abs(float(risk[0].detach())) > 1.0e-5:
            raise RuntimeError("V122 HOLD lost exact-zero value")

        block = int(self.basis.horizon.control_block_steps)
        reference_first = reference_one[0, :block].mean(dim=0)
        candidate_first = candidate_one[:, :block].mean(dim=1)
        rounded = torch.round(candidate_first / self.first_move_group_atol) * self.first_move_group_atol
        groups: dict[bytes, list[int]] = {}
        for idx, row in enumerate(rounded.detach().cpu().numpy()):
            groups.setdefault(row.astype("float64").tobytes(), []).append(idx)
        passive_indices = [
            idx for idx in range(candidate_one.shape[0])
            if bool(torch.allclose(candidate_first[idx], reference_first, rtol=0.0, atol=self.first_move_group_atol))
        ]
        group_records: list[tuple[int, float, bool]] = []
        for indices in groups.values():
            is_passive = any(idx in passive_indices for idx in indices)
            if is_passive:
                representative = next(idx for idx in indices if idx in passive_indices)
                score = 0.0
            else:
                values = risk[indices]
                median = values.median()
                nearest = torch.argmin(torch.abs(values - median))
                representative = indices[int(nearest.item())]
                score = float(median.detach())
            group_records.append((representative, score, is_passive))
        eligible = [
            item for item in group_records
            if not item[2] and item[1] < -self.min_predicted_improvement_m3
        ]
        if eligible:
            selected, group_score, _ = min(eligible, key=lambda item: item[1])
            valid = True
        else:
            selected, group_score, valid = 0, 0.0, False
        selected_risk = float(risk[selected].detach()) if valid else 0.0
        return FirstMovePolicyResultV122(
            settings=candidate_one[selected].detach(), candidate_valid=valid,
            tfv_risk_m3=selected_risk, predicted_delta_tfv_m3=selected_risk,
            selected_candidate_index=int(selected), candidate_count=int(candidate_one.shape[0]),
            scenario_count=scenarios, reference_is_best=bool(selected == 0),
            scoring_projection_applied=bool(projection_max > 1.0e-9),
            scoring_projection_max=float(projection_max),
            selected_group_score_m3=float(group_score),
            first_move_group_count=len(group_records),
            tail_only_noop_candidate_count=max(len(passive_indices) - 1, 0),
        )


__all__ = ["FirstMoveFinitePolicyV122", "FirstMovePolicyResultV122"]
