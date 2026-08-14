"""V12.3 finite shooting policy: TFV primary, PFV soft, first-move rich.

This is intentionally still a finite policy.  Continuous coefficient optimisation is
not enabled until the existing authoritative SWMM gradient gate passes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .step2_control_basis_v60 import ControlBasisV60
from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_value_v123 import DualVolumeValueV123
from .step2_policy_v120 import RuntimeNormalizationV120, _project_executable_sequences_v120
from .step3_candidates_v123 import FirstMoveCandidateDesignV123, candidate_coefficients_v123
from .step3_objective_v123 import TFVPFVObjectiveV123, tfv_pfv_score_v123

V123_POLICY_CONTRACT = "PROJECT7_V123_FIRST_MOVE_TFV_PRIMARY_PFV_SOFT_POLICY_V1"


@dataclass(frozen=True)
class FirstMoveTFVPFVResultV123:
    settings: torch.Tensor
    candidate_valid: bool
    selected_candidate_index: int
    raw_candidate_count: int
    first_move_group_count: int
    tail_only_noop_candidate_count: int
    scenario_count: int
    predicted_delta_tfv_m3: float
    predicted_delta_pfv_m3: float
    tfv_risk_m3: float
    pfv_risk_m3: float
    pfv_soft_excess_m3: float
    pfv_penalty_m3_equivalent: float
    objective_score_m3_equivalent: float
    false_benefit_margin_m3: float
    scoring_projection_max: float


class FirstMoveTFVPFVPolicyV123:
    accepts_previous_requested_settings = True

    def __init__(
        self,
        *,
        model: DualVolumeValueV123,
        basis: ControlBasisV60,
        prepared: PreparedStaticV60,
        normalization: RuntimeNormalizationV120,
        objective: TFVPFVObjectiveV123,
        false_benefit_margin_m3: float,
        candidate_design: FirstMoveCandidateDesignV123 = FirstMoveCandidateDesignV123(),
        first_move_group_atol: float = 1.0e-7,
    ) -> None:
        basis.validate()
        normalization.validate()
        objective.validate()
        if not math.isfinite(float(false_benefit_margin_m3)) or false_benefit_margin_m3 < 0.0:
            raise ValueError("V123 false-benefit margin must be finite and non-negative")
        if not math.isfinite(float(first_move_group_atol)) or first_move_group_atol <= 0.0:
            raise ValueError("V123 first-move grouping tolerance must be positive")
        self.model = model
        self.basis = basis
        self.prepared = prepared
        self.normalization = normalization
        self.objective = objective
        self.false_benefit_margin_m3 = float(false_benefit_margin_m3)
        self.first_move_group_atol = float(first_move_group_atol)
        self._coefficients = candidate_coefficients_v123(basis, design=candidate_design)

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
    ) -> FirstMoveTFVPFVResultV123:
        if initial_state.ndim == 3 and initial_state.shape[0] == 1:
            initial_state = initial_state[0]
        if initial_state.ndim != 2 or rainfall_scenarios.ndim != 4:
            raise ValueError("V123 policy received incompatible state/rainfall shape")
        if fallback_settings.ndim != 3 or fallback_settings.shape[0] != 1:
            raise ValueError("V123 fallback settings must be [1,H,A]")
        if current_settings is None:
            raise ValueError("V123 policy requires current-setting readback")

        actuator_count = self.basis.grouping.actuator_count
        current = current_settings.reshape(-1)
        previous_target = None if previous_requested_settings is None else previous_requested_settings.reshape(-1)
        if current.numel() != actuator_count or (previous_target is not None and previous_target.numel() != actuator_count):
            raise ValueError("V123 actuator readback count mismatch")
        if previous_actuator_flow is None:
            previous_actuator_flow = torch.zeros(actuator_count, dtype=initial_state.dtype, device=initial_state.device)
        flow0 = previous_actuator_flow.reshape(-1)

        coeff = torch.as_tensor(self._coefficients, dtype=fallback_settings.dtype, device=fallback_settings.device)
        reference_one = fallback_settings
        reference_for_candidates = reference_one[:, None].expand(1, coeff.shape[0], -1, -1)
        candidate_one = self.basis.decode(reference_for_candidates, coeff[None])[0]

        frozen_delta = float(self.basis.contract.max_setting_delta_per_update)
        runtime_delta = frozen_delta
        if max_delta_per_update is not None:
            raw = torch.as_tensor(max_delta_per_update, dtype=torch.float32).reshape(-1)
            if not bool(torch.isfinite(raw).all()):
                raise ValueError("V123 runtime max delta is non-finite")
            runtime_delta = float(raw.max())
            if runtime_delta > frozen_delta + 1.0e-9:
                raise ValueError("V123 runtime max delta is looser than frozen basis")

        candidate_one, projection_max = _project_executable_sequences_v120(
            candidate_one,
            current_settings=current,
            previous_requested_settings=previous_target,
            min_settings=torch.as_tensor(self.basis.min_setting, dtype=candidate_one.dtype, device=candidate_one.device),
            max_settings=torch.as_tensor(self.basis.max_setting, dtype=candidate_one.dtype, device=candidate_one.device),
            max_delta_per_update=runtime_delta,
            control_block_steps=int(self.basis.horizon.control_block_steps),
        )
        if not torch.allclose(candidate_one[0], reference_one[0], rtol=0.0, atol=1.0e-7):
            raise RuntimeError("V123 engineering projection changed PASSIVE reference")

        scenarios = int(rainfall_scenarios.shape[0])
        state = initial_state[None].expand(scenarios, -1, -1)
        reference = reference_one.expand(scenarios, -1, -1)
        candidate = candidate_one[None].expand(scenarios, -1, -1, -1)
        flow = flow0[None].expand(scenarios, -1)
        output = self.model(
            self.normalization.state(state),
            self.normalization.rainfall(rainfall_scenarios),
            reference,
            candidate,
            self.normalization.flow(flow),
            self.prepared,
        )
        if output.delta_tfv_m3.shape != (scenarios, candidate_one.shape[0]):
            raise RuntimeError("V123 Value output shape drift")

        block = int(self.basis.horizon.control_block_steps)
        reference_first = reference_one[0, :block].mean(dim=0)
        first = candidate_one[:, :block].mean(dim=1)
        movement = torch.mean(torch.abs(first - reference_first[None]), dim=1)
        scored = tfv_pfv_score_v123(
            output.delta_tfv_m3,
            output.delta_pfv_m3,
            movement=movement,
            contract=self.objective,
        )
        score = scored["score_m3_equivalent"]
        tfv_risk = scored["tfv_risk_m3"]
        if abs(float(tfv_risk[0].detach())) > 1.0e-5 or abs(float(scored["pfv_risk_m3"][0].detach())) > 1.0e-5:
            raise RuntimeError("V123 PASSIVE lost exact-zero TFV/PFV value")

        rounded = torch.round(first / self.first_move_group_atol) * self.first_move_group_atol
        groups: dict[bytes, list[int]] = {}
        for index, row in enumerate(rounded.detach().cpu().numpy()):
            groups.setdefault(row.astype("float64").tobytes(), []).append(index)
        passive_indices = [
            index for index in range(candidate_one.shape[0])
            if bool(torch.allclose(first[index], reference_first, rtol=0.0, atol=self.first_move_group_atol))
        ]

        # Robustly summarize candidates that projection collapsed to the same executable
        # first move. The representative closest to the median combined objective is kept.
        records: list[tuple[int, float, float, bool]] = []
        for indices in groups.values():
            passive = any(index in passive_indices for index in indices)
            if passive:
                representative = next(index for index in indices if index in passive_indices)
                records.append((representative, 0.0, 0.0, True))
                continue
            values = score[indices]
            median_score = values.median()
            nearest = torch.argmin(torch.abs(values - median_score))
            representative = indices[int(nearest.item())]
            median_tfv_risk = float(tfv_risk[indices].median().detach())
            records.append((representative, float(median_score.detach()), median_tfv_risk, False))

        # Admission is evidence-aware: combined objective must improve and the TFV upper
        # risk itself must beat a TrainFit-derived false-benefit margin. PFV remains soft.
        eligible = [
            item for item in records
            if not item[3]
            and item[1] < -self.false_benefit_margin_m3
            and item[2] < -self.false_benefit_margin_m3
        ]
        if eligible:
            selected, _, _, _ = min(eligible, key=lambda item: item[1])
            valid = True
        else:
            selected, valid = 0, False

        def scalar(name: str) -> float:
            return float(scored[name][selected].detach()) if valid else 0.0

        return FirstMoveTFVPFVResultV123(
            settings=candidate_one[selected].detach(),
            candidate_valid=valid,
            selected_candidate_index=int(selected),
            raw_candidate_count=int(candidate_one.shape[0]),
            first_move_group_count=int(len(groups)),
            tail_only_noop_candidate_count=max(int(len(passive_indices)) - 1, 0),
            scenario_count=scenarios,
            predicted_delta_tfv_m3=scalar("delta_tfv_mean_m3"),
            predicted_delta_pfv_m3=scalar("delta_pfv_mean_m3"),
            tfv_risk_m3=scalar("tfv_risk_m3"),
            pfv_risk_m3=scalar("pfv_risk_m3"),
            pfv_soft_excess_m3=scalar("pfv_soft_excess_m3"),
            pfv_penalty_m3_equivalent=scalar("pfv_penalty_m3_equivalent"),
            objective_score_m3_equivalent=scalar("score_m3_equivalent"),
            false_benefit_margin_m3=float(self.false_benefit_margin_m3),
            scoring_projection_max=float(projection_max),
        )


__all__ = [
    "FirstMoveTFVPFVPolicyV123",
    "FirstMoveTFVPFVResultV123",
    "V123_POLICY_CONTRACT",
]
