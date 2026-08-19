"""Practical Project7 Step3: support-bounded H10 proposals ranked by receding-policy return.

This is the deployed Development control path. It intentionally does **not** call the historical
12x109 L-BFGS-B full-plan optimizer. The frozen base Step2 generates a direction through finite H10
single-actuator probes, a type-aware hydraulic proposal supplies an independent physical direction,
and the policy-return critic ranks at most three candidates under the exact first-action estimand.

Every candidate is represented as H10 candidate -> H350 HOLD for model scoring and support geometry.
Only the first H10 is executed before re-observation. This removes the two historical mismatches that
blocked the paper objective: continuous optimizer extrapolation outside training geometry and
persistent/open-loop action encoding for a receding first-action label.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np
import torch

from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    encode_policy_return_action_token,
    policy_return_margin_m3,
)
from .direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_learned_h10_probe_proposal,
    build_policy_return_candidate_portfolio,
    score_h10_first_action_targets,
)
from .step3_tfv_value_mpc_v11 import DirectTFVPolicyReturnMPCV11


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_H10_POLICY_RETURN_PORTFOLIO_V1"
)


@dataclass(frozen=True)
class DirectTFVMPCResultV12:
    settings: torch.Tensor
    optimized_candidate_settings: torch.Tensor
    predicted_delta_tfv_m3: float
    raw_optimized_predicted_delta_tfv_m3: float
    selected_source: str
    candidate_valid: bool
    admission_margin_m3: float
    admission_upper_bound_m3: float
    admission_margin_kind: str
    admission_passed: bool
    calibrated_admission_contract: str
    elapsed_seconds: float
    screened_facility_count: int
    predicted_beneficial_facility_count: int
    active_facility_count: int
    active_facility_ids: tuple[str, ...]
    active_facility_screening_scores_m3: tuple[float, ...]
    first_move_changed_facility_count: int
    maximum_support_ratio: float
    joint_sequence_support_quantile: str
    joint_sequence_first_block_l1: float
    joint_sequence_h120_l1: float
    joint_sequence_h120_total_variation_l1: float
    joint_sequence_support_max_ratio: float
    joint_sequence_support_binding: bool
    policy_return_predicted_delta_tfv_m3: float
    policy_return_margin_m3: float
    policy_return_upper_bound_m3: float
    policy_return_admission_passed: bool
    policy_return_admission_contract: str
    policy_return_estimand: str
    policy_return_parent_continuation_sha256: str
    policy_return_portfolio_contract: str
    policy_return_portfolio_candidate_count: int
    policy_return_portfolio_selected_source: str
    policy_return_portfolio_sources: tuple[str, ...]
    policy_return_portfolio_scores_m3: tuple[float, ...]
    policy_return_portfolio_upper_bounds_m3: tuple[float, ...]
    policy_return_portfolio_base_step2_scores_m3: tuple[float, ...]
    h10_probe_generator_contract: str
    h10_probe_count: int
    policy_mode: str
    policy_mode_contract: str
    refined_first_move_semantics: str
    refined_first_move_changed_facility_count: int
    refined_first_move_changed_facility_ids: tuple[str, ...]
    optimizer_success: bool = True
    optimizer_steps: int = 0
    optimizer_starts: int = 0
    gradient_norm: float = 0.0
    scipy_message: str = "NOT_USED_PRACTICAL_H10_PORTFOLIO"
    first_move_refiner_elapsed_seconds: float = 0.0
    first_move_refiner_steps: int = 0
    first_move_refinement_gain_m3: float = 0.0


class DirectTFVPolicyReturnPortfolioMPCV12(DirectTFVPolicyReturnMPCV11):
    """Rank small supported H10 alternatives by calibrated receding-policy TFV return."""

    policy_mode = "practical_direct_tfv_h10_policy_return_portfolio"
    policy_mode_contract = DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT

    def __init__(
        self,
        *,
        policy_return_admission: Mapping[str, Any],
        proposal_probe_chunk_size: int = 24,
        **kwargs: Any,
    ) -> None:
        admission = dict(policy_return_admission)
        if str(admission.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
            raise ValueError("practical portfolio requires a current policy-return admission artifact")
        if str(admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
            raise ValueError("practical portfolio admission was calibrated with another action encoding")
        if str(admission.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
            raise ValueError("practical portfolio requires calibration from the same candidate family")
        if int(admission.get("multi_candidate_query_set_count", 0)) <= 0:
            raise ValueError("practical portfolio requires same-prefix multi-candidate calibration")
        if int(proposal_probe_chunk_size) <= 0:
            raise ValueError("proposal_probe_chunk_size must be positive")
        super().__init__(policy_return_admission=admission, **kwargs)
        self.proposal_probe_chunk_size = int(proposal_probe_chunk_size)

    def _h10_supported_target(
        self,
        target: torch.Tensor,
        active_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float | bool | str]]:
        _, sequence_batch = encode_policy_return_action_token(
            active_target,
            target,
            horizon_steps=int(self.design.prediction_horizon_steps),
            first_action_steps=int(self.design.control_block_steps),
        )
        sequence = sequence_batch[0]
        # V6 trust-region machinery remains valuable, but it is now applied to the *actual H10 pulse
        # geometry* rather than a fictitious persistent H120/H360 command.
        sequence = self._contract_to_joint_sequence_support(sequence, active_target).detach()
        supported_target = sequence[0].detach()
        changed = int(torch.count_nonzero(torch.abs(supported_target - active_target) > 1.0e-7).item())
        diagnostics = self.joint_sequence_support_diagnostics(sequence, active_target)
        return supported_target, sequence, changed, diagnostics

    def _first_move_support_ratio(self, target: torch.Tensor, active_target: torch.Tensor) -> float:
        radius = torch.as_tensor(self.first_radius, dtype=target.dtype, device=target.device).clamp_min(1.0e-12)
        ratio = torch.abs(target - active_target) / radius
        ratio = torch.where(torch.as_tensor(self.first_radius, device=target.device) > 1.0e-12, ratio, torch.zeros_like(ratio))
        return float(torch.max(ratio).detach().cpu())

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("practical portfolio requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("practical portfolio requires previous_actuator_flow [1,109]")

        ceiling = int(self.active_support_ceiling())
        learned = build_learned_h10_probe_proposal(
            model=self.model,
            normalization=self.normalization,
            graph=self.graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active_target,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
            probe_chunk_size=self.proposal_probe_chunk_size,
        )
        portfolio = build_policy_return_candidate_portfolio(
            current_state=current_state,
            rainfall_scenarios=rainfall,
            active_target=active_target,
            learned_target=learned.target,
            graph=self.graph,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
        )

        evaluated: list[
            tuple[str, torch.Tensor, torch.Tensor, int, float, float, float, float, dict[str, float | bool | str]]
        ] = []
        seen: set[bytes] = set()
        for candidate in portfolio:
            target, sequence, changed, diagnostics = self._h10_supported_target(
                candidate.target, active_target
            )
            if changed <= 0:
                continue
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            score_t = self._score_policy_return_target(
                current_state=current_state,
                rainfall=rainfall,
                previous_actuator_flow=flow,
                active_target=active_target,
                candidate_target=target,
            )
            score = float(score_t.detach().cpu())
            margin = float(policy_return_margin_m3(self.policy_return_admission, changed))
            upper = float(score + margin)
            base_score = float(
                score_h10_first_action_targets(
                    model=self.model,
                    normalization=self.normalization,
                    graph=self.graph,
                    current_state=current_state,
                    rainfall_scenarios=rainfall,
                    previous_actuator_flow=flow,
                    active_target=active_target,
                    candidate_targets=target.reshape(1, 109),
                    probe_chunk_size=1,
                )[0].detach().cpu()
            )
            evaluated.append(
                (candidate.source, target, sequence, changed, score, margin, upper, base_score, diagnostics)
            )

        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        if not evaluated:
            return DirectTFVMPCResultV12(
                settings=hold,
                optimized_candidate_settings=hold,
                predicted_delta_tfv_m3=0.0,
                raw_optimized_predicted_delta_tfv_m3=0.0,
                selected_source="LATCH_PREVIOUS_TARGET_EMPTY_PRACTICAL_PORTFOLIO",
                candidate_valid=False,
                admission_margin_m3=0.0,
                admission_upper_bound_m3=0.0,
                admission_margin_kind="policy_return_h10_action_token",
                admission_passed=False,
                calibrated_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                elapsed_seconds=float(time.perf_counter() - started),
                screened_facility_count=109,
                predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
                active_facility_count=0,
                active_facility_ids=(),
                active_facility_screening_scores_m3=learned.facility_best_scores_m3,
                first_move_changed_facility_count=0,
                maximum_support_ratio=0.0,
                joint_sequence_support_quantile=str(self._sequence_support_quantile()),
                joint_sequence_first_block_l1=0.0,
                joint_sequence_h120_l1=0.0,
                joint_sequence_h120_total_variation_l1=0.0,
                joint_sequence_support_max_ratio=0.0,
                joint_sequence_support_binding=False,
                policy_return_predicted_delta_tfv_m3=0.0,
                policy_return_margin_m3=0.0,
                policy_return_upper_bound_m3=0.0,
                policy_return_admission_passed=False,
                policy_return_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
                policy_return_estimand=DIRECT_TFV_POLICY_RETURN_ESTIMAND,
                policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
                policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
                policy_return_portfolio_candidate_count=0,
                policy_return_portfolio_selected_source="HOLD",
                policy_return_portfolio_sources=(),
                policy_return_portfolio_scores_m3=(),
                policy_return_portfolio_upper_bounds_m3=(),
                policy_return_portfolio_base_step2_scores_m3=(),
                h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
                h10_probe_count=int(learned.probe_count),
                policy_mode=self.policy_mode,
                policy_mode_contract=self.policy_mode_contract,
                refined_first_move_semantics=DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
                refined_first_move_changed_facility_count=0,
                refined_first_move_changed_facility_ids=(),
            )

        selected = min(evaluated, key=lambda row: (row[6], row[4], row[0]))
        source, target, sequence, changed, score, margin, upper, _, diagnostics = selected
        passed = bool(upper < 0.0)
        executed = sequence if passed else hold
        changed_indices = torch.nonzero(torch.abs(target - active_target) > 1.0e-7).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=float(score) if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=float(score),
            selected_source=(
                f"DIRECT_TFV_POLICY_RETURN_PORTFOLIO::{source}"
                if passed
                else "LATCH_PREVIOUS_TARGET_PORTFOLIO_UPPER_BOUND_NONNEGATIVE"
            ),
            candidate_valid=passed,
            admission_margin_m3=float(margin),
            admission_upper_bound_m3=float(upper),
            admission_margin_kind="receding_policy_return_h10_action_token",
            admission_passed=passed,
            calibrated_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=109,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=int(changed),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=int(changed) if passed else 0,
            maximum_support_ratio=self._first_move_support_ratio(target, active_target),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(diagnostics["h120_total_variation_l1"]),
            joint_sequence_support_max_ratio=float(diagnostics["max_ratio"]),
            joint_sequence_support_binding=bool(diagnostics["binding"]),
            policy_return_predicted_delta_tfv_m3=float(score),
            policy_return_margin_m3=float(margin),
            policy_return_upper_bound_m3=float(upper),
            policy_return_admission_passed=passed,
            policy_return_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
            policy_return_estimand=DIRECT_TFV_POLICY_RETURN_ESTIMAND,
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=source if passed else "HOLD",
            policy_return_portfolio_sources=tuple(row[0] for row in evaluated),
            policy_return_portfolio_scores_m3=tuple(float(row[4]) for row in evaluated),
            policy_return_portfolio_upper_bounds_m3=tuple(float(row[6]) for row in evaluated),
            policy_return_portfolio_base_step2_scores_m3=tuple(float(row[7]) for row in evaluated),
            h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
            h10_probe_count=int(learned.probe_count),
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics=DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            refined_first_move_changed_facility_count=int(changed),
            refined_first_move_changed_facility_ids=changed_ids,
        )


__all__ = [
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT",
    "DirectTFVMPCResultV12",
    "DirectTFVPolicyReturnPortfolioMPCV12",
]
