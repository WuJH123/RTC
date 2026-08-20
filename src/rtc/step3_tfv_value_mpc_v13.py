"""Project7 Practical Step3: three-family H10 proposals ranked by exact-policy-return learning.

The deployed policy keeps the frozen 109-channel Step2 representation but permits candidate/reference
differences only on the native supervisory-control mask. The current Wuhan contract therefore uses
82 online control freedoms embedded in 109 model channels. Masked q95 support and the policy-return
critic are the final trust/admission layers. Projected gradient and historical L-BFGS-B are not part
of current online execution.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Mapping

import numpy as np
import torch

from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    policy_return_margin_m3,
)
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_sequence_support import (
    changed_facility_support_limit,
    validate_direct_tfv_sequence_support,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4
from .step3_tfv_value_mpc_v12 import (
    DirectTFVMPCResultV12,
    DirectTFVPolicyReturnPortfolioMPCV12,
)


DIRECT_TFV_HYBRID_POLICY_RETURN_STEP3_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_H10_POLICY_RETURN_THREE_FAMILY_V15_82CONTROL_109REP"
)


class DirectTFVHybridPolicyReturnMPCV13(DirectTFVPolicyReturnPortfolioMPCV12):
    """At-most-three-candidate H10 receding policy improvement on the masked control subspace."""

    policy_mode = "practical_direct_tfv_h10_three_family_policy_return_portfolio"
    policy_mode_contract = DIRECT_TFV_HYBRID_POLICY_RETURN_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        supervisory_mask: np.ndarray,
        policy_return_model: Any,
        policy_return_normalization: Any,
        policy_return_admission: Mapping[str, Any],
        policy_return_checkpoint_sha256: str,
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
        proposal_probe_chunk_size: int = 24,
        projected_gradient_steps: int = 6,
        projected_gradient_step_fraction: float = 0.25,
    ) -> None:
        DirectTFVRecedingMPCV4.__init__(
            self,
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            design=design,
        )
        mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
        if mask.shape != (109,) or int(mask.sum()) <= 0:
            raise ValueError("three-family policy-return MPC requires a valid supervisory mask")
        validate_direct_tfv_sequence_support(
            sequence_support,
            actuator_ids=graph.actuator_ids,
            supervisory_mask=mask,
        )
        self.sequence_support = dict(sequence_support)
        self.supervisory_mask = mask
        self.supervisory_control_dimension = int(mask.sum())
        admission = dict(policy_return_admission)
        if str(admission.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
            raise ValueError("three-family portfolio requires current H10 policy-return admission")
        if admission.get("development_only") is not True:
            raise ValueError("policy-return portfolio admission must be Development-only")
        if str(admission.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
            raise ValueError("policy-return portfolio admission has the wrong estimand")
        if str(admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
            raise ValueError("policy-return admission was calibrated with another action encoding")
        if str(admission.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
            raise ValueError("policy-return runtime requires calibration from the same three-family contract")
        if int(admission.get("multi_candidate_query_set_count", 0)) <= 0:
            raise ValueError("three-family portfolio requires same-prefix multi-candidate calibration")
        if str(admission.get("policy_return_checkpoint_sha256", "")).lower() != str(policy_return_checkpoint_sha256).lower():
            raise ValueError("policy-return admission was calibrated on another critic")
        parent = str(admission.get("continuation_policy_sha256", "")).lower()
        if len(parent) != 64:
            raise ValueError("policy-return admission lacks frozen parent-policy lineage")
        if admission.get("generic_d3_floor_controls_execution") is not False:
            raise ValueError("generic D3 admission cannot control three-family execution")
        if admission.get("open_loop_first_move_margin_controls_execution") is not False:
            raise ValueError("open-loop first-move margin cannot control three-family execution")
        if int(proposal_probe_chunk_size) <= 0:
            raise ValueError("proposal_probe_chunk_size must be positive")
        # Retained launch compatibility only. The current portfolio never invokes projected gradient.
        if int(projected_gradient_steps) <= 0:
            raise ValueError("projected_gradient_steps must be positive")
        if not 0.0 < float(projected_gradient_step_fraction) <= 1.0:
            raise ValueError("projected_gradient_step_fraction must lie in (0,1]")
        self.policy_return_model = policy_return_model
        self.policy_return_model.eval()
        self.policy_return_normalization = policy_return_normalization
        self.policy_return_admission = admission
        self.policy_return_checkpoint_sha256 = str(policy_return_checkpoint_sha256).lower()
        self.policy_return_parent_continuation_sha256 = parent
        self.proposal_probe_chunk_size = int(proposal_probe_chunk_size)
        self.projected_gradient_steps = int(projected_gradient_steps)
        self.projected_gradient_step_fraction = float(projected_gradient_step_fraction)

    def _hold_result(
        self,
        *,
        started: float,
        active_target: torch.Tensor,
        learned: Any,
    ) -> DirectTFVMPCResultV12:
        result = super()._hold_result(
            started=started,
            active_target=active_target,
            learned=learned,
        )
        return replace(
            result,
            screened_facility_count=self.supervisory_control_dimension,
            policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            scipy_message="NOT_USED_CURRENT_THREE_FAMILY_FINITE_PORTFOLIO",
        )

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("three-family practical portfolio requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("three-family practical portfolio requires previous_actuator_flow [1,109]")

        ceiling = changed_facility_support_limit(self.sequence_support, "q95")
        hybrid = build_hybrid_policy_return_portfolio(
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
            supervisory_mask=self.supervisory_mask,
            include_projected_gradient_ablation=False,
        )
        learned = hybrid.learned_probe
        evaluated: list[tuple] = []
        seen: set[bytes] = set()
        passive = torch.as_tensor(~self.supervisory_mask, dtype=torch.bool, device=active_target.device)
        for proposal in hybrid.candidates:
            target, sequence, changed, diagnostics = self._h10_supported_target(
                proposal.target, active_target
            )
            if changed <= 0:
                continue
            if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("three-family policy-return candidate changed a passive setting channel")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            score = float(
                self._score_policy_return_target(
                    current_state=current_state,
                    rainfall=rainfall,
                    previous_actuator_flow=flow,
                    active_target=active_target,
                    candidate_target=target,
                ).detach().cpu()
            )
            margin = float(policy_return_margin_m3(self.policy_return_admission, changed))
            upper = score + margin
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
                (
                    proposal.source,
                    target,
                    sequence,
                    changed,
                    score,
                    margin,
                    upper,
                    base_score,
                    diagnostics,
                )
            )

        if not evaluated:
            return self._hold_result(
                started=started,
                active_target=active_target,
                learned=learned,
            )

        selected = min(evaluated, key=lambda row: (row[6], row[4], row[0]))
        source, target, sequence, changed, score, margin, upper, _, diagnostics = selected
        passed = bool(upper < 0.0)
        hold = active_target.reshape(1, 109).expand(
            int(self.design.prediction_horizon_steps), -1
        ).detach()
        executed = sequence if passed else hold
        changed_indices = torch.nonzero(
            torch.abs(target - active_target) > 1.0e-7
        ).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=score if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=score,
            selected_source=(
                f"DIRECT_TFV_POLICY_RETURN_THREE_FAMILY::{source}"
                if passed
                else "LATCH_PREVIOUS_TARGET_THREE_FAMILY_UPPER_BOUND_NONNEGATIVE"
            ),
            candidate_valid=passed,
            admission_margin_m3=margin,
            admission_upper_bound_m3=upper,
            admission_margin_kind="receding_policy_return_h10_action_token",
            admission_passed=passed,
            calibrated_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=self.supervisory_control_dimension,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=changed,
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=changed if passed else 0,
            maximum_support_ratio=self._first_move_support_ratio(target, active_target),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(
                diagnostics["h120_total_variation_l1"]
            ),
            joint_sequence_support_max_ratio=float(diagnostics["max_ratio"]),
            joint_sequence_support_binding=bool(diagnostics["binding"]),
            policy_return_predicted_delta_tfv_m3=score,
            policy_return_margin_m3=margin,
            policy_return_upper_bound_m3=upper,
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
            refined_first_move_changed_facility_count=changed,
            refined_first_move_changed_facility_ids=changed_ids,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            gradient_norm=0.0,
            scipy_message="NOT_USED_CURRENT_THREE_FAMILY_FINITE_PORTFOLIO",
            first_move_refiner_steps=0,
            first_move_refinement_gain_m3=0.0,
        )


__all__ = [
    "DIRECT_TFV_HYBRID_POLICY_RETURN_STEP3_CONTRACT",
    "DirectTFVHybridPolicyReturnMPCV13",
]
