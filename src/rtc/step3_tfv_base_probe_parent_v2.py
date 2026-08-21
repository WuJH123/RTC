"""Practical first-round parent pi0 with the current masked three-family H10 geometry.

The parent keeps the frozen 109-channel Step2 representation, but only the native supervisory mask
may change online. The current Wuhan contract therefore screens 82 control freedoms while retaining
all 109 channels in state/action tensors. Masked q95 support is read from the recomputed sequence-
support artifact; no Step1/base-Step2 retraining is required. Projected gradient is not part of the
current parent after the completed 82-control seen mechanism panel; it remains an archival ablation.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING, encode_policy_return_action_token
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_sequence_support import changed_facility_support_limit
from .step3_tfv_base_probe_parent import (
    DirectTFVBaseProbeParentMPC,
    DirectTFVBaseProbeParentResult,
)


DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT = (
    "PROJECT7_PRACTICAL_BASE_H10_THREE_FAMILY_PARENT_PI0_V4_82CONTROL_109REP"
)


class DirectTFVBaseHybridParentMPCV2(DirectTFVBaseProbeParentMPC):
    """Deterministic three-family pi0 on the native supervisory-control subspace."""

    policy_mode = "practical_base_h10_three_family_parent_pi0"
    policy_mode_contract = DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT

    def __init__(
        self,
        *args: Any,
        supervisory_mask: np.ndarray,
        projected_gradient_steps: int = 6,
        projected_gradient_step_fraction: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
        if mask.shape != (109,) or int(mask.sum()) <= 0:
            raise ValueError("base three-family parent requires a valid 109-channel supervisory mask")
        # Legacy CLI knobs are accepted only for backward-compatible launch scripts. They no longer
        # change the current paper-facing candidate family.
        if int(projected_gradient_steps) <= 0:
            raise ValueError("projected_gradient_steps must be positive")
        if not 0.0 < float(projected_gradient_step_fraction) <= 1.0:
            raise ValueError("projected_gradient_step_fraction must lie in (0,1]")
        self.supervisory_mask = mask
        self.supervisory_control_dimension = int(mask.sum())
        self.projected_gradient_steps = int(projected_gradient_steps)
        self.projected_gradient_step_fraction = float(projected_gradient_step_fraction)

    def optimize(self, **kwargs: Any) -> DirectTFVBaseProbeParentResult:
        started = time.perf_counter()
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        active = kwargs.get("active_target")
        if not isinstance(active, torch.Tensor) or active.shape != (109,):
            raise ValueError("base three-family parent requires active_target [109]")
        ceiling = changed_facility_support_limit(self.sequence_support, "q95")
        hybrid = build_hybrid_policy_return_portfolio(
            model=self.model,
            normalization=self.normalization,
            graph=self.graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
            probe_chunk_size=self.proposal_probe_chunk_size,
            supervisory_mask=self.supervisory_mask,
            include_projected_gradient_ablation=False,
        )
        learned = hybrid.learned_probe
        evaluated: list[
            tuple[str, torch.Tensor, torch.Tensor, int, float, dict[str, float | bool | str]]
        ] = []
        seen: set[bytes] = set()
        for proposal in hybrid.candidates:
            _, sequence_batch = encode_policy_return_action_token(
                active,
                proposal.target,
                horizon_steps=int(self.design.prediction_horizon_steps),
                first_action_steps=int(self.design.control_block_steps),
            )
            sequence = self._contract(sequence_batch[0], active).detach()
            target = sequence[0].detach()
            changed = int(torch.count_nonzero(torch.abs(target - active) > 1.0e-7).item())
            if changed <= 0:
                continue
            passive = torch.as_tensor(~self.supervisory_mask, dtype=torch.bool, device=target.device)
            if bool(torch.any(torch.abs(target[passive] - active[passive]) > 1.0e-7)):
                raise RuntimeError("three-family parent changed a passive 109-channel setting")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            score = float(
                score_h10_first_action_targets(
                    model=self.model,
                    normalization=self.normalization,
                    graph=self.graph,
                    current_state=current_state,
                    rainfall_scenarios=rainfall,
                    previous_actuator_flow=flow,
                    active_target=active,
                    candidate_targets=target.reshape(1, 109),
                    probe_chunk_size=1,
                )[0].detach().cpu()
            )
            evaluated.append(
                (proposal.source, target, sequence, changed, score, self._diagnostics(sequence, active))
            )

        hold = active.reshape(1, 109).expand(
            int(self.design.prediction_horizon_steps), -1
        ).detach()
        if not evaluated:
            selected_source = "HOLD"
            target = active
            sequence = hold
            changed = 0
            score = 0.0
            diagnostics: dict[str, float | bool | str] = {
                "quantile": self._sequence_support_quantile(),
                "first_block_l1": 0.0,
                "h120_l1": 0.0,
                "h120_total_variation_l1": 0.0,
                "max_ratio": 0.0,
                "binding": False,
            }
        else:
            selected_source, target, sequence, changed, score, diagnostics = min(
                evaluated, key=lambda row: (row[4], row[0])
            )
        passed = bool(evaluated and score < 0.0)
        executed = sequence if passed else hold
        changed_indices = (
            torch.nonzero(torch.abs(target - active) > 1.0e-7).reshape(-1).tolist()
            if evaluated
            else []
        )
        changed_ids = tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        radius = torch.as_tensor(self.first_radius, dtype=active.dtype, device=active.device)
        valid_radius = radius > 1.0e-12
        support_ratio = torch.where(
            valid_radius,
            torch.abs(target - active) / radius.clamp_min(1.0e-12),
            torch.zeros_like(radius),
        )
        return DirectTFVBaseProbeParentResult(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=score if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=score,
            selected_source=(
                f"BASE_H10_THREE_FAMILY_PARENT::{selected_source}"
                if passed
                else "LATCH_PREVIOUS_TARGET_BASE_H10_THREE_FAMILY_PARENT"
            ),
            candidate_valid=passed,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=score,
            admission_margin_kind="BASE_STEP2_DIRECTIONAL_SCORE_NO_CALIBRATED_EXECUTION_CLAIM",
            admission_passed=passed,
            calibrated_admission_contract="NONE_PARENT_PI0_ONLY",
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=self.supervisory_control_dimension,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=changed if evaluated else 0,
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=changed if passed else 0,
            maximum_support_ratio=(
                float(torch.max(support_ratio).detach().cpu()) if evaluated else 0.0
            ),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(
                diagnostics["h120_total_variation_l1"]
            ),
            joint_sequence_support_max_ratio=float(diagnostics["max_ratio"]),
            joint_sequence_support_binding=bool(diagnostics["binding"]),
            policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=selected_source if passed else "HOLD",
            policy_return_portfolio_sources=tuple(row[0] for row in evaluated),
            policy_return_portfolio_scores_m3=tuple(float(row[4]) for row in evaluated),
            policy_return_portfolio_upper_bounds_m3=tuple(float(row[4]) for row in evaluated),
            policy_return_portfolio_base_step2_scores_m3=tuple(float(row[4]) for row in evaluated),
            h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
            h10_probe_count=int(learned.probe_count),
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics=DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            refined_first_move_changed_facility_count=changed if evaluated else 0,
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
    "DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT",
    "DirectTFVBaseHybridParentMPCV2",
]
