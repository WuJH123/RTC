"""Frozen Practical parent policy for the first policy-return iteration.

The first exact paired-label round must have a deterministic causal continuation, but it does not need
an obsolete historical optimizer.  This parent uses the same frozen Step1/base-Step2 representation,
H10 probe generator, three-family candidate set and q95 support geometry as the eventual Practical
policy.  Before a policy-return critic exists it ranks candidates only by the frozen base Step2 H10
score and executes the best candidate when that score is negative; otherwise it HOLDs.

This parent is intentionally simple.  It is not presented as the final Proposed method and its score
is not used for final scientific claims.  Its purpose is to create Q^pi0 labels from a parent whose
state/action distribution is aligned with the paper method without depending on V12/L-BFGS-B or stale
open-loop admission hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import torch

from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING, encode_policy_return_action_token
from .direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_learned_h10_probe_proposal,
    build_policy_return_candidate_portfolio,
    score_h10_first_action_targets,
)
from .direct_tfv_sequence_support import (
    SEQUENCE_SUPPORT_METRICS,
    sequence_support_limit,
    validate_direct_tfv_sequence_support,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4


DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT = "PROJECT7_PRACTICAL_BASE_H10_PROBE_PARENT_PI0_V1"


@dataclass(frozen=True)
class DirectTFVBaseProbeParentResult:
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
    policy_return_portfolio_contract: str
    policy_return_portfolio_candidate_count: int
    policy_return_portfolio_selected_source: str
    policy_return_portfolio_sources: tuple[str, ...]
    policy_return_portfolio_scores_m3: tuple[float, ...]
    policy_return_portfolio_upper_bounds_m3: tuple[float, ...]
    policy_return_portfolio_base_step2_scores_m3: tuple[float, ...]
    h10_probe_generator_contract: str
    h10_probe_count: int
    policy_mode: str = "practical_base_h10_probe_parent_pi0"
    policy_mode_contract: str = DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT
    refined_first_move_semantics: str = DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
    refined_first_move_changed_facility_count: int = 0
    refined_first_move_changed_facility_ids: tuple[str, ...] = ()
    optimizer_success: bool = True
    optimizer_steps: int = 0
    optimizer_starts: int = 0
    gradient_norm: float = 0.0
    scipy_message: str = "NOT_USED_BASE_H10_PROBE_PARENT"
    first_move_refiner_elapsed_seconds: float = 0.0
    first_move_refiner_steps: int = 0
    first_move_refinement_gain_m3: float = 0.0


class DirectTFVBaseProbeParentMPC(DirectTFVRecedingMPCV4):
    """Deterministic causal pi0 with no learned return critic and no continuous optimizer."""

    policy_mode = "practical_base_h10_probe_parent_pi0"
    policy_mode_contract = DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
        proposal_probe_chunk_size: int = 24,
    ) -> None:
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            design=design,
        )
        validate_direct_tfv_sequence_support(sequence_support, actuator_ids=graph.actuator_ids)
        self.sequence_support = dict(sequence_support)
        if proposal_probe_chunk_size <= 0:
            raise ValueError("proposal_probe_chunk_size must be positive")
        self.proposal_probe_chunk_size = int(proposal_probe_chunk_size)

    def _sequence_support_quantile(self) -> str:
        return str(self.active_support_quantile_effective())

    def _geometry(self, sequence: torch.Tensor, active_target: torch.Tensor) -> dict[str, torch.Tensor]:
        block_steps = int(self.design.control_block_steps)
        blocks = sequence.reshape(-1, block_steps, 109).mean(dim=1)
        free = blocks[: int(self.design.free_control_blocks)]
        delta = free - active_target[None]
        previous = torch.cat((torch.zeros_like(delta[:1]), delta[:-1]), dim=0)
        return {
            "first_block_l1": torch.sum(torch.abs(delta[0])),
            "h120_l1": torch.sum(torch.abs(delta)),
            "h120_total_variation_l1": torch.sum(torch.abs(delta - previous)),
        }

    def _contract(self, sequence: torch.Tensor, active_target: torch.Tensor) -> torch.Tensor:
        geometry = self._geometry(sequence, active_target)
        quantile = self._sequence_support_quantile()
        scales = []
        for metric in SEQUENCE_SUPPORT_METRICS:
            mass = geometry[metric]
            limit = torch.as_tensor(
                sequence_support_limit(self.sequence_support, metric, quantile),
                dtype=sequence.dtype,
                device=sequence.device,
            )
            scales.append(torch.where(mass > 1.0e-12, limit / mass.clamp_min(1.0e-12), torch.ones_like(mass)))
        scale = torch.clamp(torch.min(torch.stack(scales)), 0.0, 1.0)
        hold = active_target[None].expand_as(sequence)
        return hold + scale * (sequence - hold)

    def _diagnostics(self, sequence: torch.Tensor, active_target: torch.Tensor) -> dict[str, float | bool | str]:
        geometry = self._geometry(sequence, active_target)
        quantile = self._sequence_support_quantile()
        result: dict[str, float | bool | str] = {"quantile": quantile}
        ratios = []
        for metric in SEQUENCE_SUPPORT_METRICS:
            mass = float(geometry[metric].detach().cpu())
            limit = float(sequence_support_limit(self.sequence_support, metric, quantile))
            ratio = 0.0 if limit <= 0.0 else mass / limit
            result[metric] = mass
            result[f"{metric}_limit"] = limit
            result[f"{metric}_ratio"] = ratio
            ratios.append(ratio)
        maximum = max(ratios, default=0.0)
        result["max_ratio"] = maximum
        result["binding"] = bool(maximum >= 0.999)
        return result

    def optimize(self, **kwargs: Any) -> DirectTFVBaseProbeParentResult:
        started = time.perf_counter()
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        active = kwargs.get("active_target")
        if not isinstance(active, torch.Tensor) or active.shape != (109,):
            raise ValueError("base-probe parent requires active_target [109]")
        ceiling = int(self.active_support_ceiling())
        learned = build_learned_h10_probe_proposal(
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
        )
        proposals = build_policy_return_candidate_portfolio(
            current_state=current_state,
            rainfall_scenarios=rainfall,
            active_target=active,
            learned_target=learned.target,
            graph=self.graph,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
        )
        evaluated: list[tuple[str, torch.Tensor, torch.Tensor, int, float, dict[str, float | bool | str]]] = []
        seen: set[bytes] = set()
        for proposal in proposals:
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
            evaluated.append((proposal.source, target, sequence, changed, score, self._diagnostics(sequence, active)))

        hold = active.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        if not evaluated:
            selected_source = "HOLD"
            target = active
            sequence = hold
            changed = 0
            score = 0.0
            diagnostics = {"quantile": self._sequence_support_quantile(), "first_block_l1": 0.0, "h120_l1": 0.0, "h120_total_variation_l1": 0.0, "max_ratio": 0.0, "binding": False}
        else:
            selected_source, target, sequence, changed, score, diagnostics = min(evaluated, key=lambda row: (row[4], row[0]))
        passed = bool(evaluated and score < 0.0)
        executed = sequence if passed else hold
        changed_indices = torch.nonzero(torch.abs(target - active) > 1.0e-7).reshape(-1).tolist() if evaluated else []
        changed_ids = tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        radius = torch.as_tensor(self.first_radius, dtype=active.dtype, device=active.device)
        valid_radius = radius > 1.0e-12
        support_ratio = torch.where(valid_radius, torch.abs(target - active) / radius.clamp_min(1.0e-12), torch.zeros_like(radius))
        return DirectTFVBaseProbeParentResult(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=score if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=score,
            selected_source=(f"BASE_H10_PROBE_PARENT::{selected_source}" if passed else "LATCH_PREVIOUS_TARGET_BASE_H10_PROBE_PARENT"),
            candidate_valid=passed,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=score,
            admission_margin_kind="BASE_STEP2_DIRECTIONAL_SCORE_NO_CALIBRATED_EXECUTION_CLAIM",
            admission_passed=passed,
            calibrated_admission_contract="NONE_PARENT_PI0_ONLY",
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=109,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=changed if evaluated else 0,
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=changed if passed else 0,
            maximum_support_ratio=float(torch.max(support_ratio).detach().cpu()) if evaluated else 0.0,
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(diagnostics["h120_total_variation_l1"]),
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
            refined_first_move_changed_facility_count=changed if evaluated else 0,
            refined_first_move_changed_facility_ids=changed_ids,
        )


__all__ = [
    "DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT",
    "DirectTFVBaseProbeParentMPC",
    "DirectTFVBaseProbeParentResult",
]
