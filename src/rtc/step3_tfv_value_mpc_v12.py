"""Practical Project7 Step3: finite H10 proposals ranked by receding-policy return.

This deployed Development controller is deliberately independent of the historical V5-V12 optimizer
admission chain. It inherits only the frozen base Step2/action-support geometry from V4; it never
calls V4's L-BFGS-B ``optimize`` implementation. Old V12 policy/first-move admissions are therefore
not prerequisites for the Practical online policy and cannot block it through stale lineage hashes.
They remain relevant only when the historical V12 policy is explicitly used as offline parent pi0.

At each decision all 109 facilities are screened with support-bounded H10 probes, at most three
first-action candidates are formed, actual H10-pulse joint support is enforced, and a separately
trained policy-return critic ranks/admit candidates under the exact receding first-action estimand.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

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
from .direct_tfv_sequence_support import (
    SEQUENCE_SUPPORT_METRICS,
    sequence_support_limit,
    validate_direct_tfv_sequence_support,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_H10_POLICY_RETURN_PORTFOLIO_V2_DECOUPLED"
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


class DirectTFVPolicyReturnPortfolioMPCV12(DirectTFVRecedingMPCV4):
    """Small support-aware H10 policy-improvement search with no online continuous optimizer."""

    policy_mode = "practical_direct_tfv_h10_policy_return_portfolio"
    policy_mode_contract = DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        policy_return_model: Any,
        policy_return_normalization: Any,
        policy_return_admission: Mapping[str, Any],
        policy_return_checkpoint_sha256: str,
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
        admission = dict(policy_return_admission)
        if str(admission.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
            raise ValueError("practical portfolio requires current H10 policy-return admission")
        if admission.get("development_only") is not True:
            raise ValueError("practical portfolio admission must be Development-only")
        if str(admission.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
            raise ValueError("practical portfolio admission has the wrong estimand")
        if str(admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
            raise ValueError("practical portfolio admission was calibrated with another action encoding")
        if str(admission.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
            raise ValueError("practical portfolio requires calibration from the same candidate family")
        if int(admission.get("multi_candidate_query_set_count", 0)) <= 0:
            raise ValueError("practical portfolio requires same-prefix multi-candidate calibration")
        if str(admission.get("policy_return_checkpoint_sha256", "")).lower() != str(policy_return_checkpoint_sha256).lower():
            raise ValueError("practical portfolio admission was calibrated on another critic")
        parent = str(admission.get("continuation_policy_sha256", "")).lower()
        if len(parent) != 64:
            raise ValueError("practical portfolio admission lacks frozen parent-policy lineage")
        if admission.get("generic_d3_floor_controls_execution") is not False:
            raise ValueError("generic D3 admission cannot control practical execution")
        if admission.get("open_loop_first_move_margin_controls_execution") is not False:
            raise ValueError("open-loop first-move margin cannot control practical execution")
        if int(proposal_probe_chunk_size) <= 0:
            raise ValueError("proposal_probe_chunk_size must be positive")
        self.policy_return_model = policy_return_model
        self.policy_return_model.eval()
        self.policy_return_normalization = policy_return_normalization
        self.policy_return_admission = admission
        self.policy_return_checkpoint_sha256 = str(policy_return_checkpoint_sha256).lower()
        self.policy_return_parent_continuation_sha256 = parent
        self.proposal_probe_chunk_size = int(proposal_probe_chunk_size)

    def _sequence_support_quantile(self) -> str:
        quantile = str(self.active_support_quantile_effective())
        if quantile not in {"q90", "q95", "q99"}:
            raise ValueError(f"unsupported Practical sequence-support quantile: {quantile}")
        return quantile

    def _joint_sequence_geometry_torch(
        self, sequence: torch.Tensor, active_target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if tuple(sequence.shape) != (self.design.prediction_horizon_steps, 109):
            raise ValueError("Practical sequence must be [H72,109]")
        if tuple(active_target.shape) != (109,):
            raise ValueError("Practical active target must contain 109 settings")
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

    def _contract_to_joint_sequence_support(
        self, sequence: torch.Tensor, active_target: torch.Tensor
    ) -> torch.Tensor:
        geometry = self._joint_sequence_geometry_torch(sequence, active_target)
        quantile = self._sequence_support_quantile()
        scales: list[torch.Tensor] = []
        for metric in SEQUENCE_SUPPORT_METRICS:
            mass = geometry[metric]
            limit = torch.as_tensor(
                sequence_support_limit(self.sequence_support, metric, quantile),
                dtype=sequence.dtype,
                device=sequence.device,
            )
            safe_scale = torch.where(
                mass > 1.0e-12,
                limit / mass.clamp_min(1.0e-12),
                torch.ones_like(mass),
            )
            scales.append(safe_scale)
        scale = torch.clamp(torch.min(torch.stack(scales)), min=0.0, max=1.0)
        hold = active_target[None].expand_as(sequence)
        return hold + scale * (sequence - hold)

    def joint_sequence_support_diagnostics(
        self, sequence: torch.Tensor, active_target: torch.Tensor
    ) -> dict[str, float | bool | str]:
        geometry = self._joint_sequence_geometry_torch(sequence, active_target)
        quantile = self._sequence_support_quantile()
        result: dict[str, float | bool | str] = {"quantile": quantile}
        ratios: list[float] = []
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

    def _normalize_return_state(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.policy_return_normalization.state_mean, dtype=value.dtype, device=value.device)
        std = torch.as_tensor(self.policy_return_normalization.state_std, dtype=value.dtype, device=value.device).clamp_min(1.0e-6)
        return (value - mean) / std

    def _normalize_return_rain(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.policy_return_normalization.rainfall_mean, dtype=value.dtype, device=value.device)
        std = torch.as_tensor(self.policy_return_normalization.rainfall_std, dtype=value.dtype, device=value.device).clamp_min(1.0e-6)
        return (value - mean) / std

    def _normalize_return_flow(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.policy_return_normalization.flow_mean, dtype=value.dtype, device=value.device)
        std = torch.as_tensor(self.policy_return_normalization.flow_std, dtype=value.dtype, device=value.device).clamp_min(1.0e-6)
        return (value - mean) / std

    def _score_policy_return_target(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
        candidate_target: torch.Tensor,
    ) -> torch.Tensor:
        if rainfall.ndim != 4 or int(rainfall.shape[0]) < 2:
            raise ValueError("policy-return critic requires [scenario,H,node,feature] causal rainfall")
        scenarios = int(rainfall.shape[0])
        horizon = int(self.design.prediction_horizon_steps)
        state = self._normalize_return_state(current_state).expand(scenarios, -1, -1)
        rain = self._normalize_return_rain(rainfall)
        flow = self._normalize_return_flow(previous_actuator_flow).expand(scenarios, -1)
        active = active_target.reshape(1, 109).expand(scenarios, -1)
        target = candidate_target.reshape(1, 109).expand(scenarios, -1)
        reference, candidate = encode_policy_return_action_token(
            active,
            target,
            horizon_steps=horizon,
            first_action_steps=int(self.design.control_block_steps),
        )
        output = self.policy_return_model(
            current_state=state,
            rainfall=rain,
            reference_settings=reference,
            candidate_settings=candidate,
            previous_actuator_flow=flow,
            actuator_upstream=torch.as_tensor(self.graph.actuator_upstream, dtype=torch.long, device=state.device),
            actuator_downstream=torch.as_tensor(self.graph.actuator_downstream, dtype=torch.long, device=state.device),
            actuator_physics=torch.as_tensor(self.graph.actuator_physics, dtype=state.dtype, device=state.device),
        )
        scores = output.total_delta_tfv_m3
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("policy-return critic produced non-finite values")
        return scores.mean()

    def _h10_supported_target(
        self, target: torch.Tensor, active_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float | bool | str]]:
        _, sequence_batch = encode_policy_return_action_token(
            active_target,
            target,
            horizon_steps=int(self.design.prediction_horizon_steps),
            first_action_steps=int(self.design.control_block_steps),
        )
        sequence = self._contract_to_joint_sequence_support(sequence_batch[0], active_target).detach()
        supported_target = sequence[0].detach()
        changed = int(torch.count_nonzero(torch.abs(supported_target - active_target) > 1.0e-7).item())
        return supported_target, sequence, changed, self.joint_sequence_support_diagnostics(sequence, active_target)

    def _first_move_support_ratio(self, target: torch.Tensor, active_target: torch.Tensor) -> float:
        raw_radius = torch.as_tensor(self.first_radius, dtype=target.dtype, device=target.device)
        valid = raw_radius > 1.0e-12
        radius = raw_radius.clamp_min(1.0e-12)
        ratio = torch.where(valid, torch.abs(target - active_target) / radius, torch.zeros_like(radius))
        return float(torch.max(ratio).detach().cpu())

    def _hold_result(
        self,
        *,
        started: float,
        active_target: torch.Tensor,
        learned: Any,
    ) -> DirectTFVMPCResultV12:
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
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
            joint_sequence_support_quantile=self._sequence_support_quantile(),
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
        evaluated: list[tuple] = []
        seen: set[bytes] = set()
        for proposal in portfolio:
            target, sequence, changed, diagnostics = self._h10_supported_target(proposal.target, active_target)
            if changed <= 0:
                continue
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
            evaluated.append((proposal.source, target, sequence, changed, score, margin, upper, base_score, diagnostics))

        if not evaluated:
            return self._hold_result(started=started, active_target=active_target, learned=learned)

        selected = min(evaluated, key=lambda row: (row[6], row[4], row[0]))
        source, target, sequence, changed, score, margin, upper, _, diagnostics = selected
        passed = bool(upper < 0.0)
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        executed = sequence if passed else hold
        changed_indices = torch.nonzero(torch.abs(target - active_target) > 1.0e-7).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=score if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=score,
            selected_source=(f"DIRECT_TFV_POLICY_RETURN_PORTFOLIO::{source}" if passed else "LATCH_PREVIOUS_TARGET_PORTFOLIO_UPPER_BOUND_NONNEGATIVE"),
            candidate_valid=passed,
            admission_margin_m3=margin,
            admission_upper_bound_m3=upper,
            admission_margin_kind="receding_policy_return_h10_action_token",
            admission_passed=passed,
            calibrated_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=109,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=changed,
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=changed if passed else 0,
            maximum_support_ratio=self._first_move_support_ratio(target, active_target),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(diagnostics["h120_total_variation_l1"]),
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
        )


__all__ = [
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT",
    "DirectTFVMPCResultV12",
    "DirectTFVPolicyReturnPortfolioMPCV12",
]
