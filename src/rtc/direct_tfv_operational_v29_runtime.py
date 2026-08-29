"""Project7 V29 regime-balanced q95-supported Development runtime.

V29 keeps the frozen Step1/Step2/Q27 backbone, the q95 executable manifold and
the established V23/V27 candidate families. It adds (1) a Train-CV-selected
continuous regime correction to Q27 and (2) one Auto-RBC shadow whose K-sparse
projection is ranked by hydraulic utility. Neither stress nor the rule shadow
has unconditional ACTION authority: every executable action still competes
against HOLD=0 under the same learned value surface.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .closed_loop import CausalObservation, ControllerAction
from .direct_tfv_operational_v28_runtime import (
    DirectTFVAuthoritativeV28Controller,
    DirectTFVOperationalV28MPC,
    V28Candidate,
    build_operational_v28_controller,
)
from .direct_tfv_sequence_support import changed_facility_support_limit
from .direct_tfv_v28_residual_value import build_v28_residual_features
from .direct_tfv_v29_auto_rbc_shadow import (
    V29_AUTO_RBC_UTILITY_CONTRACT,
    V29_AUTO_RBC_UTILITY_SOURCE,
    build_auto_rbc_hydraulic_utility_proposal,
)
from .direct_tfv_v29_regime_value import (
    V29_REGIME_VALUE_CONTRACT,
    V29RegimeValueModel,
    build_v29_regime_features,
    load_v29_regime_value_model,
)
from .runtime_controller_guard import ContinuityGuardController
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12

V29_OPERATIONAL_RUNTIME_CONTRACT = "PROJECT7_OPERATIONAL_DEVELOPMENT_V29_REGIME_BALANCED_MILD_CONTROL_V1"
V29_SELECTION_CONTRACT = "PROJECT7_V29_ARGMIN_REGIME_BALANCED_Q29_Q95_SUPPORTED_PLUS_HOLD_ZERO_V1"
V29_PORTFOLIO_CONTRACT = "PROJECT7_V29_V23_PORTFOLIO_PLUS_HYDRAULIC_UTILITY_RBC_SHADOW_V1"
V29_TFV_VALUE_ESTIMAND = (
    "SYSTEM_WIDE_CUMULATIVE_TFV_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _target_sha(target: torch.Tensor) -> str:
    array = target.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class DirectTFVOperationalV29MPC(DirectTFVOperationalV28MPC):
    """Q27 + regime residual, with one hydraulic-utility Auto-RBC candidate."""

    policy_mode = "operational_development_v29_regime_balanced_mild_control"
    policy_mode_contract = V29_OPERATIONAL_RUNTIME_CONTRACT

    def __init__(
        self,
        *,
        regime_value_model: V29RegimeValueModel,
        v29_lineage: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.regime_value_model = regime_value_model
        self.v29_lineage = dict(v29_lineage)
        self.last_v29_telemetry: dict[str, Any] = {}

    def _score_v29_candidate(self, candidate: V28Candidate) -> V28Candidate:
        v28_feature = build_v28_residual_features(
            q27_score_m3=float(candidate.q27_score_m3),
            q95_scale=float(candidate.q95_scale),
            q95_max_ratio=float(candidate.q95_max_ratio),
            q95_binding=bool(candidate.q95_binding),
            raw_first_move_l1=float(candidate.raw_first_move_l1),
            supported_first_move_l1=float(candidate.supported_first_move_l1),
            raw_to_supported_first_move_l1=float(candidate.raw_to_supported_first_move_l1),
            raw_to_supported_h120_l1=float(candidate.raw_to_supported_h120_l1),
            raw_to_supported_tv_l1=float(candidate.raw_to_supported_tv_l1),
            changed_facility_count=int(candidate.changed_facility_count),
            network_stress_q75=float(candidate.network_stress_q75),
            rain_level=float(candidate.rain_level),
            strong_storm_blend=float(candidate.strong_storm_blend),
            candidate_source=str(candidate.source),
        )
        regime_feature = build_v29_regime_features(v28_feature)
        residual = float(self.regime_value_model.predict_residual_m3(regime_feature))
        candidate.residual_m3 = residual
        candidate.q28_score_m3 = float(candidate.q27_score_m3) + residual
        return candidate

    def _make_v28_candidate(self, **kwargs: Any) -> V28Candidate | None:
        candidate = super()._make_v28_candidate(**kwargs)
        return None if candidate is None else self._score_v29_candidate(candidate)

    def _utility_candidate(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        flow: torch.Tensor,
        active_target: torch.Tensor,
    ) -> tuple[V28Candidate | None, dict[str, Any]]:
        ceiling = int(changed_facility_support_limit(self.sequence_support, "q95"))
        proposal = build_auto_rbc_hydraulic_utility_proposal(
            graph=self.graph,
            current_state=current_state,
            active_target=active_target,
            supervisory_mask=self.supervisory_mask,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
        )
        if int(proposal.retained_changed_facility_count) <= 0:
            return None, {
                "present": False,
                "raw_changed_facility_count": int(proposal.raw_changed_facility_count),
                "retained_changed_facility_count": 0,
                "retained_utility_fraction": float(proposal.retained_utility_fraction),
            }
        candidate = self._make_v28_candidate(
            current_state=current_state,
            rainfall=rainfall,
            flow=flow,
            active_target=active_target,
            source=proposal.source,
            raw_target=proposal.target,
        )
        return candidate, {
            "present": candidate is not None,
            "raw_changed_facility_count": int(proposal.raw_changed_facility_count),
            "retained_changed_facility_count": int(proposal.retained_changed_facility_count),
            "total_hydraulic_utility": float(proposal.total_hydraulic_utility),
            "retained_hydraulic_utility": float(proposal.retained_hydraulic_utility),
            "retained_utility_fraction": float(proposal.retained_utility_fraction),
            "mean_upstream_fill": float(proposal.mean_upstream_fill),
            "max_downstream_fill": float(proposal.max_downstream_fill),
        }

    def _relabel_base_result(
        self,
        result: DirectTFVMPCResultV12,
        *,
        selected_score: float,
    ) -> DirectTFVMPCResultV12:
        action = bool(result.candidate_valid)
        source = str(result.policy_return_portfolio_selected_source) if action else "HOLD"
        return replace(
            result,
            selected_source=(
                f"OPERATIONAL_V29::{source}"
                if action
                else "LATCH_PREVIOUS_TARGET_OPERATIONAL_V29_Q29_NONNEGATIVE"
            ),
            calibrated_admission_contract=V29_SELECTION_CONTRACT,
            policy_return_admission_contract=V29_SELECTION_CONTRACT,
            policy_return_portfolio_contract=V29_PORTFOLIO_CONTRACT,
            policy_return_estimand=V29_TFV_VALUE_ESTIMAND,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            scipy_message="V29_STRUCTURED_TELEMETRY_IN_CONTROLLER_DIAGNOSTICS",
            tfv_value_gate_reason="V29_Q29_NEGATIVE" if action else "V29_Q29_NONNEGATIVE_HOLD",
            raw_optimized_predicted_delta_tfv_m3=float(selected_score),
        )

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V29 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]

        base_result = super().optimize(**kwargs)
        base_telemetry = dict(getattr(self, "last_v28_telemetry", {}))
        base_payloads = [dict(value) for value in base_telemetry.get("candidates", [])]
        for payload in base_payloads:
            payload["q29_score_m3"] = float(payload.get("q28_score_m3", 0.0))
            payload["q29_residual_m3"] = float(payload.get("q27_residual_m3", 0.0))
        base_best_score = float(base_result.raw_optimized_predicted_delta_tfv_m3)

        utility, utility_diag = self._utility_candidate(
            current_state=current_state,
            rainfall=rainfall,
            flow=flow,
            active_target=active_target,
        )
        duplicate = False
        utility_selected = False
        utility_payload: dict[str, Any] | None = None
        if utility is not None:
            supported_sha = _target_sha(utility.supported_target)
            duplicate = any(
                str(payload.get("supported_target_sha256", "")).lower() == supported_sha.lower()
                for payload in base_payloads
            )
            utility_payload = utility.telemetry()
            utility_payload["q29_score_m3"] = float(utility.q28_score_m3)
            utility_payload["q29_residual_m3"] = float(utility.residual_m3)
            utility_payload["candidate_executed"] = False

        result = self._relabel_base_result(base_result, selected_score=base_best_score)
        if (
            utility is not None
            and not duplicate
            and float(utility.q28_score_m3) < 0.0
            and float(utility.q28_score_m3) < float(base_best_score)
        ):
            utility_selected = True
            if utility_payload is not None:
                utility_payload["candidate_executed"] = True
            changed_indices = torch.nonzero(
                torch.abs(utility.supported_target - active_target) > 1.0e-7
            ).reshape(-1).tolist()
            changed_ids = tuple(str(self.graph.actuator_ids[int(index)]) for index in changed_indices)
            support_geometry = self._joint_sequence_geometry_torch(
                utility.supported_sequence,
                active_target,
            )
            sources = tuple(base_result.policy_return_portfolio_sources) + (utility.source,)
            scores = tuple(base_result.policy_return_portfolio_scores_m3) + (
                float(utility.q28_score_m3),
            )
            base_scores = tuple(base_result.policy_return_portfolio_base_step2_scores_m3) + (
                float(utility.q27_score_m3),
            )
            result = replace(
                base_result,
                settings=utility.supported_sequence,
                optimized_candidate_settings=utility.supported_sequence,
                predicted_delta_tfv_m3=float(utility.q28_score_m3),
                raw_optimized_predicted_delta_tfv_m3=float(utility.q28_score_m3),
                selected_source=f"OPERATIONAL_V29::{utility.source}",
                candidate_valid=True,
                admission_upper_bound_m3=float(utility.q28_score_m3),
                admission_passed=True,
                calibrated_admission_contract=V29_SELECTION_CONTRACT,
                active_facility_count=int(utility.changed_facility_count),
                active_facility_ids=changed_ids,
                first_move_changed_facility_count=int(utility.changed_facility_count),
                maximum_support_ratio=self._first_move_support_ratio(
                    utility.supported_target,
                    active_target,
                ),
                joint_sequence_support_quantile="q95",
                joint_sequence_first_block_l1=float(
                    support_geometry["first_block_l1"].detach().cpu()
                ),
                joint_sequence_h120_l1=float(support_geometry["h120_l1"].detach().cpu()),
                joint_sequence_h120_total_variation_l1=float(
                    support_geometry["h120_total_variation_l1"].detach().cpu()
                ),
                joint_sequence_support_max_ratio=float(utility.q95_max_ratio),
                joint_sequence_support_binding=bool(utility.q95_binding),
                policy_return_predicted_delta_tfv_m3=float(utility.q28_score_m3),
                policy_return_upper_bound_m3=float(utility.q28_score_m3),
                policy_return_admission_passed=True,
                policy_return_admission_contract=V29_SELECTION_CONTRACT,
                policy_return_estimand=V29_TFV_VALUE_ESTIMAND,
                policy_return_portfolio_contract=V29_PORTFOLIO_CONTRACT,
                policy_return_portfolio_candidate_count=int(
                    base_result.policy_return_portfolio_candidate_count + 1
                ),
                policy_return_portfolio_selected_source=utility.source,
                policy_return_portfolio_sources=sources,
                policy_return_portfolio_scores_m3=scores,
                policy_return_portfolio_upper_bounds_m3=scores,
                policy_return_portfolio_base_step2_scores_m3=base_scores,
                policy_mode=self.policy_mode,
                policy_mode_contract=self.policy_mode_contract,
                refined_first_move_changed_facility_count=int(utility.changed_facility_count),
                refined_first_move_changed_facility_ids=changed_ids,
                scipy_message="V29_STRUCTURED_TELEMETRY_IN_CONTROLLER_DIAGNOSTICS",
                tfv_value_prediction_m3=float(utility.q28_score_m3),
                tfv_value_upper_bound_m3=float(utility.q28_score_m3),
                tfv_value_admission_passed=True,
                tfv_value_gate_reason="V29_Q29_NEGATIVE",
            )

        candidate_payloads = list(base_payloads)
        if utility_payload is not None and not duplicate:
            candidate_payloads.append(utility_payload)
        self.last_v29_telemetry = {
            "contract": V29_OPERATIONAL_RUNTIME_CONTRACT,
            "selection_contract": V29_SELECTION_CONTRACT,
            "regime_value_contract": V29_REGIME_VALUE_CONTRACT,
            "utility_shadow_contract": V29_AUTO_RBC_UTILITY_CONTRACT,
            "action_class": "ACTION" if result.candidate_valid else "HOLD",
            "selected_source": str(result.policy_return_portfolio_selected_source),
            "selected_q29_score_m3": float(result.raw_optimized_predicted_delta_tfv_m3),
            "selected_regime_residual_m3": float(
                next(
                    (
                        payload.get("q29_residual_m3", 0.0)
                        for payload in candidate_payloads
                        if str(payload.get("candidate_source", ""))
                        == str(result.policy_return_portfolio_selected_source)
                    ),
                    0.0,
                )
            ),
            "utility_shadow_present": bool(utility_diag.get("present", False)),
            "utility_shadow_duplicate": bool(duplicate),
            "utility_shadow_selected": bool(utility_selected),
            "utility_shadow_raw_changed_facility_count": int(
                utility_diag.get("raw_changed_facility_count", 0)
            ),
            "utility_shadow_retained_changed_facility_count": int(
                utility_diag.get("retained_changed_facility_count", 0)
            ),
            "utility_shadow_retained_utility_fraction": float(
                utility_diag.get("retained_utility_fraction", 0.0)
            ),
            "candidates": candidate_payloads,
            "q95_mandatory": True,
            "return_period_used_as_rule_or_feature": False,
            "event_duration_used_as_rule_or_feature": False,
        }
        return result


class DirectTFVAuthoritativeV29Controller(DirectTFVAuthoritativeV28Controller):
    """Write V29 structured telemetry to the authoritative decision JSONL."""

    def decide(
        self,
        obs: CausalObservation,
        *,
        observation_already_recorded: bool = False,
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        inner = self._direct_mpc_adapter.inner
        telemetry = dict(getattr(inner, "last_v29_telemetry", {}))
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "v29_runtime_contract": V29_OPERATIONAL_RUNTIME_CONTRACT,
                "v29_selection_contract": V29_SELECTION_CONTRACT,
                "v29_regime_value_contract": V29_REGIME_VALUE_CONTRACT,
                "v29_utility_shadow_contract": V29_AUTO_RBC_UTILITY_CONTRACT,
                "v29_candidate_telemetry": list(telemetry.get("candidates", [])),
                "v29_selected_source": str(telemetry.get("selected_source", "HOLD")),
                "v29_selected_q29_score_m3": float(
                    telemetry.get("selected_q29_score_m3", 0.0)
                ),
                "v29_selected_regime_residual_m3": float(
                    telemetry.get("selected_regime_residual_m3", 0.0)
                ),
                "v29_action_class": str(telemetry.get("action_class", "HOLD")),
                "v29_utility_shadow_present": bool(
                    telemetry.get("utility_shadow_present", False)
                ),
                "v29_utility_shadow_duplicate": bool(
                    telemetry.get("utility_shadow_duplicate", False)
                ),
                "v29_utility_shadow_selected": bool(
                    telemetry.get("utility_shadow_selected", False)
                ),
                "v29_utility_shadow_raw_changed_facility_count": int(
                    telemetry.get("utility_shadow_raw_changed_facility_count", 0)
                ),
                "v29_utility_shadow_retained_changed_facility_count": int(
                    telemetry.get("utility_shadow_retained_changed_facility_count", 0)
                ),
                "v29_utility_shadow_retained_utility_fraction": float(
                    telemetry.get("utility_shadow_retained_utility_fraction", 0.0)
                ),
                "v29_q95_mandatory": True,
                "v29_return_period_feature": False,
                "v29_event_duration_feature": False,
                "v29_development_only": True,
            }
        )
        source = (
            "MPC_DIRECT_TFV_V29"
            if telemetry.get("action_class") == "ACTION"
            else "LATCH_PREVIOUS_TARGET_DIRECT_TFV_V29"
        )
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


def build_operational_v29_controller(
    *,
    graph_path: str | Path,
    sensors_path: str | Path,
    config_path: str | Path,
    step1_path: str | Path,
    step2_path: str | Path,
    supervisory_control_path: str | Path,
    sequence_support_path: str | Path,
    v15_rank_checkpoint_path: str | Path,
    v21_boundary_checkpoint_path: str | Path,
    v27_value_checkpoint_path: str | Path,
    v28r1_residual_checkpoint_path: str | Path,
    v29_value_checkpoint_path: str | Path,
    dataset_manifest_path: str | Path,
    v27_dataset_manifest_path: str | Path,
    asset_manifest_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    parent_controller, graph, sensors, parent_lineage = build_operational_v28_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=supervisory_control_path,
        sequence_support_path=sequence_support_path,
        v15_rank_checkpoint_path=v15_rank_checkpoint_path,
        v21_boundary_checkpoint_path=v21_boundary_checkpoint_path,
        v27_value_checkpoint_path=v27_value_checkpoint_path,
        v28_residual_checkpoint_path=v28r1_residual_checkpoint_path,
        dataset_manifest_path=dataset_manifest_path,
        v27_dataset_manifest_path=v27_dataset_manifest_path,
        asset_manifest_path=asset_manifest_path,
        device=device,
        decision_runtime_budget_seconds=decision_runtime_budget_seconds,
        proposal_probe_chunk_size=proposal_probe_chunk_size,
    )
    parent_inner = parent_controller.controller
    old_mpc = parent_inner._direct_mpc_adapter.inner

    parent_residual = old_mpc.residual_value_model
    if (
        float(np.max(np.abs(parent_residual.weight), initial=0.0)) > 1.0e-10
        or abs(float(parent_residual.intercept)) > 1.0e-10
    ):
        raise ValueError("V29 requires the V28R1 alpha=0 Q27 fallback checkpoint")

    q27_sha = _sha(v27_value_checkpoint_path)
    regime_model, regime_payload = load_v29_regime_value_model(
        str(v29_value_checkpoint_path),
        expected_q27_checkpoint_sha256=q27_sha,
        expected_dataset_manifest_sha256=_sha(dataset_manifest_path),
    )
    lineage = dict(parent_lineage)
    lineage.update(
        {
            "contract": V29_OPERATIONAL_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "ready_for_policy_lock": False,
            "q27_frozen": True,
            "q95_mandatory": True,
            "v28r1_parent_residual_null": True,
            "v29_value_checkpoint_sha256": _sha(v29_value_checkpoint_path),
            "v29_value_checkpoint_contract": regime_payload.get("contract"),
            "v29_regime_value_contract": V29_REGIME_VALUE_CONTRACT,
            "v29_utility_shadow_contract": V29_AUTO_RBC_UTILITY_CONTRACT,
            "v29_selection_contract": V29_SELECTION_CONTRACT,
            "v29_portfolio_contract": V29_PORTFOLIO_CONTRACT,
            "v29_tfv_value_estimand": V29_TFV_VALUE_ESTIMAND,
            "return_period_used_as_rule_or_feature": False,
            "event_duration_used_as_rule_or_feature": False,
            "future_realized_rainfall_used_as_model_input": False,
        }
    )

    v29 = DirectTFVOperationalV29MPC(
        model=old_mpc.model,
        graph=old_mpc.graph,
        normalization=old_mpc.normalization,
        action_support=old_mpc.action_support,
        sequence_support=old_mpc.sequence_support,
        supervisory_mask=old_mpc.supervisory_mask,
        policy_return_model=old_mpc.policy_return_model,
        policy_return_normalization=old_mpc.policy_return_normalization,
        rank_adapter=old_mpc.rank_adapter,
        boundary_calibrator=old_mpc.boundary_calibrator,
        continuation_policy_sha256=old_mpc.policy_return_parent_continuation_sha256,
        design=old_mpc.design,
        proposal_probe_chunk_size=old_mpc.proposal_probe_chunk_size,
        v27_value_model=old_mpc.v27_value_model,
        v27_lineage=dict(getattr(old_mpc, "v27_lineage", {})),
        residual_value_model=old_mpc.residual_value_model,
        v28_lineage=dict(getattr(old_mpc, "v28_lineage", {})),
        regime_value_model=regime_model,
        v29_lineage=lineage,
    )
    controller_cfg = parent_inner.config
    inner = DirectTFVAuthoritativeV29Controller(
        step1=parent_inner.step1,
        mpc=v29,
        graph=graph,
        sensor_nodes=sensors,
        forecast=parent_inner.forecast,
        config=controller_cfg,
        device=device,
    )
    controller = ContinuityGuardController(
        inner,
        max_delta_per_update=0.5,
        allow_projection=False,
        enforce_current_delta=False,
    )
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVAuthoritativeV29Controller",
    "DirectTFVOperationalV29MPC",
    "V29_OPERATIONAL_RUNTIME_CONTRACT",
    "V29_PORTFOLIO_CONTRACT",
    "V29_SELECTION_CONTRACT",
    "V29_TFV_VALUE_ESTIMAND",
    "build_operational_v29_controller",
]
