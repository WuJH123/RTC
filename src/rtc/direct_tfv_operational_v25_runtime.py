"""Development-only V25 runtime with conservative selected-action H120 value admission.

V25 preserves the V23 portfolio and frozen V15 rank path, but removes the V24 hydraulic stress
escape entirely.  The selected candidate is executed only when the matched Development value
calibrator's one-sided H120 upper bound is strictly negative.  Stress/blend remain telemetry and
cannot authorize an action.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .controller_direct_tfv_portfolio import PortfolioMemorySafeDirectTFVAuthoritativeController
from .direct_tfv_operational_v23_runtime import (
    OPERATIONAL_V23_RUNTIME_CONTRACT,
    DirectTFVOperationalV23MPC,
    build_operational_v23_controller,
)
from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
)
from .direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    V23_PORTFOLIO_CONTRACT,
)
from .direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from .direct_tfv_v25_features import V25_FEATURE_CONTRACT, build_v25_selected_query
from .direct_tfv_v25_value_calibration import (
    V25_VALUE_CALIBRATOR_CONTRACT,
    V25ValueCalibratorModule,
    load_v25_value_calibrator,
)
from .direct_tfv_v25_value_gate import (
    V25_VALUE_GATE_CONTRACT,
    evaluate_v25_value_gate,
    validate_v25_lineage,
)
from .native_supervisory_control import load_native_supervisory_control
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


OPERATIONAL_V25_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V25_VALUE_GATED_STEP3_V1"
)
V25_TFV_VALUE_ESTIMAND = (
    "TFV_H120_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DirectTFVOperationalV25MPC(DirectTFVOperationalV23MPC):
    """V23 candidate/rank geometry with V25 value-only ACTION/HOLD admission."""

    policy_mode = "operational_development_v25_value_gated_step3"
    policy_mode_contract = OPERATIONAL_V25_RUNTIME_CONTRACT

    def __init__(self, *, v25_value_calibrator: V25ValueCalibratorModule, v25_lineage: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.v25_value_calibrator = v25_value_calibrator
        self.v25_lineage = dict(v25_lineage)
        self.v25_lineage_valid = True

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V25 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("operational V25 requires previous_actuator_flow [1,109]")

        selected = build_v25_selected_query(
            mpc=self,
            current_state=current_state,
            rainfall=rainfall,
            previous_actuator_flow=flow,
            active_target=active_target,
        )
        with torch.inference_mode():
            prediction_tensor, upper_tensor = self.v25_value_calibrator.predict(selected.feature)
        prediction = float(prediction_tensor.detach().cpu())
        upper = float(upper_tensor.detach().cpu())
        margin = float(self.v25_value_calibrator.one_sided_error_margin_m3.detach().cpu())
        passive = bool(
            not torch.any(
                torch.abs(selected.selected_target[~torch.as_tensor(self.supervisory_mask, dtype=torch.bool, device=active_target.device)]
                          - active_target[~torch.as_tensor(self.supervisory_mask, dtype=torch.bool, device=active_target.device)])
                > 1.0e-7
            )
        )
        support_valid = float(selected.selected_support.get("max_ratio", float("inf"))) <= 1.0 + 1.0e-6
        expected_lineage = {
            key: value
            for key, value in self.v25_lineage.items()
            if key in {
                "base_step2_sha256",
                "v15_rank_checkpoint_sha256",
                "v21_boundary_checkpoint_sha256",
                "asset_manifest_sha256",
                "supervisory_mask_sha256",
                "sequence_support_sha256",
                "matched_truth_bank_manifest_sha256",
                "v23_portfolio_contract",
                "v23_hydraulic_candidate_contract",
                "feature_contract",
            }
        }
        gate = evaluate_v25_value_gate(
            candidate_source=selected.selected_source,
            predicted_delta_tfv_m3=prediction,
            one_sided_error_margin_m3=margin,
            tfv_value_available=True,
            calibration_available=True,
            lineage_valid=bool(self.v25_lineage_valid and validate_v25_lineage(self.v25_lineage, expected_lineage)),
            engineering_feasible=True,
            passive_channels_unchanged=passive,
            sequence_support_valid=support_valid,
            network_stress_q75=selected.network_stress_q75,
            strong_storm_blend=selected.strong_storm_blend,
        )
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        executed = selected.selected_sequence if gate.action else hold
        changed_indices = torch.nonzero(
            torch.abs(selected.selected_target - active_target) > 1.0e-7
        ).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(index)]) for index in changed_indices)
        support = selected.selected_support
        support_quantile = str(support.get("quantile", "q95"))
        message = (
            "NOT_USED_OPERATIONAL_V25_VALUE_GATED_PORTFOLIO|"
            f"v25_path={'TFV_VALUE_ACTION' if gate.action else 'TFV_VALUE_HOLD'}|"
            f"reason={gate.reason}|tfv_value_available={gate.tfv_value_available}|"
            f"tfv_value_prediction_m3={prediction:.9g}|tfv_value_upper_bound_m3={upper:.9g}|"
            f"tfv_value_admission_passed={gate.action}|engineering_feasible={gate.engineering_feasible}|"
            f"passive_channels_unchanged={gate.passive_channels_unchanged}|"
            f"sequence_support_valid={gate.sequence_support_valid}|"
            f"stress={selected.network_stress_q75:.9g}|blend={selected.strong_storm_blend:.9g}|"
            "hydraulic_stress_direct_action_authorization=false"
        )
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=selected.selected_sequence,
            predicted_delta_tfv_m3=prediction if gate.action else 0.0,
            raw_optimized_predicted_delta_tfv_m3=prediction,
            selected_source=(
                f"OPERATIONAL_V25_VALUE::{selected.selected_source}"
                if gate.action
                else f"LATCH_PREVIOUS_TARGET_OPERATIONAL_V25_TFV_VALUE_{gate.reason}"
            ),
            candidate_valid=bool(gate.action),
            admission_margin_m3=margin,
            admission_upper_bound_m3=upper,
            admission_margin_kind="v25_calibrated_one_sided_h120_tfv_upper_bound",
            admission_passed=bool(gate.action),
            calibrated_admission_contract=V25_VALUE_GATE_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=109,
            predicted_beneficial_facility_count=0,
            active_facility_count=int(selected.selected_changed_facility_count),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=(),
            first_move_changed_facility_count=(
                int(selected.selected_changed_facility_count) if gate.action else 0
            ),
            maximum_support_ratio=float(support.get("max_ratio", 0.0)),
            joint_sequence_support_quantile=support_quantile,
            joint_sequence_first_block_l1=float(support.get("first_block_l1", 0.0)),
            joint_sequence_h120_l1=float(support.get("h120_l1", 0.0)),
            joint_sequence_h120_total_variation_l1=float(support.get("h120_total_variation_l1", 0.0)),
            joint_sequence_support_max_ratio=float(support.get("max_ratio", 0.0)),
            joint_sequence_support_binding=bool(support.get("binding", False)),
            policy_return_predicted_delta_tfv_m3=prediction,
            policy_return_margin_m3=margin,
            policy_return_upper_bound_m3=upper,
            policy_return_admission_passed=bool(gate.action),
            policy_return_admission_contract=V25_VALUE_GATE_CONTRACT,
            policy_return_estimand=V25_TFV_VALUE_ESTIMAND,
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=V23_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=selected.candidate_count,
            policy_return_portfolio_selected_source=selected.selected_source if gate.action else "HOLD",
            policy_return_portfolio_sources=selected.candidate_sources,
            policy_return_portfolio_scores_m3=selected.raw_rank_scores_m3,
            policy_return_portfolio_upper_bounds_m3=selected.relative_rank_scores,
            policy_return_portfolio_base_step2_scores_m3=selected.base_step2_scores_m3,
            h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
            h10_probe_count=0,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics=DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            refined_first_move_changed_facility_count=int(selected.selected_changed_facility_count),
            refined_first_move_changed_facility_ids=changed_ids,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            gradient_norm=0.0,
            scipy_message=message,
            tfv_value_available=bool(gate.tfv_value_available),
            tfv_value_prediction_m3=gate.tfv_value_prediction_m3,
            tfv_value_upper_bound_m3=gate.tfv_value_upper_bound_m3,
            tfv_value_admission_passed=bool(gate.tfv_value_admission_passed),
            engineering_feasible=bool(gate.engineering_feasible),
            passive_channels_unchanged=bool(gate.passive_channels_unchanged),
            sequence_support_valid=bool(gate.sequence_support_valid),
            tfv_value_gate_reason=gate.reason,
        )


def build_operational_v25_controller(
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
    v25_value_checkpoint_path: str | Path,
    matched_truth_bank_manifest_path: str | Path,
    asset_manifest_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    """Construct V25 through the already-tested V23 factory, then replace only its MPC policy."""

    controller, graph, sensors, v23_lineage = build_operational_v23_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=supervisory_control_path,
        sequence_support_path=sequence_support_path,
        v15_rank_checkpoint_path=v15_rank_checkpoint_path,
        v21_boundary_checkpoint_path=v21_boundary_checkpoint_path,
        device=device,
        decision_runtime_budget_seconds=decision_runtime_budget_seconds,
        proposal_probe_chunk_size=proposal_probe_chunk_size,
    )
    old = controller.controller._direct_mpc_adapter.inner
    control_payload = json.loads(Path(supervisory_control_path).read_text(encoding="utf-8"))
    expected_lineage = {
        "base_step2_sha256": str(v23_lineage["base_step2_sha256"]),
        "v15_rank_checkpoint_sha256": str(v23_lineage["v15_rank_checkpoint_sha256"]),
        "v21_boundary_checkpoint_sha256": str(v23_lineage["v21_boundary_checkpoint_sha256"]),
        "asset_manifest_sha256": _sha(asset_manifest_path),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(sequence_support_path),
        "matched_truth_bank_manifest_sha256": _sha(matched_truth_bank_manifest_path),
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "v23_hydraulic_candidate_contract": V23_HYDRAULIC_CANDIDATE_CONTRACT,
        "feature_contract": V25_FEATURE_CONTRACT,
    }
    calibrator, checkpoint = load_v25_value_calibrator(
        str(v25_value_checkpoint_path),
        device=device,
        expected_lineage=expected_lineage,
    )
    oof_metrics = checkpoint.get("oof_metrics", {})
    if oof_metrics.get("train_oof_supported") is not True:
        raise ValueError("V25 calibrator does not pass the Development OOF gate")
    v25_mpc = DirectTFVOperationalV25MPC(
        model=old.model,
        graph=old.graph,
        normalization=old.normalization,
        action_support=old.action_support,
        sequence_support=old.sequence_support,
        supervisory_mask=old.supervisory_mask,
        policy_return_model=old.policy_return_model,
        policy_return_normalization=old.policy_return_normalization,
        rank_adapter=old.rank_adapter,
        boundary_calibrator=old.boundary_calibrator,
        continuation_policy_sha256=old.policy_return_parent_continuation_sha256,
        design=old.design,
        proposal_probe_chunk_size=old.proposal_probe_chunk_size,
        v25_value_calibrator=calibrator,
        v25_lineage=expected_lineage,
    )
    controller.controller._direct_mpc_adapter.inner = v25_mpc
    lineage = dict(v23_lineage)
    lineage.update(
        {
            "contract": OPERATIONAL_V25_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "v24_hydraulic_stress_escape_enabled": False,
            "v25_value_gate_contract": V25_VALUE_GATE_CONTRACT,
            "v25_value_calibrator_checkpoint_sha256": _sha(v25_value_checkpoint_path),
            "v25_matched_truth_bank_manifest_sha256": _sha(matched_truth_bank_manifest_path),
            "tfv_value_estimand": V25_TFV_VALUE_ESTIMAND,
            "hydraulic_stress_is_diagnostic_only": True,
            "hydraulic_candidate_uses_same_portfolio_rank_and_value_gate": True,
            "unknown_value_fail_closed_to_hold": True,
            "v23_learned_path_preserved": True,
            "conformal_admission_used": False,
            "online_swmm_candidate_search": False,
            "online_lbfgsb_used": False,
            "projected_gradient_h10_enabled": False,
            "ready_for_policy_lock": False,
        }
    )
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVOperationalV25MPC",
    "OPERATIONAL_V25_RUNTIME_CONTRACT",
    "V25_TFV_VALUE_ESTIMAND",
    "build_operational_v25_controller",
]
