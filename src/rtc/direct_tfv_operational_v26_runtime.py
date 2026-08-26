"""Development V26 runtime: direct exact-return selection across the full three-family portfolio.

V26 removes the V15-selected/V21-boundary/V25-UCB admission stack from the *decision path*.
Engineering-feasible V23 candidates are each represented with action-conditioned hydraulic features,
scored by the V26 exact-return model, and compared directly with HOLD=0.  The lowest predicted TFV
return is executed when negative; otherwise the previous target is held.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .direct_tfv_operational_v23_runtime import (
    DirectTFVOperationalV23MPC,
    build_operational_v23_controller,
)
from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
from .direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    V23_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio_v23,
)
from .direct_tfv_sequence_support import changed_facility_support_limit
from .direct_tfv_v26_hydraulic_features import (
    V26_HYDRAULIC_FEATURE_CONTRACT,
    build_v26_hydraulic_feature,
)
from .direct_tfv_v26_value_model import load_v26_value_model
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


OPERATIONAL_V26_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V26_DIRECT_EXACT_RETURN_PORTFOLIO_VALUE_V1"
)
V26_SELECTION_CONTRACT = (
    "PROJECT7_V26_MIN_PREDICTED_EXACT_RETURN_OVER_ENGINEERING_CANDIDATES_PLUS_HOLD_ZERO_V1"
)
V26_TFV_VALUE_ESTIMAND = (
    "SYSTEM_WIDE_CUMULATIVE_TFV_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DirectTFVOperationalV26MPC(DirectTFVOperationalV23MPC):
    policy_mode = "operational_development_v26_direct_exact_return_portfolio"
    policy_mode_contract = OPERATIONAL_V26_RUNTIME_CONTRACT

    def __init__(self, *, v26_value_model: torch.nn.Module, v26_lineage: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.v26_value_model = v26_value_model
        self.v26_lineage = dict(v26_lineage)

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V26 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("operational V26 requires previous_actuator_flow [1,109]")

        ceiling = changed_facility_support_limit(self.sequence_support, "q95")
        hybrid = build_hybrid_policy_return_portfolio_v23(
            model=self.model,
            normalization=self.normalization,
            graph=self.graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active_target,
            first_radius=self.first_radius,
            max_changed_facilities=int(ceiling),
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
            probe_chunk_size=self.proposal_probe_chunk_size,
            supervisory_mask=self.supervisory_mask,
        )
        learned = hybrid.learned_probe
        passive = torch.as_tensor(~self.supervisory_mask, dtype=torch.bool, device=active_target.device)
        evaluated: list[dict[str, Any]] = []
        seen: set[bytes] = set()
        state2 = current_state[0] if current_state.ndim == 3 and int(current_state.shape[0]) == 1 else current_state
        for proposal in hybrid.candidates:
            target, sequence, changed, support = self._h10_supported_target(proposal.target, active_target)
            if int(changed) <= 0:
                continue
            if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("operational V26 candidate changed a passive channel")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            with torch.inference_mode():
                built = build_v26_hydraulic_feature(
                    step2_model=self.model,
                    normalization=self.normalization,
                    graph=self.graph,
                    current_state=state2,
                    rainfall_scenarios=rainfall,
                    previous_actuator_flow=flow[0],
                    active_target=active_target,
                    candidate_target=target,
                    candidate_source=str(proposal.source),
                    supervisory_mask=self.supervisory_mask,
                    target_scale_m3=float(self.model.target_scale_m3.detach().cpu()),
                )
                prediction = float(self.v26_value_model.predict(built.feature).detach().cpu())
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
                {
                    "source": str(proposal.source),
                    "target": target,
                    "sequence": sequence,
                    "changed": int(changed),
                    "support": dict(support),
                    "prediction": prediction,
                    "base_score": base_score,
                    "feature_diag": built,
                }
            )

        if not evaluated:
            result = self._hold_result(started=started, active_target=active_target, learned=learned)
            return result

        selected_index = min(range(len(evaluated)), key=lambda index: float(evaluated[index]["prediction"]))
        row = evaluated[selected_index]
        prediction = float(row["prediction"])
        action = prediction < 0.0
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        executed = row["sequence"] if action else hold
        changed_indices = torch.nonzero(torch.abs(row["target"] - active_target) > 1.0e-7).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(index)]) for index in changed_indices)
        support = row["support"]
        hydraulic = hybrid.hydraulic_diagnostics
        portfolio_predictions = tuple(float(item["prediction"]) for item in evaluated)
        portfolio_base = tuple(float(item["base_score"]) for item in evaluated)
        portfolio_sources = tuple(str(item["source"]) for item in evaluated)
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=row["sequence"],
            predicted_delta_tfv_m3=prediction if action else 0.0,
            raw_optimized_predicted_delta_tfv_m3=prediction,
            selected_source=(
                f"OPERATIONAL_V26::{row['source']}"
                if action
                else "LATCH_PREVIOUS_TARGET_OPERATIONAL_V26_NO_NEGATIVE_EXACT_RETURN"
            ),
            candidate_valid=bool(action),
            admission_margin_m3=0.0,
            admission_upper_bound_m3=prediction,
            admission_margin_kind="v26_direct_exact_return_hold_zero_reference",
            admission_passed=bool(action),
            calibrated_admission_contract=V26_SELECTION_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=82,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=int(row["changed"]),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=int(row["changed"]) if action else 0,
            maximum_support_ratio=self._first_move_support_ratio(row["target"], active_target),
            joint_sequence_support_quantile=str(support["quantile"]),
            joint_sequence_first_block_l1=float(support["first_block_l1"]),
            joint_sequence_h120_l1=float(support["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(support["h120_total_variation_l1"]),
            joint_sequence_support_max_ratio=float(support["max_ratio"]),
            joint_sequence_support_binding=bool(support["binding"]),
            policy_return_predicted_delta_tfv_m3=prediction,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=prediction,
            policy_return_admission_passed=bool(action),
            policy_return_admission_contract=V26_SELECTION_CONTRACT,
            policy_return_estimand=V26_TFV_VALUE_ESTIMAND,
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=V23_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=row["source"] if action else "HOLD",
            policy_return_portfolio_sources=portfolio_sources,
            policy_return_portfolio_scores_m3=portfolio_predictions,
            policy_return_portfolio_upper_bounds_m3=portfolio_predictions,
            policy_return_portfolio_base_step2_scores_m3=portfolio_base,
            h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
            h10_probe_count=int(learned.probe_count),
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics=DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            refined_first_move_changed_facility_count=int(row["changed"]),
            refined_first_move_changed_facility_ids=changed_ids,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            gradient_norm=0.0,
            scipy_message=(
                "NOT_USED_OPERATIONAL_V26_DIRECT_EXACT_RETURN|"
                f"prediction={prediction:.9g}|candidate_count={len(evaluated)}|"
                f"stress={hydraulic.network_stress_q75:.9g}|blend={hydraulic.strong_storm_blend:.9g}|"
                "v15_rank_used_for_selection=false|v21_boundary_used_for_admission=false|"
                "ucb_gate_used=false"
            ),
        )


def build_operational_v26_controller(
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
    v26_value_checkpoint_path: str | Path,
    dataset_manifest_path: str | Path,
    asset_manifest_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    # Reuse the tested Step1/Step2/controller construction surface. V15/V21 are loaded only because
    # the V23 parent factory currently owns that construction code; V26 optimize() never consults
    # either object for candidate selection or ACTION/HOLD admission.
    controller, graph, sensors, parent_lineage = build_operational_v23_controller(
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
        "base_step2_sha256": _sha(step2_path),
        "asset_manifest_sha256": _sha(asset_manifest_path),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(sequence_support_path),
        "dataset_manifest_sha256": _sha(dataset_manifest_path),
        "feature_contract": V26_HYDRAULIC_FEATURE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "v23_hydraulic_candidate_contract": V23_HYDRAULIC_CANDIDATE_CONTRACT,
    }
    value_model, checkpoint = load_v26_value_model(
        str(v26_value_checkpoint_path),
        device=device,
        expected_lineage=expected_lineage,
    )
    v26_mpc = DirectTFVOperationalV26MPC(
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
        v26_value_model=value_model,
        v26_lineage=expected_lineage,
    )
    controller.controller._direct_mpc_adapter.inner = v26_mpc
    lineage = dict(parent_lineage)
    lineage.update(
        {
            "contract": OPERATIONAL_V26_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "v26_selection_contract": V26_SELECTION_CONTRACT,
            "v26_value_checkpoint_sha256": _sha(v26_value_checkpoint_path),
            "v26_dataset_manifest_sha256": _sha(dataset_manifest_path),
            "v26_value_estimand": V26_TFV_VALUE_ESTIMAND,
            "v15_rank_used_for_v26_candidate_selection": False,
            "v21_boundary_used_for_v26_action_admission": False,
            "v25_ucb_used_for_v26_action_admission": False,
            "scientific_metrics_block_runtime": False,
            "hydraulic_knowledge_role": "ACTION_CONDITIONED_FEATURES_AND_EXISTING_HYDRAULIC_CANDIDATE",
            "hold_reference_delta_tfv_m3": 0.0,
            "selection_rule": "ARGMIN_PREDICTED_EXACT_RETURN_OVER_PORTFOLIO_PLUS_HOLD",
            "q95_support_unchanged": True,
            "max_setting_delta_per_update": 0.5,
            "ready_for_policy_lock": False,
            "checkpoint_validation_report_present": isinstance(checkpoint.get("training_report"), dict),
            "checkpoint_test_report_present": isinstance(checkpoint.get("test_report"), dict),
        }
    )
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVOperationalV26MPC",
    "OPERATIONAL_V26_RUNTIME_CONTRACT",
    "V26_SELECTION_CONTRACT",
    "V26_TFV_VALUE_ESTIMAND",
    "build_operational_v26_controller",
]
