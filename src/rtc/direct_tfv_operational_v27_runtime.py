"""Project7 V27 Development runtime.

V27 ranks a four-source engineering candidate portfolio with an unclipped decision-aware exact-return
latent score: the three V23 sources plus an Auto-RBC shadow proposal.  HOLD has latent value zero.
Auto-RBC never receives unconditional execution authority.

The actual execution path retains q95 joint-sequence support.  In parallel, V27 scores the same raw
pre-contraction candidates and reports whether q95 would change the selected source/value.  This is a
report-only support ablation, not a second unsafe execution path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .direct_tfv_operational_v23_runtime import DirectTFVOperationalV23MPC, build_operational_v23_controller
from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING, encode_policy_return_action_token
from .direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    V23_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio_v23,
)
from .direct_tfv_sequence_support import changed_facility_support_limit
from .direct_tfv_v27_auto_rbc_shadow import (
    V27_AUTO_RBC_SHADOW_CONTRACT,
    V27_AUTO_RBC_SHADOW_SOURCE,
    build_auto_rbc_shadow_proposal,
)
from .direct_tfv_v27_decision_value import load_v27_value_model
from .direct_tfv_v27_hydraulic_features import V27_HYDRAULIC_FEATURE_CONTRACT, build_v27_hydraulic_feature
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


OPERATIONAL_V27_RUNTIME_CONTRACT = "PROJECT7_OPERATIONAL_DEVELOPMENT_V27_DECISION_AWARE_RBC_SHADOW_V1"
V27_PORTFOLIO_CONTRACT = "PROJECT7_V27_V23_THREE_FAMILY_PLUS_AUTO_RBC_SHADOW_V1"
V27_SELECTION_CONTRACT = "PROJECT7_V27_ARGMIN_UNCLIPPED_LATENT_EXACT_RETURN_PLUS_HOLD_ZERO_V1"
V27_TFV_VALUE_ESTIMAND = (
    "SYSTEM_WIDE_CUMULATIVE_TFV_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DirectTFVOperationalV27MPC(DirectTFVOperationalV23MPC):
    policy_mode = "operational_development_v27_decision_aware_rbc_shadow"
    policy_mode_contract = OPERATIONAL_V27_RUNTIME_CONTRACT

    def __init__(self, *, v27_value_model: torch.nn.Module, v27_lineage: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.v27_value_model = v27_value_model
        self.v27_lineage = dict(v27_lineage)

    def _feature_and_scores(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        flow: torch.Tensor,
        active_target: torch.Tensor,
        target: torch.Tensor,
        source: str,
    ) -> tuple[Any, float, float, bool]:
        state2 = current_state[0] if current_state.ndim == 3 and int(current_state.shape[0]) == 1 else current_state
        built = build_v27_hydraulic_feature(
            step2_model=self.model,
            normalization=self.normalization,
            graph=self.graph,
            current_state=state2,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow[0],
            active_target=active_target,
            candidate_target=target,
            candidate_source=source,
            supervisory_mask=self.supervisory_mask,
            target_scale_m3=float(self.model.target_scale_m3.detach().cpu()),
        )
        latent = float(self.v27_value_model.latent_score(built.feature).detach().cpu())
        reported = float(self.v27_value_model.reported_prediction_m3(built.feature).detach().cpu())
        clip_hit = bool(self.v27_value_model.clip_hit(built.feature))
        return built, latent, reported, clip_hit

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V27 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("operational V27 requires previous_actuator_flow [1,109]")

        ceiling = int(changed_facility_support_limit(self.sequence_support, "q95"))
        hybrid = build_hybrid_policy_return_portfolio_v23(
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
        )
        learned = hybrid.learned_probe
        shadow = build_auto_rbc_shadow_proposal(
            graph=self.graph,
            current_state=current_state,
            active_target=active_target,
            supervisory_mask=self.supervisory_mask,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
        )
        proposals: list[tuple[str, torch.Tensor]] = [
            (str(proposal.source), proposal.target) for proposal in hybrid.candidates
        ]
        if int(shadow.retained_changed_facility_count) > 0:
            proposals.append((shadow.source, shadow.target))

        passive = torch.as_tensor(~self.supervisory_mask, dtype=torch.bool, device=active_target.device)
        evaluated: list[dict[str, Any]] = []
        seen: set[bytes] = set()
        shadow_duplicate = False
        for source, raw_target in proposals:
            if bool(torch.any(torch.abs(raw_target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("operational V27 raw candidate changed a passive channel")
            _, raw_sequence_batch = encode_policy_return_action_token(
                active_target,
                raw_target,
                horizon_steps=int(self.design.prediction_horizon_steps),
                first_action_steps=int(self.design.control_block_steps),
            )
            raw_sequence = raw_sequence_batch[0].detach()
            raw_changed = int(torch.count_nonzero(torch.abs(raw_target - active_target) > 1.0e-7).item())
            raw_diag = self.joint_sequence_support_diagnostics(raw_sequence, active_target)
            with torch.inference_mode():
                _, raw_latent, raw_reported, raw_clip_hit = self._feature_and_scores(
                    current_state=current_state,
                    rainfall=rainfall,
                    flow=flow,
                    active_target=active_target,
                    target=raw_target,
                    source=source,
                )

            target, sequence, changed, support = self._h10_supported_target(raw_target, active_target)
            if int(changed) <= 0:
                continue
            if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("operational V27 supported candidate changed a passive channel")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                if source == V27_AUTO_RBC_SHADOW_SOURCE:
                    shadow_duplicate = True
                continue
            seen.add(key)
            with torch.inference_mode():
                built, latent, reported, clip_hit = self._feature_and_scores(
                    current_state=current_state,
                    rainfall=rainfall,
                    flow=flow,
                    active_target=active_target,
                    target=target,
                    source=source,
                )
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
                    "source": source,
                    "target": target,
                    "sequence": sequence,
                    "changed": int(changed),
                    "support": dict(support),
                    "raw_support": dict(raw_diag),
                    "latent": latent,
                    "reported": reported,
                    "clip_hit": clip_hit,
                    "raw_latent": raw_latent,
                    "raw_reported": raw_reported,
                    "raw_clip_hit": raw_clip_hit,
                    "raw_changed": raw_changed,
                    "base_score": base_score,
                    "feature": built,
                }
            )

        if not evaluated:
            return self._hold_result(started=started, active_target=active_target, learned=learned)

        supported_best = min(evaluated, key=lambda row: (float(row["latent"]), str(row["source"])))
        raw_best = min(evaluated, key=lambda row: (float(row["raw_latent"]), str(row["source"])))
        action = float(supported_best["latent"]) < 0.0
        row = supported_best
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        executed = row["sequence"] if action else hold
        changed_indices = torch.nonzero(torch.abs(row["target"] - active_target) > 1.0e-7).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(index)]) for index in changed_indices)
        support = row["support"]
        binding_count = sum(bool(item["support"]["binding"]) for item in evaluated)
        clip_hits = sum(bool(item["clip_hit"]) for item in evaluated)
        raw_clip_hits = sum(bool(item["raw_clip_hit"]) for item in evaluated)
        support_changed_source = str(raw_best["source"]) != str(supported_best["source"])
        shadow_present = any(item["source"] == V27_AUTO_RBC_SHADOW_SOURCE for item in evaluated)
        shadow_selected = bool(action and row["source"] == V27_AUTO_RBC_SHADOW_SOURCE)
        diagnostics_text = (
            "V27_DECISION_AWARE|"
            f"latent_min={float(row['latent']):.9g}|reported_m3={float(row['reported']):.9g}|"
            f"clip_hits={clip_hits}|raw_clip_hits={raw_clip_hits}|candidate_count={len(evaluated)}|"
            f"q95_binding_candidates={binding_count}|raw_best={raw_best['source']}|"
            f"supported_best={supported_best['source']}|q95_selection_changed={str(support_changed_source).lower()}|"
            f"raw_best_latent={float(raw_best['raw_latent']):.9g}|"
            f"supported_best_latent={float(supported_best['latent']):.9g}|"
            f"shadow_present={str(shadow_present).lower()}|shadow_selected={str(shadow_selected).lower()}|"
            f"shadow_duplicate={str(shadow_duplicate).lower()}|shadow_raw_changed={shadow.raw_changed_facility_count}|"
            f"shadow_retained={shadow.retained_changed_facility_count}"
        )
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=row["sequence"],
            predicted_delta_tfv_m3=float(row["reported"]) if action else 0.0,
            raw_optimized_predicted_delta_tfv_m3=float(row["reported"]),
            selected_source=(f"OPERATIONAL_V27::{row['source']}" if action else "LATCH_PREVIOUS_TARGET_OPERATIONAL_V27_LATENT_NONNEGATIVE"),
            candidate_valid=bool(action),
            admission_margin_m3=0.0,
            admission_upper_bound_m3=float(row["reported"]),
            admission_margin_kind="v27_unclipped_latent_hold_zero_reference",
            admission_passed=bool(action),
            calibrated_admission_contract=V27_SELECTION_CONTRACT,
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
            policy_return_predicted_delta_tfv_m3=float(row["reported"]),
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=float(row["reported"]),
            policy_return_admission_passed=bool(action),
            policy_return_admission_contract=V27_SELECTION_CONTRACT,
            policy_return_estimand=V27_TFV_VALUE_ESTIMAND,
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=V27_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=str(row["source"]) if action else "HOLD",
            policy_return_portfolio_sources=tuple(str(item["source"]) for item in evaluated),
            policy_return_portfolio_scores_m3=tuple(float(item["reported"]) for item in evaluated),
            policy_return_portfolio_upper_bounds_m3=tuple(float(item["reported"]) for item in evaluated),
            policy_return_portfolio_base_step2_scores_m3=tuple(float(item["base_score"]) for item in evaluated),
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
            scipy_message=diagnostics_text,
            tfv_value_available=True,
            tfv_value_prediction_m3=float(row["reported"]),
            tfv_value_upper_bound_m3=float(row["reported"]),
            tfv_value_admission_passed=bool(action),
            engineering_feasible=True,
            passive_channels_unchanged=True,
            sequence_support_valid=True,
            tfv_value_gate_reason=diagnostics_text,
        )


def build_operational_v27_controller(
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
    dataset_manifest_path: str | Path,
    asset_manifest_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
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
        "feature_contract": V27_HYDRAULIC_FEATURE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "v23_hydraulic_candidate_contract": V23_HYDRAULIC_CANDIDATE_CONTRACT,
    }
    value_model, checkpoint = load_v27_value_model(
        str(v27_value_checkpoint_path),
        device=device,
        expected_lineage=expected_lineage,
    )
    v27 = DirectTFVOperationalV27MPC(
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
        v27_value_model=value_model,
        v27_lineage=expected_lineage,
    )
    controller.controller._direct_mpc_adapter.inner = v27
    lineage = dict(parent_lineage)
    lineage.update(
        {
            "contract": OPERATIONAL_V27_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "v27_selection_contract": V27_SELECTION_CONTRACT,
            "v27_portfolio_contract": V27_PORTFOLIO_CONTRACT,
            "v27_value_checkpoint_sha256": _sha(v27_value_checkpoint_path),
            "v27_dataset_manifest_sha256": _sha(dataset_manifest_path),
            "v27_value_estimand": V27_TFV_VALUE_ESTIMAND,
            "v27_auto_rbc_shadow_contract": V27_AUTO_RBC_SHADOW_CONTRACT,
            "v27_auto_rbc_shadow_is_candidate_only": True,
            "v27_runtime_ranking_uses_unclipped_latent": True,
            "v27_q95_support_execution": True,
            "v27_q95_precontraction_counterfactual_scoring": True,
            "v15_rank_used_for_v27_candidate_selection": False,
            "v21_boundary_used_for_v27_action_admission": False,
            "v25_ucb_used_for_v27_action_admission": False,
            "scientific_metrics_block_runtime": False,
            "ready_for_policy_lock": False,
            "checkpoint_training_report_present": isinstance(checkpoint.get("training_report"), dict),
            "checkpoint_test_report_present": isinstance(checkpoint.get("test_report"), dict),
        }
    )
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVOperationalV27MPC",
    "OPERATIONAL_V27_RUNTIME_CONTRACT",
    "V27_PORTFOLIO_CONTRACT",
    "V27_SELECTION_CONTRACT",
    "V27_TFV_VALUE_ESTIMAND",
    "build_operational_v27_controller",
]
