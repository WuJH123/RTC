"""Development-only operational runtime for frozen V15 rank + V21 selected/HOLD boundary.

This lane exists to steer RTC development with authoritative closed-loop SWMM before the offline
Step3 paper gate is solved. It never changes Formal/Policy-Lock requirements: no conformal/admission
claim is made, the raw V21 physical-zero boundary alone chooses ACTION/HOLD, and every run is stamped
as non-promotable Development evidence.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .controller_direct_tfv_portfolio import PortfolioMemorySafeDirectTFVAuthoritativeController
from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING, DIRECT_TFV_POLICY_RETURN_ESTIMAND
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_policy_return_query_margin import build_query_margin_features, load_query_margin_checkpoint
from .direct_tfv_policy_return_facility_boundary_v20 import build_facility_boundary_parts_v20
from .direct_tfv_policy_return_selected_boundary_v21 import (
    BOUNDARY_ZERO,
    DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT,
    SelectedBoundaryCalibratorV21,
    SelectedBoundaryPreprocessorV21,
    build_selected_portfolio_feature_v21,
)
from .direct_tfv_sequence_support import changed_facility_support_limit, validate_direct_tfv_sequence_support
from .forecast import PersistenceDecayForecast
from .native_supervisory_control import load_native_supervisory_control
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12
from .step3_tfv_value_mpc_v13 import DirectTFVHybridPolicyReturnMPCV13


OPERATIONAL_V21_RUNTIME_CONTRACT = "PROJECT7_OPERATIONAL_DEVELOPMENT_V21_RAW_BOUNDARY_RTC_V1"
OPERATIONAL_RAW_BOUNDARY_ADMISSION = "DEVELOPMENT_RAW_V21_ZERO_BOUNDARY_NO_CONFORMAL"
CURRENT_THREE_FAMILY_PORTFOLIO = (
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v21_calibrator(path: str | Path, *, device: torch.device) -> tuple[SelectedBoundaryCalibratorV21, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT:
        raise ValueError("operational V21 runtime requires the V21 selected-boundary checkpoint")
    if payload.get("development_only") is not True:
        raise ValueError("V21 checkpoint must remain Development-only")
    pre = SelectedBoundaryPreprocessorV21(
        feature_scale=torch.as_tensor(payload["feature_scale"], dtype=torch.float32, device=device),
        components=torch.as_tensor(payload["svd_components"], dtype=torch.float32, device=device),
    )
    model = SelectedBoundaryCalibratorV21(
        preprocessor=pre,
        boundary_weight=torch.as_tensor(payload["boundary_weight"], dtype=torch.float32, device=device),
        magnitude_weight=torch.as_tensor(payload["magnitude_weight"], dtype=torch.float32, device=device),
        target_scale_m3=float(payload["target_scale_m3"]),
    ).to(device)
    model.eval()
    return model, payload


class DirectTFVOperationalV21MPC(DirectTFVHybridPolicyReturnMPCV13):
    policy_mode = "operational_development_v21_selected_boundary"
    policy_mode_contract = OPERATIONAL_V21_RUNTIME_CONTRACT

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
        rank_adapter: Any,
        boundary_calibrator: SelectedBoundaryCalibratorV21,
        continuation_policy_sha256: str,
        design: DirectTFVMPCDesignV4,
        proposal_probe_chunk_size: int = 24,
    ) -> None:
        # Intentionally bypass V13's calibrated-admission constructor. This lane is raw-boundary
        # Development only and is forbidden from Policy Lock.
        DirectTFVRecedingMPCV4.__init__(
            self,
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            design=design,
        )
        mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
        if mask.shape != (109,) or int(mask.sum()) != 82:
            raise ValueError("operational V21 runtime requires frozen 82/109 supervisory geometry")
        validate_direct_tfv_sequence_support(sequence_support, actuator_ids=graph.actuator_ids, supervisory_mask=mask)
        self.sequence_support = dict(sequence_support)
        self.supervisory_mask = mask
        self.supervisory_control_dimension = 82
        self.policy_return_model = policy_return_model
        self.policy_return_model.eval()
        self.policy_return_normalization = policy_return_normalization
        self.rank_adapter = rank_adapter
        self.rank_adapter.eval()
        self.boundary_calibrator = boundary_calibrator
        self.boundary_calibrator.eval()
        self.policy_return_parent_continuation_sha256 = str(continuation_policy_sha256).lower()
        self.proposal_probe_chunk_size = int(proposal_probe_chunk_size)
        self.projected_gradient_steps = 1
        self.projected_gradient_step_fraction = 0.25

    def _hold_result(self, *, started: float, active_target: torch.Tensor, learned: Any) -> DirectTFVMPCResultV12:
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        return DirectTFVMPCResultV12(
            settings=hold,
            optimized_candidate_settings=hold,
            predicted_delta_tfv_m3=0.0,
            raw_optimized_predicted_delta_tfv_m3=0.0,
            selected_source="LATCH_PREVIOUS_TARGET_EMPTY_OPERATIONAL_V21_PORTFOLIO",
            candidate_valid=False,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=0.0,
            admission_margin_kind="development_raw_v21_zero_boundary",
            admission_passed=False,
            calibrated_admission_contract=OPERATIONAL_RAW_BOUNDARY_ADMISSION,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=82,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=0,
            active_facility_ids=(),
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=0,
            maximum_support_ratio=0.0,
            joint_sequence_support_quantile="q95",
            joint_sequence_first_block_l1=0.0,
            joint_sequence_h120_l1=0.0,
            joint_sequence_h120_total_variation_l1=0.0,
            joint_sequence_support_max_ratio=0.0,
            joint_sequence_support_binding=False,
            policy_return_predicted_delta_tfv_m3=0.0,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=0.0,
            policy_return_admission_passed=False,
            policy_return_admission_contract=OPERATIONAL_RAW_BOUNDARY_ADMISSION,
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
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            gradient_norm=0.0,
            scipy_message="NOT_USED_OPERATIONAL_V21_FINITE_PORTFOLIO",
        )

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V21 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("operational V21 requires previous_actuator_flow [1,109]")

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
        evaluated: list[dict[str, Any]] = []
        seen: set[bytes] = set()
        passive = torch.as_tensor(~self.supervisory_mask, dtype=torch.bool, device=active_target.device)
        for proposal in hybrid.candidates:
            target, sequence, changed, diagnostics = self._h10_supported_target(proposal.target, active_target)
            if changed <= 0:
                continue
            if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("operational V21 candidate changed a passive channel")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            raw_rank = float(self._score_policy_return_target(
                current_state=current_state,
                rainfall=rainfall,
                previous_actuator_flow=flow,
                active_target=active_target,
                candidate_target=target,
            ).detach().cpu())
            base_score = float(score_h10_first_action_targets(
                model=self.model,
                normalization=self.normalization,
                graph=self.graph,
                current_state=current_state,
                rainfall_scenarios=rainfall,
                previous_actuator_flow=flow,
                active_target=active_target,
                candidate_targets=target.reshape(1, 109),
                probe_chunk_size=1,
            )[0].detach().cpu())
            facility = build_facility_boundary_parts_v20(
                step2_model=self.model,
                normalization=self.normalization,
                graph=self.graph,
                current_state=(current_state[0] if current_state.ndim == 3 and int(current_state.shape[0]) == 1 else current_state),
                rainfall_scenarios=rainfall,
                previous_actuator_flow=flow[0],
                active_target=active_target,
                candidate_target=target,
                candidate_source=str(proposal.source),
                supervisory_mask=self.supervisory_mask,
                target_scale_m3=float(self.model.target_scale_m3.detach().cpu()),
            ).feature
            evaluated.append({
                "source": str(proposal.source), "target": target, "sequence": sequence,
                "changed": changed, "raw_rank": raw_rank, "base_score": base_score,
                "diagnostics": diagnostics, "facility_feature": facility,
            })
        if not evaluated:
            return self._hold_result(started=started, active_target=active_target, learned=learned)

        targets = torch.stack([row["target"] for row in evaluated])
        raw_rank = torch.as_tensor([row["raw_rank"] for row in evaluated], dtype=targets.dtype, device=targets.device)
        base_scores = torch.as_tensor([row["base_score"] for row in evaluated], dtype=targets.dtype, device=targets.device)
        context_features, candidate_features = build_query_margin_features(
            current_state=(current_state[0] if current_state.ndim == 3 and int(current_state.shape[0]) == 1 else current_state),
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow[0],
            active_target=active_target,
            candidate_targets=targets,
            base_step2_scores_m3=base_scores,
            candidate_sources=[row["source"] for row in evaluated],
            supervisory_mask=self.supervisory_mask,
            target_scale_m3=float(self.policy_return_model.target_scale_m3.detach().cpu()),
        )
        with torch.no_grad():
            joint = self.rank_adapter(
                raw_rank_scores_m3=raw_rank,
                context_features=context_features,
                candidate_features=candidate_features,
            )
        relative = joint.relative_rank_normalized
        selected = int(torch.argmin(relative).item())
        row = evaluated[selected]
        facility_features = torch.stack([item["facility_feature"] for item in evaluated])
        delta = (row["target"] - active_target)[torch.as_tensor(self.supervisory_mask, dtype=torch.bool, device=active_target.device)]
        action_mass = torch.sqrt(torch.mean(torch.square(delta)))
        parts = build_selected_portfolio_feature_v21(
            candidate_features=facility_features,
            rank_scores=relative,
            selected_index=selected,
            selected_action_mass=action_mass,
        )
        with torch.no_grad():
            boundary = self.boundary_calibrator.predict(parts)
        score = float(boundary.hold_score.detach().cpu())
        advantage = float(boundary.advantage_m3.detach().cpu())
        passed = bool(score < BOUNDARY_ZERO)
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        executed = row["sequence"] if passed else hold
        changed_indices = torch.nonzero(torch.abs(row["target"] - active_target) > 1.0e-7).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        diagnostics = row["diagnostics"]
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=row["sequence"],
            predicted_delta_tfv_m3=advantage if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=advantage,
            selected_source=(f"OPERATIONAL_V21::{row['source']}" if passed else "LATCH_PREVIOUS_TARGET_OPERATIONAL_V21_BOUNDARY_NONNEGATIVE"),
            candidate_valid=passed,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=advantage,
            admission_margin_kind="development_raw_v21_zero_boundary",
            admission_passed=passed,
            calibrated_admission_contract=OPERATIONAL_RAW_BOUNDARY_ADMISSION,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=82,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=int(row["changed"]),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=int(row["changed"]) if passed else 0,
            maximum_support_ratio=self._first_move_support_ratio(row["target"], active_target),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(diagnostics["h120_total_variation_l1"]),
            joint_sequence_support_max_ratio=float(diagnostics["max_ratio"]),
            joint_sequence_support_binding=bool(diagnostics["binding"]),
            policy_return_predicted_delta_tfv_m3=advantage,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=advantage,
            policy_return_admission_passed=passed,
            policy_return_admission_contract=OPERATIONAL_RAW_BOUNDARY_ADMISSION,
            policy_return_estimand=DIRECT_TFV_POLICY_RETURN_ESTIMAND,
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=row["source"] if passed else "HOLD",
            policy_return_portfolio_sources=tuple(item["source"] for item in evaluated),
            policy_return_portfolio_scores_m3=tuple(float(item["raw_rank"]) for item in evaluated),
            policy_return_portfolio_upper_bounds_m3=tuple(float(item["raw_rank"]) for item in evaluated),
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
            scipy_message="NOT_USED_OPERATIONAL_V21_FINITE_PORTFOLIO",
        )


def build_operational_v21_controller(
    *, graph_path: str | Path, sensors_path: str | Path, config_path: str | Path,
    step1_path: str | Path, step2_path: str | Path, supervisory_control_path: str | Path,
    sequence_support_path: str | Path, v15_rank_checkpoint_path: str | Path,
    v21_boundary_checkpoint_path: str | Path, device: torch.device,
    decision_runtime_budget_seconds: float = 180.0, proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    graph = _load_graph(graph_path)
    sensors = _load_lines(sensors_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    base_model, base_norm, base = load_direct_tfv_runtime_checkpoint(step2_path, graph=graph, device=device)
    control, mask = load_native_supervisory_control(supervisory_control_path, actuator_ids=graph.actuator_ids)
    support = json.loads(Path(sequence_support_path).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(support, actuator_ids=graph.actuator_ids, step2_checkpoint_sha256=_sha(step2_path), supervisory_mask=mask, supervisory_control_contract=str(control["contract"]))
    rank_model, rank_norm, rank_adapter, rank_checkpoint = load_query_margin_checkpoint(v15_rank_checkpoint_path, graph=graph, base_step2_path=step2_path, device=device)
    boundary, boundary_checkpoint = load_v21_calibrator(v21_boundary_checkpoint_path, device=device)
    if str(boundary_checkpoint.get("base_step2_sha256", "")).lower() != _sha(step2_path).lower():
        raise ValueError("V21 boundary/base Step2 lineage mismatch")
    if str(boundary_checkpoint.get("rank_source_checkpoint_sha256", "")).lower() != _sha(v15_rank_checkpoint_path).lower():
        raise ValueError("V21 boundary/V15 rank lineage mismatch")
    if str(boundary_checkpoint.get("supervisory_mask_sha256", "")).lower() != str(control["supervisory_mask_sha256"]).lower():
        raise ValueError("V21 boundary uses another supervisory mask")
    parent = str(boundary_checkpoint.get("continuation_policy_sha256", "")).lower()
    if len(parent) != 64:
        raise ValueError("V21 boundary lacks continuation-policy lineage")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72, control_block_steps=2, max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_OPERATIONAL_V21_DEVELOPMENT_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(maxiter=1, deadline_seconds=30.0, active_facility_count=0, active_support_quantile="q95")
    mpc = DirectTFVOperationalV21MPC(
        model=base_model, graph=graph, normalization=base_norm, action_support=base["action_support"],
        sequence_support=support, supervisory_mask=mask, policy_return_model=rank_model,
        policy_return_normalization=rank_norm, rank_adapter=rank_adapter, boundary_calibrator=boundary,
        continuation_policy_sha256=parent, design=design, proposal_probe_chunk_size=int(proposal_probe_chunk_size),
    )
    inner = PortfolioMemorySafeDirectTFVAuthoritativeController(
        step1=step1, mpc=mpc, graph=graph, sensor_nodes=sensors,
        forecast=PersistenceDecayForecast(decay_per_step=0.92, scenario_multipliers=(0.8, 1.0, 1.2), history_steps_for_level=3),
        config=controller_cfg, device=device,
    )
    controller = ContinuityGuardController(inner, max_delta_per_update=0.5, allow_projection=False, enforce_current_delta=False)
    lineage = {
        "contract": OPERATIONAL_V21_RUNTIME_CONTRACT,
        "development_only": True,
        "formal_promotion_forbidden": True,
        "offline_step3_gate_bypassed_for_operational_steering": True,
        "v21_train_oof_supported": bool(boundary_checkpoint.get("train_oof_boundary_supported")),
        "v21_validation_truth_loaded": bool(boundary_checkpoint.get("validation_truth_loaded", False)),
        "v15_rank_checkpoint_sha256": _sha(v15_rank_checkpoint_path),
        "v21_boundary_checkpoint_sha256": _sha(v21_boundary_checkpoint_path),
        "base_step2_sha256": _sha(step2_path),
        "supervisory_mask_sha256": str(control["supervisory_mask_sha256"]),
        "candidate_portfolio_families": list(CURRENT_THREE_FAMILY_PORTFOLIO),
        "candidate_selection_uses_frozen_v15_rank": True,
        "hold_decision_uses_v21_raw_zero_boundary": True,
        "conformal_admission_used": False,
        "online_swmm_candidate_search": False,
        "online_lbfgsb_used": False,
        "projected_gradient_h10_enabled": False,
        "ready_for_policy_lock": False,
        "rank_checkpoint_contract": rank_checkpoint.get("contract"),
    }
    return controller, graph, sensors, lineage


__all__ = [
    "CURRENT_THREE_FAMILY_PORTFOLIO",
    "DirectTFVOperationalV21MPC",
    "OPERATIONAL_RAW_BOUNDARY_ADMISSION",
    "OPERATIONAL_V21_RUNTIME_CONTRACT",
    "build_operational_v21_controller",
    "load_v21_calibrator",
]
