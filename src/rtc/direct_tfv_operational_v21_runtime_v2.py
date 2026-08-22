"""Corrected Development-only operational runtime for frozen V15 rank + V21 boundary.

V1 incorrectly routed the V15 selection-consistent rank checkpoint through the later full
query-margin loader and also constructed the retired 11/9 query features online.  V2 restores exact
training/runtime parity without changing any learned checkpoint: the frozen base Step2 model is used
for raw rank scores and latent features, the validated V15 rank branch is imported through its
rank-only contract, and V21 alone decides selected-candidate versus HOLD.  This remains operational
Development evidence only; it cannot be promoted to Policy Lock or Final evidence.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .controller_direct_tfv_portfolio import PortfolioMemorySafeDirectTFVAuthoritativeController
from .direct_tfv_operational_v21_runtime import (
    CURRENT_THREE_FAMILY_PORTFOLIO,
    OPERATIONAL_RAW_BOUNDARY_ADMISSION,
    DirectTFVOperationalV21MPC,
    load_v21_calibrator,
)
from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
)
from .direct_tfv_policy_return_facility_boundary_v20 import build_facility_boundary_parts_v20
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_policy_return_query_margin_v2 import build_query_margin_v2_features
from .direct_tfv_policy_return_query_margin_v17 import (
    LEGACY_V15_CHECKPOINT_CONTRACT,
    QueryConditionedPolicyReturnAdapterV17,
    import_v15_rank_state,
    rank_state_sha256,
)
from .direct_tfv_policy_return_selected_boundary_v21 import (
    BOUNDARY_ZERO,
    build_selected_portfolio_feature_v21,
)
from .direct_tfv_sequence_support import (
    changed_facility_support_limit,
    validate_direct_tfv_sequence_support,
)
from .forecast import PersistenceDecayForecast
from .native_supervisory_control import load_native_supervisory_control
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


OPERATIONAL_V21_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V21_RAW_BOUNDARY_RTC_V2_V15_RANK_PARITY"
)
RANK_REUSE_ACCURACY_MIN = 0.90


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v15_rank_only_adapter(
    path: str | Path,
    *,
    target_scale_m3: float,
    base_step2_sha256: str,
    device: torch.device,
) -> tuple[QueryConditionedPolicyReturnAdapterV17, dict[str, Any]]:
    """Load only the frozen V15 rank path; never require/import its failed HOLD margin heads."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("contract") != LEGACY_V15_CHECKPOINT_CONTRACT:
        raise ValueError("operational V21 V2 requires the V15 selection-consistent rank source")
    if str(payload.get("base_step2_sha256", "")).lower() != str(base_step2_sha256).lower():
        raise ValueError("V15 rank source/base Step2 lineage mismatch")
    metrics = payload.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("V15 rank source lacks its frozen validation metrics")
    if float(metrics.get("within_query_pairwise_rank_accuracy", 0.0)) < RANK_REUSE_ACCURACY_MIN:
        raise ValueError("V15 rank source pairwise accuracy is below the frozen reuse threshold")
    if float(metrics.get("within_query_candidate_top1_accuracy", 0.0)) < RANK_REUSE_ACCURACY_MIN:
        raise ValueError("V15 rank source top1 accuracy is below the frozen reuse threshold")
    context_dim = int(payload.get("context_dim", 0))
    candidate_dim = int(payload.get("candidate_dim", 0))
    if min(context_dim, candidate_dim) <= 0:
        raise ValueError("V15 rank source lacks latent feature dimensions")
    adapter = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=float(target_scale_m3),
        context_dim=context_dim,
        candidate_dim=candidate_dim,
    ).to(device)
    import_v15_rank_state(adapter, payload)
    adapter.eval()
    return adapter, payload


class DirectTFVOperationalV21MPCV2(DirectTFVOperationalV21MPC):
    """V21 operational MPC with exact V15 latent-rank feature parity."""

    policy_mode = "operational_development_v21_selected_boundary_v2"
    policy_mode_contract = OPERATIONAL_V21_RUNTIME_CONTRACT

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V21 V2 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("operational V21 V2 requires previous_actuator_flow [1,109]")

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
        passive = torch.as_tensor(
            ~self.supervisory_mask,
            dtype=torch.bool,
            device=active_target.device,
        )
        for proposal in hybrid.candidates:
            target, sequence, changed, diagnostics = self._h10_supported_target(
                proposal.target,
                active_target,
            )
            if changed <= 0:
                continue
            if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("operational V21 V2 candidate changed a passive channel")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            raw_rank = float(
                self._score_policy_return_target(
                    current_state=current_state,
                    rainfall=rainfall,
                    previous_actuator_flow=flow,
                    active_target=active_target,
                    candidate_target=target,
                )
                .detach()
                .cpu()
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
                )[0]
                .detach()
                .cpu()
            )
            facility = build_facility_boundary_parts_v20(
                step2_model=self.model,
                normalization=self.normalization,
                graph=self.graph,
                current_state=(
                    current_state[0]
                    if current_state.ndim == 3 and int(current_state.shape[0]) == 1
                    else current_state
                ),
                rainfall_scenarios=rainfall,
                previous_actuator_flow=flow[0],
                active_target=active_target,
                candidate_target=target,
                candidate_source=str(proposal.source),
                supervisory_mask=self.supervisory_mask,
                target_scale_m3=float(self.model.target_scale_m3.detach().cpu()),
            ).feature
            evaluated.append(
                {
                    "source": str(proposal.source),
                    "target": target,
                    "sequence": sequence,
                    "changed": changed,
                    "raw_rank": raw_rank,
                    "base_score": base_score,
                    "diagnostics": diagnostics,
                    "facility_feature": facility,
                }
            )
        if not evaluated:
            return self._hold_result(
                started=started,
                active_target=active_target,
                learned=learned,
            )

        targets = torch.stack([row["target"] for row in evaluated])
        raw_rank = torch.as_tensor(
            [row["raw_rank"] for row in evaluated],
            dtype=targets.dtype,
            device=targets.device,
        )
        base_scores = torch.as_tensor(
            [row["base_score"] for row in evaluated],
            dtype=targets.dtype,
            device=targets.device,
        )
        context_features, candidate_features = build_query_margin_v2_features(
            step2_model=self.model,
            normalization=self.normalization,
            graph=self.graph,
            current_state=(
                current_state[0]
                if current_state.ndim == 3 and int(current_state.shape[0]) == 1
                else current_state
            ),
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow[0],
            active_target=active_target,
            candidate_targets=targets,
            base_step2_scores_m3=base_scores,
            candidate_sources=[row["source"] for row in evaluated],
            supervisory_mask=self.supervisory_mask,
            target_scale_m3=float(self.model.target_scale_m3.detach().cpu()),
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
        facility_features = torch.stack(
            [item["facility_feature"] for item in evaluated]
        )
        mask = torch.as_tensor(
            self.supervisory_mask,
            dtype=torch.bool,
            device=active_target.device,
        )
        delta = (row["target"] - active_target)[mask]
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
        hold = active_target.reshape(1, 109).expand(
            int(self.design.prediction_horizon_steps), -1
        ).detach()
        executed = row["sequence"] if passed else hold
        changed_indices = torch.nonzero(
            torch.abs(row["target"] - active_target) > 1.0e-7
        ).reshape(-1).tolist()
        changed_ids = tuple(
            str(self.graph.actuator_ids[int(i)]) for i in changed_indices
        )
        diagnostics = row["diagnostics"]
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=row["sequence"],
            predicted_delta_tfv_m3=advantage if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=advantage,
            selected_source=(
                f"OPERATIONAL_V21_V2::{row['source']}"
                if passed
                else "LATCH_PREVIOUS_TARGET_OPERATIONAL_V21_V2_BOUNDARY_NONNEGATIVE"
            ),
            candidate_valid=passed,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=advantage,
            admission_margin_kind="development_raw_v21_zero_boundary_v2",
            admission_passed=passed,
            calibrated_admission_contract=OPERATIONAL_RAW_BOUNDARY_ADMISSION,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=82,
            predicted_beneficial_facility_count=int(
                learned.predicted_beneficial_facility_count
            ),
            active_facility_count=int(row["changed"]),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=int(row["changed"]) if passed else 0,
            maximum_support_ratio=self._first_move_support_ratio(
                row["target"], active_target
            ),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(
                diagnostics["h120_total_variation_l1"]
            ),
            joint_sequence_support_max_ratio=float(diagnostics["max_ratio"]),
            joint_sequence_support_binding=bool(diagnostics["binding"]),
            policy_return_predicted_delta_tfv_m3=advantage,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=advantage,
            policy_return_admission_passed=passed,
            policy_return_admission_contract=OPERATIONAL_RAW_BOUNDARY_ADMISSION,
            policy_return_estimand=DIRECT_TFV_POLICY_RETURN_ESTIMAND,
            policy_return_parent_continuation_sha256=(
                self.policy_return_parent_continuation_sha256
            ),
            policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=(
                row["source"] if passed else "HOLD"
            ),
            policy_return_portfolio_sources=tuple(
                item["source"] for item in evaluated
            ),
            policy_return_portfolio_scores_m3=tuple(
                float(item["raw_rank"]) for item in evaluated
            ),
            policy_return_portfolio_upper_bounds_m3=tuple(
                float(item["raw_rank"]) for item in evaluated
            ),
            policy_return_portfolio_base_step2_scores_m3=tuple(
                float(item["base_score"]) for item in evaluated
            ),
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
            scipy_message="NOT_USED_OPERATIONAL_V21_V2_FINITE_PORTFOLIO",
        )


def build_operational_v21_controller(
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
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    graph = _load_graph(graph_path)
    sensors = _load_lines(sensors_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    base_model, base_norm, base = load_direct_tfv_runtime_checkpoint(
        step2_path,
        graph=graph,
        device=device,
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    step2_sha = _sha(step2_path)

    control, mask = load_native_supervisory_control(
        supervisory_control_path,
        actuator_ids=graph.actuator_ids,
    )
    support = json.loads(Path(sequence_support_path).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=step2_sha,
        supervisory_mask=mask,
        supervisory_control_contract=str(control["contract"]),
    )
    rank_adapter, rank_checkpoint = load_v15_rank_only_adapter(
        v15_rank_checkpoint_path,
        target_scale_m3=float(base_model.target_scale_m3.detach().cpu()),
        base_step2_sha256=step2_sha,
        device=device,
    )
    boundary, boundary_checkpoint = load_v21_calibrator(
        v21_boundary_checkpoint_path,
        device=device,
    )
    if str(boundary_checkpoint.get("base_step2_sha256", "")).lower() != step2_sha.lower():
        raise ValueError("V21 boundary/base Step2 lineage mismatch")
    rank_source_sha = _sha(v15_rank_checkpoint_path)
    if str(boundary_checkpoint.get("rank_source_checkpoint_sha256", "")).lower() != rank_source_sha.lower():
        raise ValueError("V21 boundary/V15 rank lineage mismatch")
    if str(boundary_checkpoint.get("supervisory_mask_sha256", "")).lower() != str(
        control["supervisory_mask_sha256"]
    ).lower():
        raise ValueError("V21 boundary uses another supervisory mask")
    imported_rank_sha = rank_state_sha256(rank_adapter)
    expected_rank_sha = str(
        boundary_checkpoint.get("rank_branch_state_sha256_before_boundary", "")
    ).lower()
    if len(expected_rank_sha) != 64 or imported_rank_sha.lower() != expected_rank_sha:
        raise ValueError("runtime V15 rank state does not reproduce the V21 training rank")
    expected_after = str(
        boundary_checkpoint.get("rank_branch_state_sha256_after_boundary", expected_rank_sha)
    ).lower()
    if expected_after != expected_rank_sha:
        raise ValueError("V21 checkpoint reports rank mutation during boundary fitting")
    parent = str(boundary_checkpoint.get("continuation_policy_sha256", "")).lower()
    if len(parent) != 64:
        raise ValueError("V21 boundary lacks continuation-policy lineage")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_OPERATIONAL_V21_V2_DEVELOPMENT_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVOperationalV21MPCV2(
        model=base_model,
        graph=graph,
        normalization=base_norm,
        action_support=base["action_support"],
        sequence_support=support,
        supervisory_mask=mask,
        policy_return_model=base_model,
        policy_return_normalization=base_norm,
        rank_adapter=rank_adapter,
        boundary_calibrator=boundary,
        continuation_policy_sha256=parent,
        design=design,
        proposal_probe_chunk_size=int(proposal_probe_chunk_size),
    )
    inner = PortfolioMemorySafeDirectTFVAuthoritativeController(
        step1=step1,
        mpc=mpc,
        graph=graph,
        sensor_nodes=sensors,
        forecast=PersistenceDecayForecast(
            decay_per_step=0.92,
            scenario_multipliers=(0.8, 1.0, 1.2),
            history_steps_for_level=3,
        ),
        config=controller_cfg,
        device=device,
    )
    controller = ContinuityGuardController(
        inner,
        max_delta_per_update=0.5,
        allow_projection=False,
        enforce_current_delta=False,
    )
    lineage = {
        "contract": OPERATIONAL_V21_RUNTIME_CONTRACT,
        "development_only": True,
        "formal_promotion_forbidden": True,
        "offline_step3_gate_bypassed_for_operational_steering": True,
        "v21_train_oof_supported": bool(
            boundary_checkpoint.get("train_oof_boundary_supported")
        ),
        "v21_validation_truth_loaded": bool(
            boundary_checkpoint.get("validation_truth_loaded", False)
        ),
        "v15_rank_checkpoint_sha256": rank_source_sha,
        "v15_rank_checkpoint_contract": rank_checkpoint.get("contract"),
        "v15_rank_loaded_rank_only": True,
        "v15_failed_margin_heads_loaded": False,
        "v15_rank_state_sha256": imported_rank_sha,
        "rank_feature_contract": "FROZEN_STEP2_LATENT_V2_TRAIN_RUNTIME_PARITY",
        "rank_feature_builder": "build_query_margin_v2_features",
        "base_step2_sha256": step2_sha,
        "base_step2_frozen_at_runtime": True,
        "v21_boundary_checkpoint_sha256": _sha(v21_boundary_checkpoint_path),
        "supervisory_mask_sha256": str(control["supervisory_mask_sha256"]),
        "candidate_portfolio_families": list(CURRENT_THREE_FAMILY_PORTFOLIO),
        "candidate_selection_uses_frozen_v15_rank": True,
        "hold_decision_uses_v21_raw_zero_boundary": True,
        "conformal_admission_used": False,
        "online_swmm_candidate_search": False,
        "online_lbfgsb_used": False,
        "projected_gradient_h10_enabled": False,
        "ready_for_policy_lock": False,
    }
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVOperationalV21MPCV2",
    "OPERATIONAL_V21_RUNTIME_CONTRACT",
    "build_operational_v21_controller",
    "load_v15_rank_only_adapter",
]
