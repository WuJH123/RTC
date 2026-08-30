"""Project7 V30 conservative policy-improvement runtime.

V30 targets the current Step3 performance bottleneck rather than adding another candidate family.
V27 already contains a causal Auto-RBC shadow, but its learned value ranker can override that strong
candidate on the basis of one learned score.  V30 preserves the full V27 candidate search and then
places a conservative selector above it:

* Auto-RBC is reconstructed from the same sparse-Step1 state and the same engineering/q95 envelope.
* V27's best supported action is the sole learned challenger.
* The challenger may replace Auto-RBC only when both the frozen V27 exact-return value and the frozen
  base Step2 H10 score strictly prefer it.
* If Auto-RBC produces no action, the challenger must beat HOLD under both estimators.
* If both estimators rate Auto-RBC no better than HOLD, HOLD is allowed; one-model disagreement falls
  back to Auto-RBC.

This is Development-only conservative policy improvement, not a formal guarantee of lower SWMM TFV.
A fresh authoritative scientific gate is mandatory before a new Policy Lock.
"""
from __future__ import annotations

from dataclasses import replace
import json
import time
from typing import Any

import torch

from .direct_tfv_operational_v27_runtime import (
    DirectTFVOperationalV27MPC,
    V27_TFV_VALUE_ESTIMAND,
    build_operational_v27_controller,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_sequence_support import changed_facility_support_limit
from .direct_tfv_v27_auto_rbc_shadow import (
    V27_AUTO_RBC_SHADOW_CONTRACT,
    V27_AUTO_RBC_SHADOW_SOURCE,
    build_auto_rbc_shadow_proposal,
)
from .project7_conservative_policy_improvement import (
    CONSERVATIVE_SELECTOR_CONTRACT,
    CandidateScore,
    select_conservative_improvement,
)
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


OPERATIONAL_V30_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V30_RBC_ANCHORED_CONSERVATIVE_IMPROVEMENT_V1"
)
V30_SELECTION_CONTRACT = CONSERVATIVE_SELECTOR_CONTRACT


def _changed_target_ids(graph: Any, target: torch.Tensor, active_target: torch.Tensor) -> tuple[str, ...]:
    indices = torch.nonzero(torch.abs(target - active_target) > 1.0e-7).reshape(-1).tolist()
    return tuple(str(graph.actuator_ids[int(index)]) for index in indices)


def _v27_best_supported_source(parent: DirectTFVMPCResultV12) -> str:
    """Recover the exact V27 candidate-family label used to build its value feature.

    V27 deliberately reports ``policy_return_portfolio_selected_source=HOLD`` when its best latent
    score is non-negative even though ``optimized_candidate_settings`` still exposes that best
    supported action.  V30 may rescore this unexecuted challenger, so using an invented source label
    would silently change V27's family-indicator feature.  The V27 diagnostics contain the frozen
    ``supported_best`` source; failure to recover/validate it is fatal rather than guessed.
    """
    selected = str(parent.policy_return_portfolio_selected_source)
    if selected != "HOLD":
        return selected
    prefix = "supported_best="
    matches = [
        token[len(prefix) :]
        for token in str(parent.scipy_message).split("|")
        if token.startswith(prefix)
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError("V30 cannot recover V27 supported-best candidate source")
    source = matches[0]
    available = tuple(str(value) for value in parent.policy_return_portfolio_sources)
    if source not in available:
        raise RuntimeError("V30 recovered a V27 candidate source absent from its portfolio lineage")
    return source


class DirectTFVOperationalV30MPC(DirectTFVOperationalV27MPC):
    policy_mode = "operational_development_v30_rbc_anchored_conservative_improvement"
    policy_mode_contract = OPERATIONAL_V30_RUNTIME_CONTRACT

    def _score_target(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        flow: torch.Tensor,
        active_target: torch.Tensor,
        target: torch.Tensor,
        source: str,
    ) -> tuple[CandidateScore, float]:
        with torch.inference_mode():
            _, latent, reported, _ = self._feature_and_scores(
                current_state=current_state,
                rainfall=rainfall,
                flow=flow,
                active_target=active_target,
                target=target,
                source=source,
            )
            base = float(
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
        return (
            CandidateScore(
                source=source,
                latent_score=float(latent),
                base_step2_score_m3=float(base),
            ),
            float(reported),
        )

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        current_state = kwargs.get("current_state")
        rainfall = kwargs.get("rainfall")
        flow = kwargs.get("previous_actuator_flow")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V30 requires active_target [109]")
        if not isinstance(current_state, torch.Tensor):
            raise ValueError("operational V30 requires reconstructed current_state")
        if not isinstance(rainfall, torch.Tensor):
            raise ValueError("operational V30 requires causal rainfall scenarios")
        if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (1, 109):
            raise ValueError("operational V30 requires previous_actuator_flow [1,109]")

        # Keep V27 candidate generation/ranking unchanged. V30 changes only the final authority
        # granted to that learned choice relative to a same-information Auto-RBC anchor.
        parent = super().optimize(**kwargs)
        horizon = int(self.design.prediction_horizon_steps)
        hold = active_target.reshape(1, 109).expand(horizon, -1).detach()

        shadow = build_auto_rbc_shadow_proposal(
            graph=self.graph,
            current_state=current_state,
            active_target=active_target,
            supervisory_mask=self.supervisory_mask,
            first_radius=self.first_radius,
            max_changed_facilities=int(
                changed_facility_support_limit(self.sequence_support, "q95")
            ),
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
        )
        anchor_target, anchor_sequence, anchor_changed, anchor_support = self._h10_supported_target(
            shadow.target, active_target
        )
        anchor_score: CandidateScore | None = None
        anchor_reported = 0.0
        if int(anchor_changed) > 0:
            anchor_score, anchor_reported = self._score_target(
                current_state=current_state,
                rainfall=rainfall,
                flow=flow,
                active_target=active_target,
                target=anchor_target,
                source=V27_AUTO_RBC_SHADOW_SOURCE,
            )

        candidate_sequence = parent.optimized_candidate_settings.detach()
        if candidate_sequence.ndim != 2 or tuple(candidate_sequence.shape) != (horizon, 109):
            raise RuntimeError("V30 parent did not expose one supported H10 candidate sequence")
        candidate_target = candidate_sequence[0].detach()
        candidate_changed = int(
            torch.count_nonzero(torch.abs(candidate_target - active_target) > 1.0e-7).item()
        )
        same_as_anchor = bool(
            anchor_score is not None
            and torch.allclose(candidate_target, anchor_target, rtol=0.0, atol=1.0e-7)
        )
        challenger_score: CandidateScore | None = None
        challenger_reported = 0.0
        challenger_source: str | None = None
        if candidate_changed > 0 and not same_as_anchor:
            challenger_source = _v27_best_supported_source(parent)
            challenger_score, challenger_reported = self._score_target(
                current_state=current_state,
                rainfall=rainfall,
                flow=flow,
                active_target=active_target,
                target=candidate_target,
                source=challenger_source,
            )

        selection = select_conservative_improvement(
            challenger=challenger_score,
            anchor=anchor_score,
        )

        if selection.execute_hold:
            selected_target = active_target.detach()
            selected_sequence = hold
            selected_support = None
            selected_score = CandidateScore("HOLD", 0.0, 0.0)
            selected_reported = 0.0
            selected_changed = 0
            selected_ids: tuple[str, ...] = ()
        elif selection.challenger_selected:
            if challenger_score is None:
                raise RuntimeError("V30 selected a missing challenger")
            selected_target = candidate_target
            selected_sequence = candidate_sequence
            selected_support = {
                "quantile": parent.joint_sequence_support_quantile,
                "first_block_l1": parent.joint_sequence_first_block_l1,
                "h120_l1": parent.joint_sequence_h120_l1,
                "h120_total_variation_l1": parent.joint_sequence_h120_total_variation_l1,
                "max_ratio": parent.joint_sequence_support_max_ratio,
                "binding": parent.joint_sequence_support_binding,
            }
            selected_score = challenger_score
            selected_reported = challenger_reported
            selected_changed = candidate_changed
            selected_ids = _changed_target_ids(self.graph, candidate_target, active_target)
        else:
            if anchor_score is None or int(anchor_changed) <= 0:
                raise RuntimeError("V30 selected a missing Auto-RBC anchor")
            selected_target = anchor_target
            selected_sequence = anchor_sequence
            selected_support = anchor_support
            selected_score = anchor_score
            selected_reported = anchor_reported
            selected_changed = int(anchor_changed)
            selected_ids = _changed_target_ids(self.graph, anchor_target, active_target)

        if selected_support is None:
            support_quantile = "q95"
            support_first_l1 = 0.0
            support_h120_l1 = 0.0
            support_tv = 0.0
            support_ratio = 0.0
            support_binding = False
            maximum_support_ratio = 0.0
        else:
            support_quantile = str(selected_support["quantile"])
            support_first_l1 = float(selected_support["first_block_l1"])
            support_h120_l1 = float(selected_support["h120_l1"])
            support_tv = float(selected_support["h120_total_variation_l1"])
            support_ratio = float(selected_support["max_ratio"])
            support_binding = bool(selected_support["binding"])
            maximum_support_ratio = self._first_move_support_ratio(selected_target, active_target)

        diagnostics = {
            "contract": OPERATIONAL_V30_RUNTIME_CONTRACT,
            "selector_contract": V30_SELECTION_CONTRACT,
            "selection_mode": selection.mode,
            "selection_reason": selection.reason,
            "selected_source": selection.selected_source,
            "parent_v27_candidate_valid": bool(parent.candidate_valid),
            "parent_v27_selected_source": str(parent.selected_source),
            "challenger_source": challenger_source,
            "challenger_same_as_anchor": same_as_anchor,
            "challenger_latent": (
                None if challenger_score is None else float(challenger_score.latent_score)
            ),
            "challenger_base_step2_m3": (
                None if challenger_score is None else float(challenger_score.base_step2_score_m3)
            ),
            "anchor_changed": int(anchor_changed),
            "anchor_latent": None if anchor_score is None else float(anchor_score.latent_score),
            "anchor_base_step2_m3": (
                None if anchor_score is None else float(anchor_score.base_step2_score_m3)
            ),
            "selected_latent": float(selected_score.latent_score),
            "selected_base_step2_m3": float(selected_score.base_step2_score_m3),
            "final_outcomes_used_for_policy_selection": False,
        }
        diagnostics_text = "V30_CONSERVATIVE|" + json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":")
        )
        action = not selection.execute_hold

        return replace(
            parent,
            settings=selected_sequence,
            optimized_candidate_settings=selected_sequence,
            predicted_delta_tfv_m3=float(selected_reported) if action else 0.0,
            raw_optimized_predicted_delta_tfv_m3=float(selected_reported),
            selected_source=(
                f"OPERATIONAL_V30::{selection.selected_source}" if action else "HOLD::OPERATIONAL_V30"
            ),
            candidate_valid=bool(action),
            admission_upper_bound_m3=float(selected_reported),
            admission_margin_kind="v30_rbc_anchored_dual_estimator_consensus",
            admission_passed=bool(action),
            calibrated_admission_contract=V30_SELECTION_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            active_facility_count=int(selected_changed),
            active_facility_ids=selected_ids,
            first_move_changed_facility_count=int(selected_changed),
            maximum_support_ratio=float(maximum_support_ratio),
            joint_sequence_support_quantile=support_quantile,
            joint_sequence_first_block_l1=support_first_l1,
            joint_sequence_h120_l1=support_h120_l1,
            joint_sequence_h120_total_variation_l1=support_tv,
            joint_sequence_support_max_ratio=support_ratio,
            joint_sequence_support_binding=support_binding,
            policy_return_predicted_delta_tfv_m3=float(selected_reported),
            policy_return_upper_bound_m3=float(selected_reported),
            policy_return_admission_passed=bool(action),
            policy_return_admission_contract=V30_SELECTION_CONTRACT,
            policy_return_estimand=V27_TFV_VALUE_ESTIMAND,
            policy_return_portfolio_selected_source=selection.selected_source,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_changed_facility_count=int(selected_changed),
            refined_first_move_changed_facility_ids=selected_ids,
            scipy_message=diagnostics_text,
            tfv_value_available=True,
            tfv_value_prediction_m3=float(selected_reported),
            tfv_value_upper_bound_m3=float(selected_reported),
            tfv_value_admission_passed=bool(action),
            engineering_feasible=True,
            passive_channels_unchanged=True,
            sequence_support_valid=True,
            tfv_value_gate_reason=diagnostics_text,
        )


def build_operational_v30_controller(**kwargs: Any):
    controller, graph, sensors, lineage = build_operational_v27_controller(**kwargs)
    old = controller.controller._direct_mpc_adapter.inner
    if not isinstance(old, DirectTFVOperationalV27MPC):
        raise TypeError("V30 factory expected the V27 parent runtime")
    v30 = DirectTFVOperationalV30MPC(
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
        v27_value_model=old.v27_value_model,
        v27_lineage=old.v27_lineage,
    )
    controller.controller._direct_mpc_adapter.inner = v30
    updated = dict(lineage)
    updated.update(
        {
            "contract": OPERATIONAL_V30_RUNTIME_CONTRACT,
            "v30_selection_contract": V30_SELECTION_CONTRACT,
            "v30_parent_contract": old.policy_mode_contract,
            "v30_auto_rbc_anchor_contract": V27_AUTO_RBC_SHADOW_CONTRACT,
            "v30_anchor_same_sparse_step1_information": True,
            "v30_challenger_requires_dual_estimator_dominance": True,
            "v30_hold_requires_dual_estimator_consensus_when_anchor_action_exists": True,
            "v30_candidate_source_semantics_preserved": True,
            "v30_final_outcomes_used_for_policy_selection": False,
            "development_only": True,
            "formal_evidence": False,
            "requires_new_policy_lock": True,
            "ready_for_policy_lock": False,
        }
    )
    return controller, graph, sensors, updated


__all__ = [
    "DirectTFVOperationalV30MPC",
    "OPERATIONAL_V30_RUNTIME_CONTRACT",
    "V30_SELECTION_CONTRACT",
    "build_operational_v30_controller",
]
