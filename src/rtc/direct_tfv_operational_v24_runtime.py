"""Development-only V24 RBC-informed discriminative RTC runtime.

V24 preserves the frozen V23 learned path (Step1 -> Direct-TFV V5 -> V15 ranking -> V21
selected/HOLD boundary) and changes only what happens after a learned HOLD decision.  A HOLD
remains valid under low/moderate hydraulic stress.  When the sparse-sensing Step1 reconstruction
indicates broadly high actuator-upstream filling, V24 executes the existing type-aware hydraulic
candidate as a deterministic engineering stress escape instead of allowing repeated HOLD decisions.

The stress escape deliberately uses only causal reconstructed hydraulic state and the already-built
V23 candidate.  It does not use future realized rainfall, future SWMM truth, online SWMM candidate
search, gradient search, or Final outcomes.  The escape candidate remains inside the same q95 joint
sequence support, changed-facility ceiling, supervisory mask, type-aware SWMM release-setting
semantics, and 0.5 target-command slew used by V23.

This is a new controller version.  It must not rewrite the historical locked V23 evidence and is not
Formal evidence until a fresh V24 pre-lock validation and policy lock have been completed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .direct_tfv_operational_v23_runtime import (
    DirectTFVOperationalV23MPC,
    build_operational_v23_controller,
)
from .direct_tfv_policy_return_portfolio_v23 import (
    HYDRAULIC_SOURCE,
    build_hybrid_policy_return_portfolio_v23,
)


OPERATIONAL_V24_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V24_RBC_INFORMED_STRESS_ESCAPE_V1"
)
V24_HYBRID_ADMISSION_CONTRACT = (
    "PROJECT7_V24_LEARNED_V21_OR_CAUSAL_HYDRAULIC_STRESS_ESCAPE_V1"
)
# Auto-RBC's frozen high-fill threshold is 0.75.  Reusing that engineering threshold avoids
# tuning the escape trigger to V23 Final outcomes.  In the V23 smooth stress gate this corresponds
# to a strong-storm blend of 0.40.
V24_STRESS_ESCAPE_Q75 = 0.75
V24_STRESS_ESCAPE_BLEND_MIN = 0.40


def hydraulic_stress_escape_active(
    network_stress_q75: float,
    strong_storm_blend: float,
    *,
    stress_threshold: float = V24_STRESS_ESCAPE_Q75,
    blend_minimum: float = V24_STRESS_ESCAPE_BLEND_MIN,
) -> bool:
    """Return whether a learned HOLD may be replaced by the hydraulic engineering prior."""
    stress = float(network_stress_q75)
    blend = float(strong_storm_blend)
    if not 0.0 <= float(stress_threshold) <= 1.5:
        raise ValueError("V24 stress threshold must lie in [0,1.5]")
    if not 0.0 <= float(blend_minimum) <= 1.0:
        raise ValueError("V24 blend minimum must lie in [0,1]")
    return bool(stress >= float(stress_threshold) and blend >= float(blend_minimum))


class DirectTFVOperationalV24StressEscapeMPC:
    """Wrap frozen V23 and replace only pathological high-stress HOLD with a hydraulic action."""

    policy_mode = "operational_development_v24_rbc_informed_stress_escape"
    policy_mode_contract = OPERATIONAL_V24_RUNTIME_CONTRACT

    def __init__(
        self,
        inner: DirectTFVOperationalV23MPC,
        *,
        stress_threshold: float = V24_STRESS_ESCAPE_Q75,
        blend_minimum: float = V24_STRESS_ESCAPE_BLEND_MIN,
    ) -> None:
        if not isinstance(inner, DirectTFVOperationalV23MPC):
            raise TypeError("V24 requires the frozen V23 MPC as its learned parent")
        self.inner = inner
        self.stress_threshold = float(stress_threshold)
        self.blend_minimum = float(blend_minimum)
        # Validate once at construction.
        hydraulic_stress_escape_active(
            0.0,
            0.0,
            stress_threshold=self.stress_threshold,
            blend_minimum=self.blend_minimum,
        )

    def __getattr__(self, name: str) -> Any:
        # Keep the optimizer-agnostic authoritative adapter compatible with the wrapped V23 object.
        return getattr(self.inner, name)

    def optimize(self, **kwargs: Any):
        learned_result = self.inner.optimize(**kwargs)
        if bool(learned_result.admission_passed):
            return replace(
                learned_result,
                policy_mode=self.policy_mode,
                policy_mode_contract=self.policy_mode_contract,
                scipy_message=(
                    str(learned_result.scipy_message)
                    + "|v24_path=LEARNED_V21_ACTION|hydraulic_escape=false"
                ),
            )

        active_target = kwargs.get("active_target")
        current_state = kwargs.get("current_state")
        rainfall = kwargs.get("rainfall")
        flow = kwargs.get("previous_actuator_flow")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V24 requires active_target [109]")
        if not isinstance(current_state, torch.Tensor):
            raise ValueError("operational V24 requires causal reconstructed current_state")
        if not isinstance(rainfall, torch.Tensor):
            raise ValueError("operational V24 requires causal rainfall scenarios")
        if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (1, 109):
            raise ValueError("operational V24 requires previous_actuator_flow [1,109]")

        ceiling = self.inner.active_support_ceiling()
        hybrid = build_hybrid_policy_return_portfolio_v23(
            model=self.inner.model,
            normalization=self.inner.normalization,
            graph=self.inner.graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active_target,
            first_radius=self.inner.first_radius,
            max_changed_facilities=int(ceiling),
            max_delta_per_update=float(self.inner.design.max_setting_delta_per_update),
            probe_chunk_size=int(self.inner.proposal_probe_chunk_size),
            supervisory_mask=self.inner.supervisory_mask,
        )
        hydraulic = hybrid.hydraulic_diagnostics
        escape = hydraulic_stress_escape_active(
            hydraulic.network_stress_q75,
            hydraulic.strong_storm_blend,
            stress_threshold=self.stress_threshold,
            blend_minimum=self.blend_minimum,
        )
        if not escape:
            return replace(
                learned_result,
                policy_mode=self.policy_mode,
                policy_mode_contract=self.policy_mode_contract,
                scipy_message=(
                    str(learned_result.scipy_message)
                    + "|v24_path=HOLD_LOW_MODERATE_STRESS"
                    + f"|escape_stress_threshold={self.stress_threshold:.6g}"
                    + f"|escape_blend_minimum={self.blend_minimum:.6g}"
                ),
            )

        proposal = next(
            (row for row in hybrid.candidates if str(row.source) == HYDRAULIC_SOURCE),
            None,
        )
        if proposal is None:
            return replace(
                learned_result,
                policy_mode=self.policy_mode,
                policy_mode_contract=self.policy_mode_contract,
                scipy_message=(
                    str(learned_result.scipy_message)
                    + "|v24_path=HOLD_NO_HYDRAULIC_ESCAPE_CANDIDATE"
                ),
            )

        target, sequence, changed, support = self.inner._h10_supported_target(
            proposal.target,
            active_target,
        )
        if int(changed) <= 0:
            return replace(
                learned_result,
                policy_mode=self.policy_mode,
                policy_mode_contract=self.policy_mode_contract,
                scipy_message=(
                    str(learned_result.scipy_message)
                    + "|v24_path=HOLD_ZERO_CHANGE_HYDRAULIC_ESCAPE"
                ),
            )

        passive = torch.as_tensor(
            ~self.inner.supervisory_mask,
            dtype=torch.bool,
            device=active_target.device,
        )
        if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
            raise RuntimeError("V24 hydraulic stress escape changed a passive channel")
        changed_indices = torch.nonzero(
            torch.abs(target - active_target) > 1.0e-7
        ).reshape(-1).tolist()
        changed_ids = tuple(str(self.inner.graph.actuator_ids[int(i)]) for i in changed_indices)

        # The learned V21 boundary explicitly rejected ACTION.  V24 therefore does not relabel the
        # engineering escape as a surrogate-predicted TFV improvement.  Prediction fields are zeroed
        # and the admission contract states that this action came from the causal hydraulic prior.
        return replace(
            learned_result,
            settings=sequence,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=0.0,
            raw_optimized_predicted_delta_tfv_m3=0.0,
            selected_source=f"OPERATIONAL_V24_HYDRAULIC_STRESS_ESCAPE::{HYDRAULIC_SOURCE}",
            candidate_valid=True,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=0.0,
            admission_margin_kind="causal_hydraulic_stress_escape_not_surrogate_calibrated",
            admission_passed=True,
            calibrated_admission_contract=V24_HYBRID_ADMISSION_CONTRACT,
            active_facility_count=int(changed),
            active_facility_ids=changed_ids,
            first_move_changed_facility_count=int(changed),
            maximum_support_ratio=self.inner._first_move_support_ratio(target, active_target),
            joint_sequence_support_quantile=str(support["quantile"]),
            joint_sequence_first_block_l1=float(support["first_block_l1"]),
            joint_sequence_h120_l1=float(support["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(
                support["h120_total_variation_l1"]
            ),
            joint_sequence_support_max_ratio=float(support["max_ratio"]),
            joint_sequence_support_binding=bool(support["binding"]),
            policy_return_predicted_delta_tfv_m3=0.0,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=0.0,
            policy_return_admission_passed=True,
            policy_return_admission_contract=V24_HYBRID_ADMISSION_CONTRACT,
            policy_return_portfolio_selected_source=HYDRAULIC_SOURCE,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_changed_facility_count=int(changed),
            refined_first_move_changed_facility_ids=changed_ids,
            scipy_message=(
                "NOT_USED_OPERATIONAL_V24_FINITE_PORTFOLIO"
                "|v24_path=RBC_INFORMED_HYDRAULIC_STRESS_ESCAPE"
                "|learned_v21_boundary_passed=false"
                f"|stress={hydraulic.network_stress_q75:.6g}"
                f"|blend={hydraulic.strong_storm_blend:.6g}"
                f"|escape_stress_threshold={self.stress_threshold:.6g}"
                f"|escape_blend_minimum={self.blend_minimum:.6g}"
            ),
        )


def build_operational_v24_controller(
    *,
    graph_path,
    sensors_path,
    config_path,
    step1_path,
    step2_path,
    supervisory_control_path,
    sequence_support_path,
    v15_rank_checkpoint_path,
    v21_boundary_checkpoint_path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    """Build V23 unchanged, then wrap only its MPC admission with the V24 stress escape."""
    controller, graph, sensors, lineage = build_operational_v23_controller(
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
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(proposal_probe_chunk_size),
    )
    portfolio_controller = getattr(controller, "controller", None)
    adapter = getattr(portfolio_controller, "_direct_mpc_adapter", None)
    base_mpc = getattr(adapter, "inner", None)
    if not isinstance(base_mpc, DirectTFVOperationalV23MPC):
        raise RuntimeError("V24 could not locate the frozen V23 MPC inside the authoritative runtime")
    adapter.inner = DirectTFVOperationalV24StressEscapeMPC(base_mpc)

    v24_lineage = dict(lineage)
    v24_lineage.update(
        {
            "contract": OPERATIONAL_V24_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_promotion_forbidden_until_fresh_v24_validation_and_policy_lock": True,
            "historical_v23_policy_mutated": False,
            "v23_parent_runtime_contract": lineage.get("contract"),
            "v23_learned_path_preserved": True,
            "step2_role": "DISCRIMINATIVE_INTERNAL_REPRESENTATION_NOT_STANDALONE_SIMULATOR",
            "hold_semantics": "PERMITTED_LOW_MODERATE_STRESS_ESCAPED_AT_BROAD_HIGH_STRESS",
            "hydraulic_stress_escape_enabled": True,
            "hydraulic_stress_escape_q75": V24_STRESS_ESCAPE_Q75,
            "hydraulic_stress_escape_blend_minimum": V24_STRESS_ESCAPE_BLEND_MIN,
            "hydraulic_stress_escape_source": HYDRAULIC_SOURCE,
            "hydraulic_escape_uses_future_truth": False,
            "hydraulic_escape_uses_online_swmm_search": False,
            "hydraulic_escape_preserves_q95_support": True,
            "hydraulic_escape_preserves_supervisory_mask": True,
            "hydraulic_escape_preserves_target_slew": True,
            "ready_for_policy_lock": False,
        }
    )
    return controller, graph, sensors, v24_lineage


__all__ = [
    "DirectTFVOperationalV24StressEscapeMPC",
    "OPERATIONAL_V24_RUNTIME_CONTRACT",
    "V24_HYBRID_ADMISSION_CONTRACT",
    "V24_STRESS_ESCAPE_BLEND_MIN",
    "V24_STRESS_ESCAPE_Q75",
    "build_operational_v24_controller",
    "hydraulic_stress_escape_active",
]
