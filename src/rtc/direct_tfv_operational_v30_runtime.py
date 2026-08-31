"""Project7 V30 objective-driven runtime.

V30 intentionally does not add a second policy-admission layer above V27. V27 already constructs an
engineering-supported candidate portfolio, evaluates every retained candidate with the same frozen
exact-return value model, and selects the minimum latent value against HOLD=0. V30 keeps that
scientifically transparent decision rule unchanged.

Auto-RBC, when present in the V27 portfolio, is a normal heuristic candidate only. It has no veto,
anchor, dominance, or fallback authority. Standalone Auto-RBC remains an independent comparator.
Historical completed evaluation outcomes may be inspected during development, but they are never fed
into this online runtime; model fitting/calibration and evaluation splits must remain disjoint.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .direct_tfv_operational_v27_runtime import (
    DirectTFVOperationalV27MPC,
    V27_SELECTION_CONTRACT,
    build_operational_v27_controller,
)


OPERATIONAL_V30_RUNTIME_CONTRACT = "PROJECT7_OPERATIONAL_V30_OBJECTIVE_DRIVEN_PORTFOLIO_V1"
V30_SELECTION_CONTRACT = "PROJECT7_V30_ARGMIN_FROZEN_EXACT_RETURN_PLUS_HOLD_ZERO_V1"


class DirectTFVOperationalV30MPC(DirectTFVOperationalV27MPC):
    """Thin publication-candidate wrapper around the V27 objective-driven selector."""

    policy_mode = "operational_v30_objective_driven_portfolio"
    policy_mode_contract = OPERATIONAL_V30_RUNTIME_CONTRACT

    def optimize(self, **kwargs: Any):
        result = super().optimize(**kwargs)
        selected_source = str(result.selected_source)
        if selected_source.startswith("OPERATIONAL_V27::"):
            selected_source = "OPERATIONAL_V30::" + selected_source.split("::", 1)[1]
        elif "OPERATIONAL_V27" in selected_source:
            selected_source = selected_source.replace("OPERATIONAL_V27", "OPERATIONAL_V30")

        diagnostics = (
            "V30_OBJECTIVE_DRIVEN|"
            f"parent_selection_contract={V27_SELECTION_CONTRACT}|"
            "baseline_dominance_gate=false|dual_estimator_gate=false|"
            "auto_rbc_special_authority=false|"
            + str(result.scipy_message)
        )
        return replace(
            result,
            selected_source=selected_source,
            calibrated_admission_contract=V30_SELECTION_CONTRACT,
            policy_return_admission_contract=V30_SELECTION_CONTRACT,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            scipy_message=diagnostics,
            tfv_value_gate_reason=diagnostics,
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
    # V27 was explicitly a Development contract. V30 is the publication-candidate implementation;
    # inherited promotion booleans must not make otherwise valid runs unusable downstream.
    for obsolete_flag in (
        "development_only",
        "formal_evidence",
        "requires_new_policy_lock",
        "ready_for_policy_lock",
    ):
        updated.pop(obsolete_flag, None)
    updated.update(
        {
            "contract": OPERATIONAL_V30_RUNTIME_CONTRACT,
            "publication_candidate": True,
            "v30_selection_contract": V30_SELECTION_CONTRACT,
            "v30_parent_selection_contract": V27_SELECTION_CONTRACT,
            "v30_selection_semantics": "argmin_same_frozen_exact_return_value_plus_hold_zero",
            "v30_baseline_dominance_gate": False,
            "v30_dual_estimator_gate": False,
            "v30_auto_rbc_special_authority": False,
            "v30_auto_rbc_role": "ordinary_candidate_and_separate_standalone_comparator",
            "historical_completed_outcomes_may_inform_development": True,
            "historical_outcomes_are_online_inputs": False,
            "training_evaluation_split_must_remain_disjoint": True,
            "scientific_metrics_block_runtime": False,
            "additional_v30_sha_admission_gate": False,
        }
    )
    return controller, graph, sensors, updated


__all__ = [
    "DirectTFVOperationalV30MPC",
    "OPERATIONAL_V30_RUNTIME_CONTRACT",
    "V30_SELECTION_CONTRACT",
    "build_operational_v30_controller",
]
