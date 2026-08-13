"""Scientific gate order for Project7 Step2 V11.2.

V11.2 does not authorize a new production model by itself. Sparse D2 effects are
first represented as state-conditioned support. The Hydraulic learning horizon
remains 0-120 min, while the original 360-min D2 branch is retained as a
read-only lag/non-local diagnostic so delayed responses are not misclassified
as impossible.
"""
from __future__ import annotations

from dataclasses import dataclass

V112_CONTRACT = "PROJECT7_STEP2_V112_INFLUENCE_ATLAS_FIRST_V1"


@dataclass(frozen=True)
class Step2V112Contract:
    full_d2_source_branches: int = 4800
    source_d2_checkpoint_states: int = 192
    actuator_count: int = 109
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    hydraulic_effect_horizon_minutes: int = 120
    diagnostic_source_horizon_minutes: int = 360
    mpc_horizon_minutes: int = 360
    seed: int = 42
    formal_authorized: bool = False
    validation_outcomes_allowed: bool = False
    final_outcomes_allowed: bool = False
    new_swmm_authorized: bool = False
    sum_d2_magnitudes_for_d3: bool = False
    hard_hop_support_mask: bool = False

    def validate(self) -> None:
        if (self.full_d2_source_branches, self.source_d2_checkpoint_states, self.actuator_count) != (4800, 192, 109):
            raise ValueError("V112 D2 source census drift")
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V112 causal/control clock drift")
        if (self.hydraulic_effect_horizon_minutes, self.diagnostic_source_horizon_minutes, self.mpc_horizon_minutes) != (120, 360, 360):
            raise ValueError("V112 Hydraulic/diagnostic/MPC horizon drift")
        if self.seed != 42:
            raise ValueError("V112 seed drift")
        if any((self.formal_authorized, self.validation_outcomes_allowed,
                self.final_outcomes_allowed, self.new_swmm_authorized,
                self.sum_d2_magnitudes_for_d3, self.hard_hop_support_mask)):
            raise ValueError("V112 scientific safety boundary was weakened")

    @property
    def development_order(self) -> tuple[str, ...]:
        return (
            "reconcile_full_4800_source_vs_current_derived_cache",
            "build_trainfit_state_conditioned_influence_atlas",
            "audit_trainfit_delayed_influence_120_360_diagnostic_only",
            "verify_setting_to_realized_facility_flow_gate",
            "verify_support_lag_sign_and_global_escape",
            "local_endpoint_representation_sanity_control",
            "train_support_conditioned_direct_signed_hydraulic_tiny_micro",
            "canonical_trainfit_d2_gate",
            "internal_holdout_once_after_freeze",
            "direct_authoritative_d3_joint_action_training",
            "forecast_shift_test",
            "runtime_gradient_and_mpc_smoke",
        )


__all__ = ["Step2V112Contract", "V112_CONTRACT"]
