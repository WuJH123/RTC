"""Project7 V27R1 Development-only physical-support ablation.

The numerical V27 value model, candidate portfolio, causal inputs and all physical first-move
constraints are unchanged.  The only ablation is the learned q95 *joint-sequence* contraction: the
candidate produced after mask/bounds/first-radius/K/slew projection is executed without the extra
q95 L1/TV shrink toward HOLD.  The raw action is still scored by the same frozen V27 value model.

This lane is diagnostic only and must not replace the primary V27 q95-supported policy without new
Development evidence.  It exists because the completed V27 Benchmark5 showed q95 changed candidate
selection on 43.65% of decisions and ACTION/HOLD sign on 16.61%; an authoritative SWMM ablation is
needed to determine whether that support layer is helpful or harmful rather than inferring causality
from latent scores alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .direct_tfv_operational_v27_runtime import (
    DirectTFVOperationalV27MPC,
    OPERATIONAL_V27_RUNTIME_CONTRACT,
    build_operational_v27_controller,
)
from .direct_tfv_policy_return import encode_policy_return_action_token


OPERATIONAL_V27R1_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V27R1_PHYSICAL_ONLY_Q95_SEQUENCE_ABLATION_V1"
)
V27R1_SUPPORT_CONTRACT = (
    "MASK_BOUNDS_FIRST_RADIUS_K_CEILING_SLEW_ONLY_Q95_JOINT_SEQUENCE_REPORT_ONLY_V1"
)


class DirectTFVOperationalV27R1PhysicalOnlyMPC(DirectTFVOperationalV27MPC):
    """V27 with q95 joint L1/TV contraction disabled for Development ablation only."""

    policy_mode = "operational_development_v27r1_physical_only_q95_ablation"
    policy_mode_contract = OPERATIONAL_V27R1_RUNTIME_CONTRACT

    def _h10_supported_target(
        self, target: torch.Tensor, active_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float | bool | str]]:
        # ``target`` has already passed the common candidate generator's supervisory mask, actuator
        # bounds, first-radius, changed-facility ceiling and per-update setting delta.  Reconstruct the
        # standard H10 action token but intentionally do not apply `_contract_to_joint_sequence_support`.
        _, sequence_batch = encode_policy_return_action_token(
            active_target,
            target,
            horizon_steps=int(self.design.prediction_horizon_steps),
            first_action_steps=int(self.design.control_block_steps),
        )
        sequence = sequence_batch[0].detach()
        raw_target = sequence[0].detach()
        changed = int(torch.count_nonzero(torch.abs(raw_target - active_target) > 1.0e-7).item())
        q95 = dict(self.joint_sequence_support_diagnostics(sequence, active_target))
        # Preserve the q95 counterfactual diagnostic explicitly while making the *executed* support
        # contract unambiguous.  Common execution telemetry must not mislabel this intentional
        # ablation as a q95 engineering violation.
        diagnostics: dict[str, float | bool | str] = {
            "quantile": "PHYSICAL_ONLY_Q95_ABLATION",
            "first_block_l1": float(q95.get("first_block_l1", 0.0)),
            "h120_l1": float(q95.get("h120_l1", 0.0)),
            "h120_total_variation_l1": float(q95.get("h120_total_variation_l1", 0.0)),
            "max_ratio": 0.0,
            "binding": False,
            "q95_counterfactual_quantile": str(q95.get("quantile", "q95")),
            "q95_counterfactual_max_ratio": float(q95.get("max_ratio", 0.0)),
            "q95_counterfactual_binding": bool(q95.get("binding", False)),
        }
        return raw_target, sequence, changed, diagnostics


def build_operational_v27r1_controller(
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
    """Build the ablation by reusing the already-tested V27 construction surface."""

    controller, graph, sensors, parent_lineage = build_operational_v27_controller(
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
        dataset_manifest_path=dataset_manifest_path,
        asset_manifest_path=asset_manifest_path,
        device=device,
        decision_runtime_budget_seconds=decision_runtime_budget_seconds,
        proposal_probe_chunk_size=proposal_probe_chunk_size,
    )
    old = controller.controller._direct_mpc_adapter.inner
    ablation = DirectTFVOperationalV27R1PhysicalOnlyMPC(
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
    controller.controller._direct_mpc_adapter.inner = ablation
    lineage = dict(parent_lineage)
    lineage.update(
        {
            "contract": OPERATIONAL_V27R1_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "v27_parent_runtime_contract": OPERATIONAL_V27_RUNTIME_CONTRACT,
            "v27r1_support_contract": V27R1_SUPPORT_CONTRACT,
            "q95_joint_sequence_contraction_executed": False,
            "q95_joint_sequence_support_role": "REPORT_ONLY_COUNTERFACTUAL",
            "supervisory_mask_preserved": True,
            "actuator_bounds_preserved": True,
            "first_radius_preserved": True,
            "changed_facility_ceiling_preserved": True,
            "max_setting_delta_per_update_preserved": True,
            "value_model_changed": False,
            "candidate_generator_changed": False,
            "auto_rbc_shadow_changed": False,
            "ready_for_policy_lock": False,
        }
    )
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVOperationalV27R1PhysicalOnlyMPC",
    "OPERATIONAL_V27R1_RUNTIME_CONTRACT",
    "V27R1_SUPPORT_CONTRACT",
    "build_operational_v27r1_controller",
]
