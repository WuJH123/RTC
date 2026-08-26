"""Development-only V25R2 runtime aligned to the exact receding policy-return objective.

V25R2 preserves the V23 candidate generator, frozen V15 rank and V25 conservative value gate.  The
only scientific correction relative to V25 V1 is the supervised value estimand: the calibrator is
trained on exact system-wide TFV policy return over the stored candidate/HOLD branches, while H120
remains diagnostic.  Hydraulic stress never directly authorizes ACTION.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .direct_tfv_operational_v23_runtime import build_operational_v23_controller
from .direct_tfv_operational_v25_runtime import DirectTFVOperationalV25MPC
from .direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    V23_PORTFOLIO_CONTRACT,
)
from .direct_tfv_v25_value_gate import V25_VALUE_GATE_CONTRACT
from .direct_tfv_v25r2_value_calibration import (
    V25R2_ESTIMAND,
    V25R2_VALUE_FEATURE_CONTRACT,
    load_v25r2_value_calibrator,
)

OPERATIONAL_V25R2_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V25R2_EXACT_POLICY_RETURN_GATE_V1"
)
V25R2_VALUE_GATE_CONTRACT = (
    "PROJECT7_V25R2_EXACT_POLICY_RETURN_ONE_SIDED_UCB_ADMISSION_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DirectTFVOperationalV25R2MPC(DirectTFVOperationalV25MPC):
    """V25 gate mechanics with the corrected exact policy-return semantics."""

    policy_mode = "operational_development_v25r2_exact_policy_return_gate"
    policy_mode_contract = OPERATIONAL_V25R2_RUNTIME_CONTRACT

    def optimize(self, **kwargs: Any):
        result = super().optimize(**kwargs)
        message = str(result.scipy_message).replace("OPERATIONAL_V25", "OPERATIONAL_V25R2")
        message += "|v25r2_exact_policy_return_target=true|h120_target_used_for_admission=false"
        return replace(
            result,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            admission_margin_kind="v25r2_calibrated_one_sided_exact_policy_return_tfv_upper_bound",
            calibrated_admission_contract=V25R2_VALUE_GATE_CONTRACT,
            policy_return_admission_contract=V25R2_VALUE_GATE_CONTRACT,
            policy_return_estimand=V25R2_ESTIMAND,
            scipy_message=message,
        )


def build_operational_v25r2_controller(
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
    v25r2_value_checkpoint_path: str | Path,
    matched_truth_bank_manifest_path: str | Path,
    asset_manifest_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
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
        "feature_contract": V25R2_VALUE_FEATURE_CONTRACT,
    }
    calibrator, checkpoint = load_v25r2_value_calibrator(
        str(v25r2_value_checkpoint_path),
        device=device,
        expected_lineage=expected_lineage,
    )
    metrics = checkpoint.get("oof_metrics", {})
    if metrics.get("train_oof_supported") is not True:
        raise ValueError("V25R2 exact-return calibrator does not pass the Development OOF gate")
    v25r2_mpc = DirectTFVOperationalV25R2MPC(
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
    controller.controller._direct_mpc_adapter.inner = v25r2_mpc
    lineage = dict(v23_lineage)
    lineage.update(
        {
            "contract": OPERATIONAL_V25R2_RUNTIME_CONTRACT,
            "development_only": True,
            "formal_evidence": False,
            "v24_hydraulic_stress_escape_enabled": False,
            "v25r2_value_gate_contract": V25R2_VALUE_GATE_CONTRACT,
            "v25r2_value_calibrator_checkpoint_sha256": _sha(v25r2_value_checkpoint_path),
            "v25_matched_truth_bank_manifest_sha256": _sha(matched_truth_bank_manifest_path),
            "tfv_value_estimand": V25R2_ESTIMAND,
            "h120_value_is_diagnostic_only": True,
            "exact_policy_return_target_used_for_admission": True,
            "hydraulic_stress_is_diagnostic_only": True,
            "hydraulic_candidate_uses_same_portfolio_rank_and_value_gate": True,
            "unknown_value_fail_closed_to_hold": True,
            "v23_learned_path_preserved": True,
            "conformal_admission_used": False,
            "online_swmm_candidate_search": False,
            "online_lbfgsb_used": False,
            "projected_gradient_h10_enabled": False,
            "legacy_v25_gate_contract": V25_VALUE_GATE_CONTRACT,
            "ready_for_policy_lock": False,
        }
    )
    return controller, graph, sensors, lineage


__all__ = [
    "DirectTFVOperationalV25R2MPC",
    "OPERATIONAL_V25R2_RUNTIME_CONTRACT",
    "V25R2_VALUE_GATE_CONTRACT",
    "build_operational_v25r2_controller",
]
