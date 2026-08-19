"""Build the current Practical H10 policy-return controller.

The current online policy keeps a frozen 109-channel Step2 representation but changes only the native
supervisory-control subspace. For the Wuhan testbed that means 82 online control freedoms embedded in
109 action channels. The four-family H10 portfolio and its admission must be trained/calibrated under
that same control mask. Historical L-BFGS-B and V12 admissions remain archival only.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .controller_direct_tfv_portfolio import PortfolioMemorySafeDirectTFVAuthoritativeController
from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    load_policy_return_checkpoint,
)
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
)
from .direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from .forecast import PersistenceDecayForecast
from .native_supervisory_control import load_native_supervisory_control
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v13 import (
    DIRECT_TFV_HYBRID_POLICY_RETURN_STEP3_CONTRACT,
    DirectTFVHybridPolicyReturnMPCV13,
)


POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT = (
    "PROJECT7_PRACTICAL_POLICY_RETURN_FACTORY_V7_H10_HYBRID_82CONTROL_109REP"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_frozen_policy_return_continuation_controller(
    *,
    graph_path: str | Path,
    sensors_path: str | Path,
    config_path: str | Path,
    step1_path: str | Path,
    step2_path: str | Path,
    supervisory_control_path: str | Path,
    sequence_support_path: str | Path,
    policy_return_checkpoint_path: str | Path,
    policy_return_admission_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
    projected_gradient_steps: int = 6,
    projected_gradient_step_fraction: float = 0.25,
) -> tuple[object, object, tuple[str, ...], dict]:
    graph = _load_graph(graph_path)
    sensors = _load_lines(sensors_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    base_model, base_norm, base = load_direct_tfv_runtime_checkpoint(
        step2_path, graph=graph, device=device
    )
    control, mask = load_native_supervisory_control(
        supervisory_control_path,
        actuator_ids=graph.actuator_ids,
    )
    support = json.loads(Path(sequence_support_path).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=_sha(step2_path),
        supervisory_mask=mask,
        supervisory_control_contract=str(control["contract"]),
    )
    return_model, return_norm, return_checkpoint = load_policy_return_checkpoint(
        policy_return_checkpoint_path,
        graph=graph,
        device=device,
        expected_base_step2_sha256=_sha(step2_path),
    )
    return_admission = json.loads(Path(policy_return_admission_path).read_text(encoding="utf-8"))
    if str(return_admission.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
        raise ValueError("Practical policy-return runtime requires current H10 admission")
    if str(return_admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("Practical policy-return admission uses another action encoding")
    checkpoint_sha = _sha(policy_return_checkpoint_path)
    if str(return_admission.get("policy_return_checkpoint_sha256", "")).lower() != checkpoint_sha.lower():
        raise ValueError("Practical policy-return admission/critic mismatch")
    checkpoint_parent = str(return_checkpoint.get("continuation_policy_sha256", "")).lower()
    admission_parent = str(return_admission.get("continuation_policy_sha256", "")).lower()
    if len(checkpoint_parent) != 64 or checkpoint_parent != admission_parent:
        raise ValueError("Practical policy-return critic/admission parent-policy mismatch")
    checkpoint_portfolio = str(return_checkpoint.get("candidate_portfolio_contract", ""))
    admission_portfolio = str(return_admission.get("candidate_portfolio_contract", ""))
    if checkpoint_portfolio != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT or admission_portfolio != checkpoint_portfolio:
        raise ValueError("Practical critic/admission must use the current masked hybrid H10 portfolio")
    if str(return_checkpoint.get("supervisory_mask_sha256", "")).lower() != str(control["supervisory_mask_sha256"]).lower():
        raise ValueError("policy-return critic was trained under another supervisory-control mask")
    if str(return_admission.get("supervisory_mask_sha256", "")).lower() != str(control["supervisory_mask_sha256"]).lower():
        raise ValueError("policy-return admission was calibrated under another supervisory-control mask")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_PRACTICAL_HYBRID_POLICY_RETURN_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVHybridPolicyReturnMPCV13(
        model=base_model,
        graph=graph,
        normalization=base_norm,
        action_support=base["action_support"],
        sequence_support=support,
        supervisory_mask=mask,
        policy_return_model=return_model,
        policy_return_normalization=return_norm,
        policy_return_admission=return_admission,
        policy_return_checkpoint_sha256=checkpoint_sha,
        design=design,
        proposal_probe_chunk_size=int(proposal_probe_chunk_size),
        projected_gradient_steps=int(projected_gradient_steps),
        projected_gradient_step_fraction=float(projected_gradient_step_fraction),
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
    step3_path = Path(__file__).resolve().parent / "step3_tfv_value_mpc_v13.py"
    lineage = {
        "factory_contract": POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "base_step2_sha256": _sha(step2_path),
        "supervisory_control_sha256": _sha(supervisory_control_path),
        "supervisory_control_contract": control["contract"],
        "supervisory_control_dimension": int(mask.sum()),
        "model_action_channel_count": 109,
        "sequence_support_sha256": _sha(sequence_support_path),
        "policy_return_step3_source_sha256": _sha(step3_path),
        "policy_return_step3_contract": DIRECT_TFV_HYBRID_POLICY_RETURN_STEP3_CONTRACT,
        "policy_return_checkpoint_sha256": checkpoint_sha,
        "policy_return_admission_sha256": _sha(policy_return_admission_path),
        "policy_return_action_encoding": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "critic_parent_continuation_policy_sha256": checkpoint_parent,
        "candidate_portfolio_contract": checkpoint_portfolio,
        "candidate_portfolio_family_count_max": 4,
        "projected_gradient_h10_enabled": True,
        "projected_gradient_generator_contract": DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
        "projected_gradient_free_dimension": int(mask.sum()),
        "projected_gradient_tensor_channels": 109,
        "projected_gradient_steps": int(projected_gradient_steps),
        "projected_gradient_step_fraction": float(projected_gradient_step_fraction),
        "portfolio_mode": True,
        "online_lbfgsb_used": False,
        "h10_probe_generator_contract": DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
        "legacy_v12_admission_required_online": False,
        "graph_sha256": _sha(graph_path),
        "sensors_sha256": _sha(sensors_path),
        "config_sha256": _sha(config_path),
        "step1_retrained_for_control_mask": False,
        "base_step2_retrained_for_control_mask": False,
        "memory_safe_runtime": True,
    }
    return controller, graph, sensors, lineage


__all__ = [
    "POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT",
    "build_frozen_policy_return_continuation_controller",
]
