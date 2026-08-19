"""Build the current Practical H10 policy-return controller.

This factory is intentionally **current-only**. It has no historical V12 policy/first-move admission
arguments and no L-BFGS-B branch. Historical parent/ablation code remains in archival modules; the
first current paired-label round instead uses ``direct_tfv_base_probe_runtime_factory``.
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
from .direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
)
from .direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from .forecast import PersistenceDecayForecast
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v12 import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT,
    DirectTFVPolicyReturnPortfolioMPCV12,
)


POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT = (
    "PROJECT7_PRACTICAL_POLICY_RETURN_FACTORY_V5_NO_LEGACY_V12"
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
    sequence_support_path: str | Path,
    policy_return_checkpoint_path: str | Path,
    policy_return_admission_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict]:
    graph = _load_graph(graph_path)
    sensors = _load_lines(sensors_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    base_model, base_norm, base = load_direct_tfv_runtime_checkpoint(step2_path, graph=graph, device=device)
    support = json.loads(Path(sequence_support_path).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=_sha(step2_path),
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
        raise ValueError("Practical critic/admission must use the current H10 candidate portfolio")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_PRACTICAL_POLICY_RETURN_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVPolicyReturnPortfolioMPCV12(
        model=base_model,
        graph=graph,
        normalization=base_norm,
        action_support=base["action_support"],
        sequence_support=support,
        policy_return_model=return_model,
        policy_return_normalization=return_norm,
        policy_return_admission=return_admission,
        policy_return_checkpoint_sha256=checkpoint_sha,
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
    step3_path = Path(__file__).resolve().parent / "step3_tfv_value_mpc_v12.py"
    lineage = {
        "factory_contract": POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "base_step2_sha256": _sha(step2_path),
        "sequence_support_sha256": _sha(sequence_support_path),
        "policy_return_step3_source_sha256": _sha(step3_path),
        "policy_return_step3_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT,
        "policy_return_checkpoint_sha256": checkpoint_sha,
        "policy_return_admission_sha256": _sha(policy_return_admission_path),
        "policy_return_action_encoding": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "critic_parent_continuation_policy_sha256": checkpoint_parent,
        "candidate_portfolio_contract": checkpoint_portfolio,
        "portfolio_mode": True,
        "online_lbfgsb_used": False,
        "h10_probe_generator_contract": DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
        "legacy_v12_admission_required_online": False,
        "graph_sha256": _sha(graph_path),
        "sensors_sha256": _sha(sensors_path),
        "config_sha256": _sha(config_path),
        "memory_safe_runtime": True,
    }
    return controller, graph, sensors, lineage


__all__ = [
    "POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT",
    "build_frozen_policy_return_continuation_controller",
]
