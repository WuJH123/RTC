"""Build the query-conditioned Project7 three-family H10 policy-return controller."""
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
)
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
)
from .direct_tfv_policy_return_query_margin import (
    DIRECT_TFV_QUERY_MARGIN_CONTRACT,
    load_query_margin_checkpoint,
)
from .direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from .forecast import PersistenceDecayForecast
from .native_supervisory_control import load_native_supervisory_control
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v14 import (
    DIRECT_TFV_QUERY_MARGIN_STEP3_CONTRACT,
    DirectTFVQueryMarginMPCV14,
)


QUERY_MARGIN_RUNTIME_FACTORY_CONTRACT = (
    "PROJECT7_QUERY_CONDITIONED_POLICY_RETURN_FACTORY_V1_82CONTROL_109REP"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_frozen_query_margin_controller(
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
        step2_path,
        graph=graph,
        device=device,
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
    rank_model, rank_norm, adapter, checkpoint = load_query_margin_checkpoint(
        policy_return_checkpoint_path,
        graph=graph,
        base_step2_path=step2_path,
        device=device,
    )
    if checkpoint.get("fresh_validation_verified") is not True:
        raise ValueError(
            "query-margin runtime requires a critic accepted on fresh validation"
        )

    admission = json.loads(
        Path(policy_return_admission_path).read_text(encoding="utf-8")
    )
    if (
        str(admission.get("contract", ""))
        != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT
    ):
        raise ValueError("query-margin runtime requires current policy-return admission")
    if (
        str(admission.get("action_encoding_contract", ""))
        != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
    ):
        raise ValueError("query-margin admission uses another H10 encoding")
    if (
        str(admission.get("candidate_portfolio_contract", ""))
        != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
    ):
        raise ValueError("query-margin admission uses another candidate portfolio")

    checkpoint_sha = _sha(policy_return_checkpoint_path)
    if (
        str(admission.get("policy_return_checkpoint_sha256", "")).lower()
        != checkpoint_sha.lower()
    ):
        raise ValueError("query-margin admission/checkpoint mismatch")
    if (
        str(checkpoint.get("supervisory_mask_sha256", "")).lower()
        != str(control["supervisory_mask_sha256"]).lower()
    ):
        raise ValueError("query-margin checkpoint uses another supervisory mask")
    if (
        str(admission.get("supervisory_mask_sha256", "")).lower()
        != str(control["supervisory_mask_sha256"]).lower()
    ):
        raise ValueError("query-margin admission uses another supervisory mask")

    parent = str(checkpoint.get("continuation_policy_sha256", "")).lower()
    if (
        len(parent) != 64
        or parent != str(admission.get("continuation_policy_sha256", "")).lower()
    ):
        raise ValueError("query-margin critic/admission continuation lineage mismatch")
    if admission.get("projected_gradient_online") not in (False, None):
        raise ValueError("query-margin admission unexpectedly enables projected gradient")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_QUERY_CONDITIONED_POLICY_RETURN_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVQueryMarginMPCV14(
        model=base_model,
        graph=graph,
        normalization=base_norm,
        action_support=base["action_support"],
        sequence_support=support,
        supervisory_mask=mask,
        policy_return_model=rank_model,
        policy_return_normalization=rank_norm,
        query_margin_adapter=adapter,
        policy_return_admission=admission,
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
    step3_path = Path(__file__).resolve().parent / "step3_tfv_value_mpc_v14.py"
    lineage = {
        "factory_contract": QUERY_MARGIN_RUNTIME_FACTORY_CONTRACT,
        "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "base_step2_sha256": _sha(step2_path),
        "supervisory_control_sha256": _sha(supervisory_control_path),
        "supervisory_control_contract": control["contract"],
        "supervisory_control_dimension": int(mask.sum()),
        "model_action_channel_count": 109,
        "sequence_support_sha256": _sha(sequence_support_path),
        "policy_return_step3_source_sha256": _sha(step3_path),
        "policy_return_step3_contract": DIRECT_TFV_QUERY_MARGIN_STEP3_CONTRACT,
        "policy_return_checkpoint_sha256": checkpoint_sha,
        "policy_return_admission_sha256": _sha(policy_return_admission_path),
        "policy_return_action_encoding": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "critic_parent_continuation_policy_sha256": parent,
        "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "candidate_portfolio_family_count_max": 3,
        "candidate_portfolio_families": [
            "STEP2_H10_PROBE_SCALE_0.50",
            "STEP2_H10_PROBE_SCALE_1.00",
            "TYPE_AWARE_HYDRAULIC_PRESSURE",
        ],
        "candidate_selection_uses_relative_rank_only": True,
        "hold_decision_uses_query_best_margin_only": True,
        "conformal_uncertainty_reranks_candidates": False,
        "fresh_validation_verified": True,
        "projected_gradient_h10_enabled": False,
        "projected_gradient_ablation_available": True,
        "projected_gradient_generator_contract": DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
        "projected_gradient_cli_knobs_affect_current_policy": False,
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
    "QUERY_MARGIN_RUNTIME_FACTORY_CONTRACT",
    "build_frozen_query_margin_controller",
]
