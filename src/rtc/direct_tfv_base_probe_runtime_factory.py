"""Build the frozen Practical three-family H10 parent pi0.

The pretrained representation remains 109-channel, while the online parent uses the frozen native
supervisory-control mask and matching masked q95 support. For the Wuhan testbed this is 82 online
control freedoms embedded in 109 Step2 channels. Current pi0 uses only Step2 0.5/1.0 H10 probes plus
type-aware hydraulic pressure. Projected gradient is Development ablation only.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController
from .direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from .forecast import PersistenceDecayForecast
from .native_supervisory_control import load_native_supervisory_control
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_base_probe_parent_v2 import (
    DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT,
    DirectTFVBaseHybridParentMPCV2,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4


DIRECT_TFV_BASE_PROBE_PARENT_FACTORY_CONTRACT = (
    "PROJECT7_PRACTICAL_BASE_H10_THREE_FAMILY_PARENT_FACTORY_V4_82CONTROL_109REP"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_frozen_base_probe_parent_controller(
    *,
    graph_path: str | Path,
    sensors_path: str | Path,
    config_path: str | Path,
    step1_path: str | Path,
    step2_path: str | Path,
    supervisory_control_path: str | Path,
    sequence_support_path: str | Path,
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
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVBaseHybridParentMPCV2(
        model=base_model,
        graph=graph,
        normalization=base_norm,
        action_support=base["action_support"],
        sequence_support=support,
        design=design,
        supervisory_mask=mask,
        proposal_probe_chunk_size=int(proposal_probe_chunk_size),
        projected_gradient_steps=int(projected_gradient_steps),
        projected_gradient_step_fraction=float(projected_gradient_step_fraction),
    )
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_BASE_H10_THREE_FAMILY_PARENT_FALLBACK",
    )
    controller_cfg.validate()
    inner = MemorySafeDirectTFVAuthoritativeController(
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
    source_path = Path(__file__).resolve().parent / "step3_tfv_base_probe_parent_v2.py"
    lineage = {
        "factory_contract": DIRECT_TFV_BASE_PROBE_PARENT_FACTORY_CONTRACT,
        "policy_contract": DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "base_step2_sha256": _sha(step2_path),
        "supervisory_control_sha256": _sha(supervisory_control_path),
        "supervisory_control_contract": control["contract"],
        "supervisory_control_dimension": int(mask.sum()),
        "model_action_channel_count": 109,
        "sequence_support_sha256": _sha(sequence_support_path),
        "policy_source_sha256": _sha(source_path),
        "graph_sha256": _sha(graph_path),
        "sensors_sha256": _sha(sensors_path),
        "config_sha256": _sha(config_path),
        "candidate_portfolio_family_count_max": 3,
        "candidate_portfolio_families": [
            "STEP2_H10_PROBE_SCALE_0.50",
            "STEP2_H10_PROBE_SCALE_1.00",
            "TYPE_AWARE_HYDRAULIC_PRESSURE",
        ],
        "projected_gradient_h10_enabled": False,
        "projected_gradient_ablation_available": True,
        "projected_gradient_cli_knobs_affect_current_policy": False,
        "online_lbfgsb_used": False,
        "legacy_policy_admission_required": False,
        "legacy_first_move_admission_required": False,
        "future_realized_rainfall_used_online": False,
        "step1_retrained_for_control_mask": False,
        "base_step2_retrained_for_control_mask": False,
        "memory_safe_runtime": True,
    }
    return controller, graph, sensors, lineage


__all__ = [
    "DIRECT_TFV_BASE_PROBE_PARENT_FACTORY_CONTRACT",
    "build_frozen_base_probe_parent_controller",
]
