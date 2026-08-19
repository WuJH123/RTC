"""Build the frozen Practical base-H10-probe parent pi0.

This factory deliberately has no historical policy-admission, first-move-admission or L-BFGS-B
arguments.  It is the default continuation for the first exact policy-return label round and keeps
that round on the same candidate/support geometry as the final Practical controller.
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
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_base_probe_parent import DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT, DirectTFVBaseProbeParentMPC
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4


DIRECT_TFV_BASE_PROBE_PARENT_FACTORY_CONTRACT = "PROJECT7_PRACTICAL_BASE_H10_PROBE_PARENT_FACTORY_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_frozen_base_probe_parent_controller(
    *,
    graph_path: str | Path,
    sensors_path: str | Path,
    config_path: str | Path,
    step1_path: str | Path,
    step2_path: str | Path,
    sequence_support_path: str | Path,
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
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVBaseProbeParentMPC(
        model=base_model,
        graph=graph,
        normalization=base_norm,
        action_support=base["action_support"],
        sequence_support=support,
        design=design,
        proposal_probe_chunk_size=int(proposal_probe_chunk_size),
    )
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_BASE_H10_PROBE_PARENT_FALLBACK",
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
    source_path = Path(__file__).resolve().parent / "step3_tfv_base_probe_parent.py"
    lineage = {
        "factory_contract": DIRECT_TFV_BASE_PROBE_PARENT_FACTORY_CONTRACT,
        "policy_contract": DIRECT_TFV_BASE_PROBE_PARENT_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "base_step2_sha256": _sha(step2_path),
        "sequence_support_sha256": _sha(sequence_support_path),
        "policy_source_sha256": _sha(source_path),
        "graph_sha256": _sha(graph_path),
        "sensors_sha256": _sha(sensors_path),
        "config_sha256": _sha(config_path),
        "online_lbfgsb_used": False,
        "legacy_policy_admission_required": False,
        "legacy_first_move_admission_required": False,
        "future_realized_rainfall_used_online": False,
        "memory_safe_runtime": True,
    }
    return controller, graph, sensors, lineage


__all__ = [
    "DIRECT_TFV_BASE_PROBE_PARENT_FACTORY_CONTRACT",
    "build_frozen_base_probe_parent_controller",
]
