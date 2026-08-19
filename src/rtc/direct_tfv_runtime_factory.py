"""Factories for offline policy-return replay using a frozen V12 continuation policy.

The policy-return estimand only requires CANDIDATE and HOLD branches to use the exact same frozen
continuation policy.  A historical V12 first-move admission fingerprint is useful provenance, but a
mismatch does not invalidate a Development paired counterfactual when the *actual loaded parent*
is frozen identically for both branches.  Production/current-policy construction remains strict by
default; offline policy-iteration experiments must opt in explicitly to provenance-only parent use.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController
from .direct_tfv_first_move_admission import DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT
from .direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from .direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from .direct_tfv_v12_lineage import direct_tfv_v12_behavioral_sha256
from .forecast import PersistenceDecayForecast
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
    DirectTFVScenarioMeanMPCV10,
)


V12_FROZEN_CONTINUATION_FACTORY_CONTRACT = (
    "PROJECT7_V12_FROZEN_CONTINUATION_FACTORY_V3_EXPLICIT_PARENT_PROVENANCE_MODE"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_frozen_v12_continuation_controller(
    *,
    graph_path: str | Path,
    sensors_path: str | Path,
    config_path: str | Path,
    step1_path: str | Path,
    step2_path: str | Path,
    policy_admission_path: str | Path,
    first_move_admission_path: str | Path,
    sequence_support_path: str | Path,
    device: torch.device,
    lbfgsb_maxiter: int = 30,
    optimizer_deadline_seconds: float = 120.0,
    decision_runtime_budget_seconds: float = 180.0,
    first_move_maxiter: int = 12,
    first_move_deadline_seconds: float = 30.0,
    require_behavioral_match: bool = True,
) -> tuple[object, object, tuple[str, ...], dict]:
    graph = _load_graph(graph_path)
    sensors = _load_lines(sensors_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        step2_path, graph=graph, device=device
    )
    policy = json.loads(Path(policy_admission_path).read_text(encoding="utf-8"))
    if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("V12 continuation requires current policy admission")
    first = json.loads(Path(first_move_admission_path).read_text(encoding="utf-8"))
    if str(first.get("contract", "")) != DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT:
        raise ValueError("V12 continuation requires current first-move admission")
    if str(first.get("query_step3_contract", "")) != DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT:
        raise ValueError("continuation first-move admission is not V12 scenario-matched")
    if str(first.get("rainfall_scenario_contract", "")) != DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT:
        raise ValueError("continuation rainfall scenario contract differs from V12")
    first_lineage = first.get("lineage") if isinstance(first.get("lineage"), dict) else {}
    calibrated_behavior = str(
        first.get(
            "v12_behavioral_source_sha256",
            first_lineage.get("v12_behavioral_source_sha256", ""),
        )
    ).lower()
    current_behavior = direct_tfv_v12_behavioral_sha256().lower()
    behavioral_match = bool(calibrated_behavior == current_behavior)
    if require_behavioral_match and not behavioral_match:
        raise ValueError("V12 continuation admission behavioral fingerprint differs from source")

    support = json.loads(Path(sequence_support_path).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=_sha(step2_path),
    )
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_V12_FROZEN_CONTINUATION_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=int(lbfgsb_maxiter),
        deadline_seconds=float(optimizer_deadline_seconds),
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVScenarioMeanMPCV10(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        policy_admission_calibration=policy,
        first_move_admission_calibration=first,
        sequence_support=support,
        design=design,
        first_move_maxiter=int(first_move_maxiter),
        first_move_deadline_seconds=float(first_move_deadline_seconds),
        minimum_rainfall_scenarios=3,
    )
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
    lineage = {
        "factory_contract": V12_FROZEN_CONTINUATION_FACTORY_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "step2_sha256": _sha(step2_path),
        "policy_admission_sha256": _sha(policy_admission_path),
        "first_move_admission_sha256": _sha(first_move_admission_path),
        "sequence_support_sha256": _sha(sequence_support_path),
        "graph_sha256": _sha(graph_path),
        "sensors_sha256": _sha(sensors_path),
        "config_sha256": _sha(config_path),
        "step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "v12_behavioral_source_sha256": current_behavior,
        "calibrated_v12_behavioral_source_sha256": calibrated_behavior,
        "v12_behavioral_match": behavioral_match,
        "behavioral_match_required": bool(require_behavioral_match),
        "parent_use_scope": (
            "STRICT_CURRENT_V12"
            if require_behavioral_match
            else "DEVELOPMENT_POLICY_RETURN_PARENT_PROVENANCE_ONLY"
        ),
        "paired_counterfactual_validity_basis": (
            "exact same loaded frozen continuation policy after H10 in candidate and HOLD branches"
        ),
        "memory_safe_runtime": True,
    }
    return controller, graph, sensors, lineage


__all__ = [
    "V12_FROZEN_CONTINUATION_FACTORY_CONTRACT",
    "build_frozen_v12_continuation_controller",
]
