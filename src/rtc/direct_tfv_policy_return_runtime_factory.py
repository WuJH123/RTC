"""Build frozen policy-return continuation controllers for Project7.

Two paths are intentionally separated:

- **Practical portfolio**: current online policy. It needs frozen Step1/base Step2, q95 sequence support,
  H10-aligned policy-return critic/admission and the Practical candidate contract. Historical V12
  optimizer admissions are neither read nor validated.
- **Legacy V11 bridge**: kept only for policy-iteration diagnostics/backward evidence and therefore
  still validates its historical V12 policy/first-move lineage.

This separation prevents stale open-loop calibration hashes from blocking an online policy that does
not use those calibrations or the L-BFGS-B optimizer.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .controller_direct_tfv_portfolio import PortfolioMemorySafeDirectTFVAuthoritativeController
from .controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController
from .direct_tfv_first_move_admission import DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT
from .direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
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
from .direct_tfv_v12_lineage import direct_tfv_v12_behavioral_sha256
from .forecast import PersistenceDecayForecast
from .production_cli import _controller_config, _load_graph, _load_lines
from .runtime_controller_guard import ContinuityGuardController
from .step1_runtime_v127 import load_frozen_step1_v127
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v10 import DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT
from .step3_tfv_value_mpc_v11 import DirectTFVPolicyReturnMPCV11
from .step3_tfv_value_mpc_v12 import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT,
    DirectTFVPolicyReturnPortfolioMPCV12,
)


POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT = (
    "PROJECT7_POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_V4_PRACTICAL_DECOUPLED"
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
    policy_admission_path: str | Path,
    v12_first_move_admission_path: str | Path,
    sequence_support_path: str | Path,
    policy_return_checkpoint_path: str | Path,
    policy_return_admission_path: str | Path,
    device: torch.device,
    lbfgsb_maxiter: int = 30,
    optimizer_deadline_seconds: float = 120.0,
    decision_runtime_budget_seconds: float = 180.0,
    first_move_maxiter: int = 12,
    first_move_deadline_seconds: float = 30.0,
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
        raise ValueError("policy-return continuation requires current H10 return admission")
    if str(return_admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("policy-return continuation admission uses another action encoding")
    checkpoint_sha = _sha(policy_return_checkpoint_path)
    if str(return_admission.get("policy_return_checkpoint_sha256", "")).lower() != checkpoint_sha.lower():
        raise ValueError("policy-return continuation admission/critic mismatch")
    checkpoint_parent = str(return_checkpoint.get("continuation_policy_sha256", "")).lower()
    admission_parent = str(return_admission.get("continuation_policy_sha256", "")).lower()
    if len(checkpoint_parent) != 64 or checkpoint_parent != admission_parent:
        raise ValueError("policy-return continuation critic/admission parent-policy mismatch")
    checkpoint_portfolio = str(return_checkpoint.get("candidate_portfolio_contract", ""))
    admission_portfolio = str(return_admission.get("candidate_portfolio_contract", ""))
    if checkpoint_portfolio != admission_portfolio:
        raise ValueError("policy-return critic/admission use different candidate query families")
    portfolio_mode = checkpoint_portfolio == DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
    if checkpoint_portfolio and not portfolio_mode:
        raise ValueError("policy-return checkpoint contains an unknown candidate portfolio contract")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_POLICY_RETURN_FROZEN_CONTINUATION_FALLBACK",
    )
    controller_cfg.validate()

    if portfolio_mode:
        # The design object carries only frozen time/support settings here. Its SciPy knobs are inert
        # because Practical Step3 overrides optimize() and never invokes the V4 L-BFGS-B path.
        design = DirectTFVMPCDesignV4(
            maxiter=1,
            deadline_seconds=min(30.0, float(optimizer_deadline_seconds)),
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
        )
        controller_cls = PortfolioMemorySafeDirectTFVAuthoritativeController
        legacy_policy_sha = "NOT_REQUIRED_FOR_PRACTICAL_PORTFOLIO"
        legacy_first_sha = "NOT_REQUIRED_FOR_PRACTICAL_PORTFOLIO"
        current_v12_behavior = "NOT_REQUIRED_FOR_PRACTICAL_PORTFOLIO"
        calibrated_v12_behavior = "NOT_REQUIRED_FOR_PRACTICAL_PORTFOLIO"
        v12_behavioral_match = None
        step3_contract = DIRECT_TFV_POLICY_RETURN_PORTFOLIO_STEP3_CONTRACT
    else:
        policy = json.loads(Path(policy_admission_path).read_text(encoding="utf-8"))
        if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
            raise ValueError("legacy policy-return bridge requires current policy admission")
        first = json.loads(Path(v12_first_move_admission_path).read_text(encoding="utf-8"))
        if str(first.get("contract", "")) != DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT:
            raise ValueError("legacy policy-return bridge requires current V12 first-move admission")
        if str(first.get("query_step3_contract", "")) != DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT:
            raise ValueError("legacy bridge requires V12 scenario-mean query lineage")
        first_lineage = first.get("lineage") if isinstance(first.get("lineage"), dict) else {}
        current_v12_behavior = direct_tfv_v12_behavioral_sha256().lower()
        calibrated_v12_behavior = str(
            first.get("v12_behavioral_source_sha256", first_lineage.get("v12_behavioral_source_sha256", ""))
        ).lower()
        v12_behavioral_match = calibrated_v12_behavior == current_v12_behavior
        design = DirectTFVMPCDesignV4(
            maxiter=int(lbfgsb_maxiter),
            deadline_seconds=float(optimizer_deadline_seconds),
            active_facility_count=0,
            active_support_quantile="q95",
        )
        mpc = DirectTFVPolicyReturnMPCV11(
            model=base_model,
            graph=graph,
            normalization=base_norm,
            action_support=base["action_support"],
            policy_admission_calibration=policy,
            first_move_admission_calibration=first,
            sequence_support=support,
            design=design,
            first_move_maxiter=int(first_move_maxiter),
            first_move_deadline_seconds=float(first_move_deadline_seconds),
            minimum_rainfall_scenarios=3,
            policy_return_model=return_model,
            policy_return_normalization=return_norm,
            policy_return_admission=return_admission,
            policy_return_checkpoint_sha256=checkpoint_sha,
        )
        controller_cls = MemorySafeDirectTFVAuthoritativeController
        legacy_policy_sha = _sha(policy_admission_path)
        legacy_first_sha = _sha(v12_first_move_admission_path)
        step3_contract = str(getattr(mpc, "policy_mode_contract", ""))

    inner = controller_cls(
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
    step3_path = Path(__file__).resolve().parent / (
        "step3_tfv_value_mpc_v12.py" if portfolio_mode else "step3_tfv_value_mpc_v11.py"
    )
    lineage = {
        "factory_contract": POLICY_RETURN_FROZEN_CONTINUATION_FACTORY_CONTRACT,
        "step1_sha256": _sha(step1_path),
        "base_step2_sha256": _sha(step2_path),
        "sequence_support_sha256": _sha(sequence_support_path),
        "policy_return_step3_source_sha256": _sha(step3_path),
        "policy_return_step3_contract": step3_contract,
        "policy_return_checkpoint_sha256": checkpoint_sha,
        "policy_return_admission_sha256": _sha(policy_return_admission_path),
        "policy_return_action_encoding": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "critic_parent_continuation_policy_sha256": checkpoint_parent,
        "candidate_portfolio_contract": checkpoint_portfolio,
        "portfolio_mode": portfolio_mode,
        "online_lbfgsb_used": False if portfolio_mode else True,
        "h10_probe_generator_contract": DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT if portfolio_mode else "",
        "legacy_policy_admission_sha256": legacy_policy_sha,
        "legacy_v12_first_move_admission_sha256": legacy_first_sha,
        "v12_behavioral_source_sha256": current_v12_behavior,
        "calibrated_v12_behavioral_source_sha256": calibrated_v12_behavior,
        "v12_behavioral_match": v12_behavioral_match,
        "legacy_v12_admission_required_online": False if portfolio_mode else True,
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
