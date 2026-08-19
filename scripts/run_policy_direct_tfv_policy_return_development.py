"""Run Development-only Direct-TFV policy-return first-action control in authoritative SWMM."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.closed_loop import run_authoritative_closed_loop
from rtc.controller_direct_tfv import DirectTFVAuthoritativeController
from rtc.direct_tfv_first_move_admission import DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    load_policy_return_checkpoint,
)
from rtc.direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import _controller_config, _controls_disabled_runtime, _load_graph, _load_lines
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config
from rtc.runtime_controller_guard import ContinuityGuardController
from rtc.step1_runtime_v127 import load_frozen_step1_v127
from rtc.step2_state_store_v127 import semantic_model_state_dict_sha256, semantic_sensor_layout_sha256
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)
from rtc.step3_tfv_value_mpc_v11 import (
    DIRECT_TFV_POLICY_RETURN_STEP3_CONTRACT,
    DirectTFVPolicyReturnMPCV11,
)


DIRECT_TFV_POLICY_RETURN_RUNTIME_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_AUTHORITATIVE_DEVELOPMENT_RTC_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: str | Path, contract: str, label: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != contract:
        raise ValueError(f"policy-return runtime requires current {label}")
    if payload.get("development_only") is not True:
        raise ValueError(f"{label} must be Development-only")
    return payload


def _require_step1_lineage(payload: dict, *, step1: torch.nn.Module, sensors: tuple[str, ...]) -> None:
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("base Direct-TFV checkpoint lacks lineage")
    expected = {
        "step1_model_semantic_sha256": semantic_model_state_dict_sha256(step1),
        "sensor_layout_semantic_sha256": semantic_sensor_layout_sha256(sensors),
    }
    for key, actual in expected.items():
        if str(lineage.get(key, "")).lower() != actual.lower():
            raise ValueError(f"runtime Step1/sensor semantics differ from base Step2 training: {key}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--policy-admission", required=True)
    p.add_argument("--v12-first-move-admission", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--policy-return-checkpoint", required=True)
    p.add_argument("--policy-return-admission", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--optimizer-deadline-seconds", type=float, default=120.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--first-move-maxiter", type=int, default=12)
    p.add_argument("--first-move-deadline-seconds", type=float, default=30.0)
    args = p.parse_args()
    if not 0.0 < args.optimizer_deadline_seconds < args.decision_runtime_budget_seconds < 600.0:
        raise ValueError("policy-return runtime budgets must fit inside one control update")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph); sensors = _load_lines(args.sensors)
    step1 = load_frozen_step1_v127(args.step1, device)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.step2, graph=graph, device=device)
    _require_step1_lineage(checkpoint, step1=step1, sensors=sensors)
    policy = _load_json(args.policy_admission, DIRECT_TFV_POLICY_ADMISSION_CONTRACT, "V2 policy admission")
    first = _load_json(
        args.v12_first_move_admission,
        DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
        "V12 scenario-mean first-move admission",
    )
    if str(first.get("query_step3_contract", "")) != DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT:
        raise ValueError("policy-return direction generator requires scenario-matched V12 admission lineage")
    if str(first.get("rainfall_scenario_contract", "")) != DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT:
        raise ValueError("V12 direction admission has the wrong rainfall scenario contract")
    sequence_support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        sequence_support, actuator_ids=graph.actuator_ids, step2_checkpoint_sha256=_sha(args.step2)
    )
    return_model, return_norm, return_checkpoint = load_policy_return_checkpoint(
        args.policy_return_checkpoint,
        graph=graph,
        device=device,
        expected_base_step2_sha256=_sha(args.step2),
    )
    return_admission = _load_json(
        args.policy_return_admission,
        DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
        "policy-return admission",
    )
    if str(return_admission.get("policy_return_checkpoint_sha256", "")).lower() != _sha(
        args.policy_return_checkpoint
    ).lower():
        raise ValueError("policy-return admission was calibrated on another critic checkpoint")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    clock = inspect_prepared_event_clock(args.inp)
    if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1e-6:
        raise ValueError("policy-return event violates common warm-up clock")
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_DIRECT_TFV_POLICY_RETURN_RUNTIME_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=int(args.lbfgsb_maxiter),
        deadline_seconds=float(args.optimizer_deadline_seconds),
        active_facility_count=0,
        active_support_quantile="q95",
    )
    mpc = DirectTFVPolicyReturnMPCV11(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        policy_admission_calibration=policy,
        first_move_admission_calibration=first,
        sequence_support=sequence_support,
        design=design,
        first_move_maxiter=int(args.first_move_maxiter),
        first_move_deadline_seconds=float(args.first_move_deadline_seconds),
        minimum_rainfall_scenarios=3,
        policy_return_model=return_model,
        policy_return_normalization=return_norm,
        policy_return_admission=return_admission,
        policy_return_checkpoint_sha256=_sha(args.policy_return_checkpoint),
    )
    controller = DirectTFVAuthoritativeController(
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
        controller, max_delta_per_update=0.5, allow_projection=False, enforce_current_delta=False
    )
    runtime_inp = _controls_disabled_runtime(
        source_inp=Path(args.inp),
        cache_dir=Path(args.out_dir) / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=args.out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=int(cfg["control_start_minutes"]),
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=False,
    )
    write_audit = audit_target_write_readback_v127(metadata_path=result.metadata_path)
    if write_audit.get("passed") is not True:
        raise RuntimeError("policy-return runtime failed target write/readback audit")
    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": "proposed_direct_tfv_policy_return",
            "direct_tfv_development_runtime_contract": DIRECT_TFV_POLICY_RETURN_RUNTIME_CONTRACT,
            "direct_tfv_step3_contract": DIRECT_TFV_POLICY_RETURN_STEP3_CONTRACT,
            "development_only": True,
            "tfv_primary": True,
            "pfv_role": "report_only_secondary_not_optimization_objective",
            "global_peak_role": "report_only",
            "future_realized_rainfall_used_as_model_input": False,
            "policy_return_checkpoint_sha256": _sha(args.policy_return_checkpoint),
            "policy_return_admission_sha256": _sha(args.policy_return_admission),
            "policy_return_parent_continuation_sha256": str(
                return_admission["continuation_policy_sha256"]
            ),
            "base_step2_sha256": _sha(args.step2),
            "sequence_support_sha256": _sha(args.sequence_support),
            "v12_open_loop_first_move_margin_controls_execution": False,
            "generic_d3_floor_controls_execution": False,
            "all_109_facilities_screened_each_decision": True,
            "target_latch_semantics": "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED",
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "prepared_event_clock": clock,
            "policy_return_training_validation_metrics": return_checkpoint.get("validation_metrics", {}),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "strategy": metadata["strategy"],
        "step3_contract": metadata["direct_tfv_step3_contract"],
        "metadata_path": result.metadata_path,
        "decision_path": result.decision_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "target_write_readback_passed": True,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2))


if __name__ == "__main__":
    main()
