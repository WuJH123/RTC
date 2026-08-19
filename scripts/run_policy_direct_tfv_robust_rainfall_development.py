"""Run the Development-only causal rainfall scenario-mean Direct-TFV policy in SWMM."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from rtc.checkpoint_direct_tfv import direct_tfv_first_move_source_sha256, load_direct_tfv_runtime_checkpoint
from rtc.closed_loop import run_authoritative_closed_loop
from rtc.controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController
from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS
from rtc.direct_tfv_first_move_admission import DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from rtc.direct_tfv_v12_lineage import direct_tfv_v12_behavioral_sha256
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
    DirectTFVScenarioMeanMPCV10,
)


DIRECT_TFV_V12_RUNTIME_CONTRACT = "PROJECT7_DIRECT_TFV_V12_SCENARIO_MEAN_AUTHORITATIVE_DEVELOPMENT_RTC_V2_MEMORY_SAFE"
DIRECT_TFV_V12_FORECAST_CONTRACT = (
    "PersistenceDecayForecast(history_steps_for_level=3,decay_per_step=0.92,scenario_multipliers=(0.8,1.0,1.2))"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: str | Path, *, contract: str, label: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != contract:
        raise ValueError(f"V12 runtime requires current {label}")
    if payload.get("development_only") is not True:
        raise ValueError(f"{label} must be Development-only")
    return payload


def _require_step1_lineage(payload: dict, *, step1: torch.nn.Module, sensors: tuple[str, ...]) -> None:
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Direct-TFV checkpoint lacks lineage")
    expected = {
        "step1_model_semantic_sha256": semantic_model_state_dict_sha256(step1),
        "sensor_layout_semantic_sha256": semantic_sensor_layout_sha256(sensors),
    }
    for key, actual in expected.items():
        if str(lineage.get(key, "")).lower() != actual.lower():
            raise ValueError(f"runtime Step1/sensor semantics differ from Direct-TFV training: {key}")


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
    p.add_argument("--first-move-admission", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--optimizer-deadline-seconds", type=float, default=120.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--first-move-maxiter", type=int, default=12)
    p.add_argument("--first-move-deadline-seconds", type=float, default=30.0)
    p.add_argument("--active-support-quantile", choices=("q95",), default="q95")
    args = p.parse_args()
    if not 0.0 < args.optimizer_deadline_seconds < args.decision_runtime_budget_seconds < 600.0:
        raise ValueError("V12 optimizer/controller budgets must fit inside one 600-s update")
    if args.optimizer_deadline_seconds + args.first_move_deadline_seconds >= args.decision_runtime_budget_seconds:
        raise ValueError("V12 full-plan plus first-move budgets exceed controller budget")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    step1 = load_frozen_step1_v127(args.step1, device)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.step2, graph=graph, device=device)
    _require_step1_lineage(checkpoint, step1=step1, sensors=sensors)
    policy = _load_json(args.policy_admission, contract=DIRECT_TFV_POLICY_ADMISSION_CONTRACT, label="V2 policy admission")
    first = _load_json(
        args.first_move_admission,
        contract=DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
        label="V12 first-move admission",
    )
    if str(first.get("execution_estimand", "")) != DIRECT_TFV_FIRST_MOVE_SEMANTICS:
        raise ValueError("V12 admission has the wrong target-latch estimand")
    if str(first.get("query_step3_contract", "")) != DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT:
        raise ValueError("V12 cannot reuse V11 single-scenario first-move admission")
    if str(first.get("rainfall_scenario_contract", "")) != DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT:
        raise ValueError("V12 admission has the wrong causal rainfall scenario contract")
    if int(first.get("calibration_rainfall_group_count", 0)) < int(first.get("minimum_calibration_rainfall_groups", 24)):
        raise ValueError("V12 first-move admission has insufficient rainfall groups")
    lineage = first.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V12 admission lacks lineage")
    if str(lineage.get("step2_checkpoint_sha256", "")).lower() != _sha(args.step2).lower():
        raise ValueError("V12 admission was calibrated on a different Step2 checkpoint")
    sequence_support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        sequence_support, actuator_ids=graph.actuator_ids, step2_checkpoint_sha256=_sha(args.step2)
    )
    if str(lineage.get("sequence_support_sha256", "")).lower() != _sha(args.sequence_support).lower():
        raise ValueError("V12 admission was calibrated with different sequence support")
    current_v12_behavior = direct_tfv_v12_behavioral_sha256()
    calibrated_v12_behavior = str(
        first.get("v12_behavioral_source_sha256", lineage.get("v12_behavioral_source_sha256", ""))
    ).lower()
    if calibrated_v12_behavior != current_v12_behavior.lower():
        raise ValueError("V12 admission behavioral fingerprint differs from runtime")
    current_full = direct_tfv_first_move_source_sha256()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    event_clock = inspect_prepared_event_clock(args.inp)
    if abs(float(event_clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1e-6:
        raise ValueError("V12 event does not satisfy the common prepared-event clock")
    controller_cfg = replace(
        _controller_config(dict(cfg["controller"]), control_block_steps=2),
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_DIRECT_TFV_V12_RUNTIME_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=int(args.lbfgsb_maxiter),
        deadline_seconds=float(args.optimizer_deadline_seconds),
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
        sequence_support=sequence_support,
        design=design,
        first_move_maxiter=int(args.first_move_maxiter),
        first_move_deadline_seconds=float(args.first_move_deadline_seconds),
        minimum_rainfall_scenarios=3,
    )
    controller = MemorySafeDirectTFVAuthoritativeController(
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
        controller,
        max_delta_per_update=0.5,
        allow_projection=False,
        enforce_current_delta=False,
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
        raise RuntimeError("V12 authoritative run failed target write/readback audit")
    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": "proposed_direct_tfv_v12_causal_scenario_mean",
            "direct_tfv_development_runtime_contract": DIRECT_TFV_V12_RUNTIME_CONTRACT,
            "direct_tfv_step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
            "development_only": True,
            "tfv_primary": True,
            "pfv_role": "report_only_secondary_not_optimization_objective",
            "global_peak_role": "report_only",
            "rainfall_scenario_contract": DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
            "rainfall_forecast_contract": DIRECT_TFV_V12_FORECAST_CONTRACT,
            "rainfall_scenario_multipliers": [0.8, 1.0, 1.2],
            "rainfall_history_steps_for_level": 3,
            "future_realized_rainfall_used_as_model_input": False,
            "all_109_facilities_screened_each_decision": True,
            "score_equals_execute_required": True,
            "target_latch_semantics": "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED",
            "active_support_quantile_effective": mpc.active_support_quantile_effective(),
            "first_move_admission_sha256": _sha(args.first_move_admission),
            "step2_model_sha256": _sha(args.step2),
            "joint_sequence_support_sha256": _sha(args.sequence_support),
            "runtime_v12_behavioral_source_sha256": current_v12_behavior,
            "runtime_first_move_full_source_sha256": current_full,
            "runtime_telemetry_graph_release": True,
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "prepared_event_clock": event_clock,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "strategy": metadata["strategy"],
        "step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "metadata_path": result.metadata_path,
        "decision_path": result.decision_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "target_write_readback_passed": True,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2))


if __name__ == "__main__":
    main()
