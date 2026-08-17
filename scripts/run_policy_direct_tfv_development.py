"""Run calibrated Direct-TFV MPC in an authoritative Development SWMM closed loop."""
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
from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import _controller_config, _controls_disabled_runtime, _load_graph, _load_lines
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config
from rtc.runtime_controller_guard import ContinuityGuardController
from rtc.step1_runtime_v127 import load_frozen_step1_v127
from rtc.step2_state_store_v127 import semantic_model_state_dict_sha256, semantic_sensor_layout_sha256
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v5 import DirectTFVRecedingMPCV5


DIRECT_TFV_DEVELOPMENT_RUNTIME_CONTRACT = "PROJECT7_DIRECT_TFV_AUTHORITATIVE_10MIN_DEVELOPMENT_RTC_V4"
DIRECT_TFV_CAUSAL_FORECAST_CONTRACT = "PersistenceDecayForecast(history_steps_for_level=1,decay_per_step=0.92,scenario_multiplier=1.0)"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_step1_lineage(payload: dict, *, step1: torch.nn.Module, sensors: tuple[str, ...]) -> None:
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Direct-TFV checkpoint lacks lineage")
    expected = {
        "step1_model_semantic_sha256": semantic_model_state_dict_sha256(step1),
        "sensor_layout_semantic_sha256": semantic_sensor_layout_sha256(sensors),
    }
    for key, actual in expected.items():
        trained = str(lineage.get(key, "")).lower()
        if not trained or trained != actual.lower():
            raise ValueError(f"runtime Step1/sensor semantics differ from Direct-TFV training: {key}")
    if str(lineage.get("causal_rainfall_forecast_contract", "")) != DIRECT_TFV_CAUSAL_FORECAST_CONTRACT:
        raise ValueError("runtime rainfall forecast differs from Direct-TFV training")


def _load_admission(path: str | Path, *, step2_path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("contract")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
        raise ValueError("runtime admission artifact has the wrong contract")
    if payload.get("development_only") is not True:
        raise ValueError("runtime admission artifact must be Development-only evidence")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict) or str(lineage.get("step2_checkpoint_sha256", "")).lower() != _sha(step2_path).lower():
        raise ValueError("runtime admission calibration was derived from a different Step2 checkpoint")
    split = payload.get("split")
    if not isinstance(split, dict) or int(split.get("rainfall_group_overlap_count", -1)) != 0:
        raise ValueError("runtime admission calibration/audit rainfall groups are not disjoint")
    return payload


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
    p.add_argument("--admission-calibration", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--optimizer-deadline-seconds", type=float, default=120.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--active-facilities", type=int, default=0)
    p.add_argument("--active-support-quantile", choices=("q90", "q95", "q99"), default="q95")
    args = p.parse_args()

    if not 0.0 < float(args.optimizer_deadline_seconds) < float(args.decision_runtime_budget_seconds) < 600.0:
        raise ValueError("Direct-TFV Development runtime requires optimizer deadline < controller budget < 600 s")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    step1 = load_frozen_step1_v127(args.step1, device)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.step2, graph=graph, device=device)
    _require_step1_lineage(checkpoint, step1=step1, sensors=sensors)
    admission = _load_admission(args.admission_calibration, step2_path=args.step2)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    event_clock = inspect_prepared_event_clock(args.inp)
    if abs(float(event_clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1e-6:
        raise ValueError("Direct-TFV event does not satisfy the common prepared-event clock")
    controller_cfg = _controller_config(dict(cfg["controller"]), control_block_steps=2)
    controller_cfg = replace(
        controller_cfg,
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        fallback_policy_id="HOLD_DIRECT_TFV_RUNTIME_FALLBACK",
    )
    controller_cfg.validate()
    design = DirectTFVMPCDesignV4(
        maxiter=int(args.lbfgsb_maxiter),
        deadline_seconds=float(args.optimizer_deadline_seconds),
        active_facility_count=int(args.active_facilities),
        active_support_quantile=str(args.active_support_quantile),
    )
    mpc = DirectTFVRecedingMPCV5(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        admission_calibration=admission,
        design=design,
    )
    controller = DirectTFVAuthoritativeController(
        step1=step1,
        mpc=mpc,
        graph=graph,
        sensor_nodes=sensors,
        forecast=PersistenceDecayForecast(decay_per_step=0.92, scenario_multipliers=(1.0,), history_steps_for_level=1),
        config=controller_cfg,
        device=device,
    )
    controller = ContinuityGuardController(controller, max_delta_per_update=0.5, allow_projection=False, enforce_current_delta=False)
    runtime_inp = _controls_disabled_runtime(
        source_inp=Path(args.inp), cache_dir=Path(args.out_dir) / "_runtime_inp", swmm_threads=int(cfg.get("swmm_threads", 1))
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
        raise RuntimeError("Direct-TFV authoritative run failed target write/readback audit")

    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    support = dict(checkpoint["action_support"])
    metadata.update(
        {
            "strategy": "proposed_direct_tfv_all109_receding_mpc",
            "direct_tfv_development_runtime_contract": DIRECT_TFV_DEVELOPMENT_RUNTIME_CONTRACT,
            "development_only": True,
            "tfv_primary": True,
            "all_109_facilities_screened_each_decision": True,
            "dynamic_active_set": True,
            "calibrated_one_sided_admission_used": True,
            "calibrated_admission_contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
            "action_rule": "execute iff calibrated upper bound on delta TFV < 0 and first move changes",
            "admission_coverage": float(admission["coverage"]),
            "admission_global_margin_m3": float(admission["global_margin_m3"]),
            "admission_dense_margin_m3": float(admission["dense_margin_m3"]),
            "admission_density_floor_changed_facilities": int(admission["density_floor_changed_facilities"]),
            "admission_calibration_sha256": _sha(args.admission_calibration),
            "active_support_quantile_requested": str(args.active_support_quantile),
            "active_support_quantile_effective": mpc.active_support_quantile_effective(),
            "active_support_ceiling": mpc.active_support_ceiling(),
            "training_joint_changed_facility_q90": float(support.get("joint_changed_facility_count_q90", 0.0)),
            "training_joint_changed_facility_q95": support.get("joint_changed_facility_count_q95"),
            "training_joint_changed_facility_q99": support.get("joint_changed_facility_count_q99"),
            "training_joint_changed_facility_max": support.get("joint_changed_facility_count_max"),
            "free_control_horizon_minutes": 120,
            "prediction_horizon_minutes": 360,
            "execute_first_move_minutes": 10,
            "future_realized_rainfall_used_as_model_input": False,
            "score_equals_execute_required": True,
            "target_write_readback_audit": write_audit,
            "step2_model_sha256": _sha(args.step2),
            "step2_training_contract": str(checkpoint.get("training_contract", "")),
            "graph_schema_sha256": _sha(args.graph),
            "step1_model_file_sha256": _sha(args.step1),
            "sensor_layout_file_sha256": _sha(args.sensors),
            "source_inp_sha256": _sha(args.inp),
            "controller_config_sha256": _sha(args.config),
            "project7_runtime_contract": project_contract,
            "prepared_event_clock": event_clock,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "strategy": metadata["strategy"],
        "runtime_contract": DIRECT_TFV_DEVELOPMENT_RUNTIME_CONTRACT,
        "metadata_path": result.metadata_path,
        "decision_path": result.decision_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "target_write_readback_passed": True,
        "active_support_quantile_effective": metadata["active_support_quantile_effective"],
        "active_support_ceiling": metadata["active_support_ceiling"],
        "admission_global_margin_m3": metadata["admission_global_margin_m3"],
        "admission_dense_margin_m3": metadata["admission_dense_margin_m3"],
        "sampled_global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
