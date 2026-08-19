"""Run policy-matched Direct-TFV V7 in an authoritative Development SWMM closed loop."""
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
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
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
from rtc.step3_tfv_value_mpc_v7 import DirectTFVRecedingMPCV7


DIRECT_TFV_DEVELOPMENT_RUNTIME_CONTRACT = (
    "PROJECT7_DIRECT_TFV_AUTHORITATIVE_10MIN_DEVELOPMENT_RTC_V6"
)
DIRECT_TFV_CAUSAL_FORECAST_CONTRACT = (
    "PersistenceDecayForecast(history_steps_for_level=1,decay_per_step=0.92,scenario_multiplier=1.0)"
)


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


def _load_policy_admission(path: str | Path, *, step2_path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("runtime requires the current policy-matched Direct-TFV admission artifact")
    if payload.get("development_only") is not True:
        raise ValueError("policy admission artifact must be Development-only")
    if int(payload.get("policy_calibration_rainfall_group_count", 0)) < 9:
        raise ValueError("policy admission has insufficient rainfall-group calibration units")
    if payload.get("legacy_optimizer_replay_controls_current_margin") is not False:
        raise ValueError("legacy pre-V6 optimizer residuals must not control current policy admission")
    if payload.get("policy_conformal_coverage_claimed") is not True:
        raise ValueError("policy admission lacks its rainfall-group conformal calibration claim")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("policy admission lacks lineage")
    trained = str(lineage.get("step2_checkpoint_sha256", "")).lower()
    if not trained or trained != _sha(step2_path).lower():
        raise ValueError("policy admission was derived from a different Step2 checkpoint")
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
    p.add_argument("--policy-admission-calibration", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--optimizer-deadline-seconds", type=float, default=120.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    p.add_argument("--active-facilities", type=int, default=0)
    p.add_argument("--active-support-quantile", choices=("q90", "q95", "q99"), default="q95")
    args = p.parse_args()

    if str(args.active_support_quantile) != "q95":
        raise ValueError("current V7 Development policy is preregistered on canonical q95 only")
    if not 0.0 < float(args.optimizer_deadline_seconds) < float(args.decision_runtime_budget_seconds) < 600.0:
        raise ValueError("Direct-TFV Development runtime requires optimizer deadline < controller budget < 600 s")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    step1 = load_frozen_step1_v127(args.step1, device)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.step2, graph=graph, device=device)
    _require_step1_lineage(checkpoint, step1=step1, sensors=sensors)
    admission = _load_policy_admission(args.policy_admission_calibration, step2_path=args.step2)
    sequence_support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        sequence_support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=_sha(args.step2),
    )

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
        active_support_quantile="q95",
    )
    mpc = DirectTFVRecedingMPCV7(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        policy_admission_calibration=admission,
        sequence_support=sequence_support,
        design=design,
    )
    controller = DirectTFVAuthoritativeController(
        step1=step1,
        mpc=mpc,
        graph=graph,
        sensor_nodes=sensors,
        forecast=PersistenceDecayForecast(
            decay_per_step=0.92,
            scenario_multipliers=(1.0,),
            history_steps_for_level=1,
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
        raise RuntimeError("Direct-TFV V7 authoritative run failed target write/readback audit")

    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    support = dict(checkpoint["action_support"])
    metadata.update(
        {
            # Keep the stable family identifier consumed by the existing closed-loop audit; the
            # exact current policy is bound by the explicit V7 contract fields below.
            "strategy": "proposed_direct_tfv_all109_receding_mpc",
            "direct_tfv_development_runtime_contract": DIRECT_TFV_DEVELOPMENT_RUNTIME_CONTRACT,
            "direct_tfv_step3_contract": mpc.policy_mode_contract,
            "development_only": True,
            "tfv_primary": True,
            "all_109_facilities_screened_each_decision": True,
            "dynamic_active_set": True,
            "policy_matched_one_sided_admission_used": True,
            "policy_admission_contract": DIRECT_TFV_POLICY_ADMISSION_CONTRACT,
            "policy_admission_sha256": _sha(args.policy_admission_calibration),
            "policy_calibration_rainfall_group_count": int(admission["policy_calibration_rainfall_group_count"]),
            "policy_optimizer_residual_conformal_upper_m3": float(admission["policy_optimizer_residual_conformal_upper_m3"]),
            "legacy_optimizer_replay_controls_current_margin": False,
            "joint_sequence_support_used": True,
            "joint_sequence_support_contract": str(sequence_support["contract"]),
            "joint_sequence_support_sha256": _sha(args.sequence_support),
            "joint_sequence_support_label_independent": True,
            "action_rule": (
                "score q95 support-contracted H120 sequence; execute iff current-policy calibrated "
                "delta-TFV upper bound < 0 and first move changes"
            ),
            "d3_conformal_coverage": float(admission["d3_conformal_coverage"]),
            "admission_global_margin_m3": float(admission["global_margin_m3"]),
            "admission_dense_margin_m3": float(admission["dense_margin_m3"]),
            "admission_density_floor_changed_facilities": int(admission["density_floor_changed_facilities"]),
            "active_support_quantile_requested": "q95",
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
    print(
        json.dumps(
            {
                "strategy": metadata["strategy"],
                "step3_contract": metadata["direct_tfv_step3_contract"],
                "runtime_contract": DIRECT_TFV_DEVELOPMENT_RUNTIME_CONTRACT,
                "metadata_path": result.metadata_path,
                "decision_path": result.decision_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "target_write_readback_passed": True,
                "active_support_quantile_effective": metadata["active_support_quantile_effective"],
                "active_support_ceiling": metadata["active_support_ceiling"],
                "policy_admission_sha256": metadata["policy_admission_sha256"],
                "admission_global_margin_m3": metadata["admission_global_margin_m3"],
                "admission_dense_margin_m3": metadata["admission_dense_margin_m3"],
                "sampled_global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
                "flow_routing_error_pct": result.flow_routing_error_pct,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
