"""Run Project7 V127 continuous differentiable MPC in authoritative SWMM closed loop."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from rtc.checkpoint_v127 import load_step2_v127
from rtc.closed_loop import run_authoritative_closed_loop
from rtc.controller_v127 import V127TorchMPCController
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import _controller_config, _controls_disabled_runtime, _load_graph, _load_lines, _load_step1
from rtc.runtime_controller_guard import ContinuityGuardController
from rtc.step3_mpc_v127 import (
    ContinuousMPCDesignV127,
    DifferentiableRollingMPCV127,
    Step2GradientEvidenceV127,
    V127_STEP3_CONTRACT,
)

V127_RUNTIME_CONTRACT = "PROJECT7_V127_AUTHORITATIVE_10MIN_CONTINUOUS_RTC_V1"
V127_GATE_CONTRACT = "PROJECT7_V127_CONTINUOUS_MPC_EVIDENCE_GATE_V1"


def _priority_indices(path: str | Path, graph, device: torch.device) -> torch.Tensor:
    nodes = tuple(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(nodes) != 8 or len(set(nodes)) != 8:
        raise ValueError("V127 requires the frozen unique Priority8 list")
    missing = sorted(set(nodes) - set(graph.node_ids))
    if missing:
        raise ValueError(f"V127 Priority8 nodes absent from graph: {missing}")
    return torch.as_tensor([graph.node_ids.index(node) for node in nodes], dtype=torch.long, device=device)


def _evidence(path: str | Path) -> tuple[Step2GradientEvidenceV127, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != V127_GATE_CONTRACT or payload.get("passed") is not True:
        raise ValueError("V127 runtime requires a passed continuous-MPC evidence gate")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("V127 continuous gate lacks metrics")
    evidence = Step2GradientEvidenceV127(**metrics)
    evidence.validate()
    return evidence, payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--priority-nodes", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--continuous-gate", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=540.0)
    p.add_argument("--pfv-soft-margin-m3", type=float, default=100.0)
    p.add_argument("--pfv-penalty-weight", type=float, default=1.0)
    p.add_argument("--min-improvement-vs-rbc-m3", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    step1 = _load_step1(args.step1, device)
    step2, step2_payload = load_step2_v127(args.step2, graph=graph, device=device)
    evidence, gate_payload = _evidence(args.continuous_gate)
    priority = _priority_indices(args.priority_nodes, graph, device)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if int(cfg.get("model_step_seconds", -1)) != 300 or int(cfg.get("control_update_seconds", -1)) != 600:
        raise ValueError("V127 runtime requires 300-s model observations and 600-s decisions")
    controller_cfg = _controller_config(dict(cfg["controller"]), control_block_steps=2)
    controller_cfg = replace(
        controller_cfg,
        horizon_steps=72,
        control_block_steps=2,
        max_setting_delta_per_update=0.5,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        fallback_policy_id="SPARSE_RBC_SAFETY_FALLBACK_V127",
    )
    controller_cfg.validate()
    design = ContinuousMPCDesignV127(
        lbfgsb_maxiter=int(args.lbfgsb_maxiter),
        pfv_soft_margin_m3=float(args.pfv_soft_margin_m3),
        pfv_penalty_weight=float(args.pfv_penalty_weight),
        min_improvement_vs_rbc_m3=float(args.min_improvement_vs_rbc_m3),
    )
    mpc = DifferentiableRollingMPCV127(
        model=step2,
        graph=graph,
        priority_indices=priority,
        evidence=evidence,
        flood_rate_index=2,
        design=design,
    )
    controller = V127TorchMPCController(
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
        controller, max_delta_per_update=0.5, allow_projection=False
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
        control_start_minutes=60,
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=bool(cfg.get("exact_global_peak", False)),
    )
    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "strategy": "proposed_v127_continuous_differentiable_mpc",
        "v127_runtime_contract": V127_RUNTIME_CONTRACT,
        "v127_step3_contract": V127_STEP3_CONTRACT,
        "continuous_gradient_search": True,
        "free_control_horizon_minutes": 120,
        "prediction_horizon_minutes": 360,
        "continuous_variable_count": 12 * 109,
        "all_writable_actuators_eligible": True,
        "rbc_role": "warm_start_and_safety_fallback_only",
        "rbc_is_value_reference": False,
        "rbc_is_action_space_ceiling": False,
        "tfv_primary": True,
        "priority8_pfv_soft_secondary": True,
        "global_peak_report_only": True,
        "future_realized_rainfall_used_as_model_input": False,
        "step2_checkpoint_contract": step2_payload.get("checkpoint_contract"),
        "step2_contract": step2_payload.get("step2_contract"),
        "continuous_gate": gate_payload,
        "lbfgsb_maxiter": design.lbfgsb_maxiter,
        "decision_runtime_budget_seconds": controller_cfg.decision_runtime_budget_seconds,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "strategy": metadata["strategy"],
        "runtime_contract": V127_RUNTIME_CONTRACT,
        "metadata_path": result.metadata_path,
        "decision_path": result.decision_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
