"""Project7 final simplified production path: Step1 -> TFV Value policy -> first move.

No nodewise Hydraulic surrogate is loaded or required. Authoritative SWMM remains
the closed-loop environment and final evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .closed_loop import run_authoritative_closed_loop
from .controller import TorchMPCController
from .forecast import PersistenceDecayForecast
from .production_cli import (
    _controller_config,
    _controls_disabled_runtime,
    _load_graph,
    _load_lines,
    _load_step1,
)
from .runtime_controller_guard import ContinuityGuardController
from .step2_runtime_v120 import load_value_only_policy_v120, v120_bundle_metadata


def run_policy_v120_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Project7 sparse-sensor TFV-value-only rolling RTC"
    )
    parser.add_argument("--strategy", required=True, choices=("proposed",))
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--priority")
    parser.add_argument("--calibration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step1", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--runtime-inp-cache-dir")
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-development-v120",
        action="store_true",
        help="allow a runtime-compatible but unpromoted V120 bundle for one bounded development smoke",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("controller config must be a JSON object")
    model_step_seconds = int(cfg["model_step_seconds"])
    control_update_seconds = int(cfg["control_update_seconds"])
    record_stride_seconds = int(cfg.get("record_stride_seconds", model_step_seconds))
    if (model_step_seconds, control_update_seconds) != (300, 600):
        raise ValueError("V120 requires the frozen 300-s/600-s clock")
    if record_stride_seconds != model_step_seconds:
        raise ValueError("V120 record stride must equal the model step")
    controller_raw = cfg.get("controller", {})
    if not isinstance(controller_raw, dict):
        raise ValueError("config controller must be an object")
    if int(controller_raw.get("horizon_steps", -1)) != 72:
        raise ValueError("V120 requires the frozen 72-step value horizon")
    raw_delta = controller_raw.get("max_setting_delta_per_update")
    if raw_delta is None or float(raw_delta) > 0.5 + 1e-9:
        raise ValueError("V120 requires max setting delta <= 0.5/update")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    if not set(sensors).issubset(graph.node_ids):
        raise ValueError("V120 runtime sensor nodes are not all present in graph")
    step1 = _load_step1(args.step1, device)
    step1_meta = dict(getattr(step1, "runtime_metadata", {}))
    if int(step1_meta.get("model_step_seconds", -1)) != model_step_seconds:
        raise ValueError("Step1 model step differs from V120 controller")
    if int(step1_meta.get("history_steps", -1)) != int(controller_raw.get("history_steps", -1)):
        raise ValueError("Step1 history length differs from V120 controller")

    bundle_meta = v120_bundle_metadata(args.step2)
    if int(bundle_meta["value_horizon_minutes"]) != 360:
        raise ValueError("V120 bundle value horizon differs from frozen 360 min")
    if int(bundle_meta["control_update_seconds"]) != control_update_seconds:
        raise ValueError("V120 bundle decision interval differs from controller")
    step1_engine = str(step1_meta.get("swmm_engine_version", "")).strip()
    bundle_engine = str(bundle_meta["swmm_engine_version"]).strip()
    if not step1_engine or step1_engine != bundle_engine:
        raise ValueError(
            f"Step1/V120 SWMM engine lineage differs: {step1_engine} != {bundle_engine}"
        )

    objective = cfg.get("objective", {})
    if not isinstance(objective, dict):
        raise ValueError("config objective must be an object")
    policy = load_value_only_policy_v120(
        graph=graph,
        bundle_path=args.step2,
        device=device,
        allow_development=bool(args.allow_development_v120),
        cvar_alpha=float(objective.get("tfv_cvar_alpha", 0.90)),
        min_predicted_improvement_m3=float(
            objective.get("min_predicted_tfv_improvement_m3", 0.0)
        ),
        movement_tiebreak=float(objective.get("movement_tiebreak", 1.0e-6)),
    )
    forecast_cfg = cfg.get("forecast", {})
    if not isinstance(forecast_cfg, dict):
        raise ValueError("config forecast must be an object")
    forecast = PersistenceDecayForecast(
        decay_per_step=float(forecast_cfg.get("decay_per_step", 0.92)),
        scenario_multipliers=tuple(
            float(x) for x in forecast_cfg.get("scenario_multipliers", [0.75, 1.0, 1.25])
        ),
        history_steps_for_level=int(forecast_cfg.get("history_steps_for_level", 3)),
    )
    control_block_steps = control_update_seconds // model_step_seconds
    controller = TorchMPCController(
        step1=step1,
        mpc=policy,
        graph=graph,
        sensor_nodes=sensors,
        forecast=forecast,
        config=_controller_config(controller_raw, control_block_steps=control_block_steps),
        device=device,
    )
    controller = ContinuityGuardController(
        controller,
        max_delta_per_update=float(raw_delta),
        allow_projection=False,
    )

    source_inp = Path(args.inp)
    cache_dir = (
        Path(args.runtime_inp_cache_dir)
        if args.runtime_inp_cache_dir
        else Path(args.out_dir) / "_runtime_inp"
    )
    runtime_inp = _controls_disabled_runtime(
        source_inp=source_inp,
        cache_dir=cache_dir,
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=args.out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=int(cfg.get("control_start_minutes", 0)),
        control_update_seconds=control_update_seconds,
        observation_update_seconds=model_step_seconds,
        record_stride_seconds=record_stride_seconds,
        exact_global_peak=bool(cfg.get("exact_global_peak", False)),
    )
    print(
        json.dumps(
            {
                "strategy": "proposed",
                "step2": "V120_TFV_VALUE_ONLY",
                "primary_objective": "whole_system_cumulative_TFV_m3",
                "nodewise_hydraulic_surrogate_online": False,
                "value_horizon_minutes": 360,
                "control_update_seconds": 600,
                "first_move_only": True,
                "native_controls_enabled": False,
                "source_inp": str(source_inp.resolve()),
                "runtime_inp": str(runtime_inp.resolve()),
                "metadata_path": result.metadata_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
                "flow_routing_error_pct": result.flow_routing_error_pct,
            },
            indent=2,
        )
    )


__all__ = ["run_policy_v120_main"]
