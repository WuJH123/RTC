from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .baselines import write_passive_no_rtc_inp
from .calibration import SafetyCalibration
from .closed_loop import CausalObservation, ControllerAction, run_authoritative_closed_loop
from .contracts import load_priority_nodes
from .controller import ControllerConfig, TorchMPCController
from .forecast import PersistenceDecayForecast
from .graph import GraphSchema
from .models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from .mpc import ContinuousSafetyMPC, StateLayout


def _load_lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _load_graph(path: str | Path) -> GraphSchema:
    raw = np.load(path, allow_pickle=False)
    return GraphSchema(
        node_ids=tuple(raw["node_ids"].astype(str).tolist()),
        edge_index=raw["edge_index"].astype(np.int64),
        static_node_features=raw["static_node_features"].astype(np.float32),
        static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
        actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
        actuator_upstream=raw["actuator_upstream"].astype(np.int64),
        actuator_downstream=raw["actuator_downstream"].astype(np.int64),
        actuator_physics=raw["actuator_physics"].astype(np.float32),
        actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
        system_units=str(raw["system_units"].item()),
    )


def _load_step1(path: str | Path, device: torch.device) -> SparseStateEstimator:
    payload = torch.load(path, map_location=device)
    if payload.get("scientific_split") != "development":
        raise ValueError("Step1 checkpoint was not trained under development-only lineage")
    cfg = dict(payload["model_config"])
    cfg.pop("state_weights", None)
    model = SparseStateEstimator(**cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def _load_step2(path: str | Path, device: torch.device) -> DifferentiableHydraulicWorldModel:
    payload = torch.load(path, map_location=device)
    if payload.get("scientific_split") != "development":
        raise ValueError("Step2 checkpoint was not trained under development-only lineage")
    cfg = dict(payload["model_config"])
    cfg.pop("state_weights", None)
    cfg.pop("flow_loss_weight", None)
    model = DifferentiableHydraulicWorldModel(**cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def _site_vector(config: dict[str, object], key: str, priority: tuple[str, ...]) -> np.ndarray:
    raw = config.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"production config requires object {key}")
    missing = [node for node in priority if node not in raw]
    if missing:
        raise ValueError(f"{key} missing priority nodes: {missing}")
    return np.asarray([float(raw[node]) for node in priority], dtype=np.float32)


def _controller_config(raw: dict[str, object], *, control_block_steps: int) -> ControllerConfig:
    allowed = {f.name for f in fields(ControllerConfig)}
    payload = {k: v for k, v in raw.items() if k in allowed}
    payload["control_block_steps"] = control_block_steps
    return ControllerConfig(**payload)


def _constant_controller(value: float):
    def controller(obs: CausalObservation) -> ControllerAction:
        return ControllerAction(
            settings={aid: float(value) for aid in obs.actuator_ids},
            source="ALL_OPEN" if value >= 1.0 else "ALL_CLOSED",
        )
    return controller


def _frozen_hold_controller():
    frozen: np.ndarray | None = None

    def controller(obs: CausalObservation) -> ControllerAction:
        nonlocal frozen
        if frozen is None:
            frozen = np.asarray(obs.actuator_current_setting, dtype=float).copy()
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, frozen, strict=True)),
            source="FROZEN_HOLD",
        )
    return controller


def run_policy_main() -> None:
    parser = argparse.ArgumentParser(description="Run proposed RTC or a frozen authoritative baseline")
    parser.add_argument("--strategy", required=True, choices=[
        "proposed", "native_rules", "passive_no_rtc", "hold", "all_open", "all_closed"
    ])
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph")
    parser.add_argument("--step1")
    parser.add_argument("--step2")
    parser.add_argument("--calibration")
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_step_seconds = int(cfg["model_step_seconds"])
    control_update_seconds = int(cfg["control_update_seconds"])
    record_stride_seconds = int(cfg.get("record_stride_seconds", model_step_seconds))
    if model_step_seconds <= 0 or control_update_seconds % model_step_seconds:
        raise ValueError("control_update_seconds must be a positive integer multiple of model_step_seconds")
    control_block_steps = control_update_seconds // model_step_seconds
    control_start_minutes = int(cfg.get("control_start_minutes", 0))
    sensors = _load_lines(args.sensors)
    priority = load_priority_nodes(args.priority)
    controller = None
    inp_for_run = Path(args.inp)

    if args.strategy == "proposed":
        for name, value in (("graph", args.graph), ("step1", args.step1), ("step2", args.step2), ("calibration", args.calibration)):
            if not value:
                raise ValueError(f"--{name} is required for proposed")
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        graph = _load_graph(args.graph)
        if not set(priority).issubset(graph.node_ids):
            raise ValueError("priority nodes are not all present in the frozen graph schema")
        if not set(sensors).issubset(graph.node_ids):
            raise ValueError("sensor nodes are not all present in the frozen graph schema")
        step1 = _load_step1(args.step1, device)
        step2 = _load_step2(args.step2, device)
        calibration = SafetyCalibration.from_json(args.calibration)
        if calibration.priority_nodes != priority:
            raise ValueError("calibration priority-node order differs from production priority file")
        priority_index = torch.as_tensor(
            [graph.node_ids.index(node) for node in priority], dtype=torch.long, device=device
        )
        safety_cfg = cfg.get("safety", {})
        if not isinstance(safety_cfg, dict):
            raise ValueError("config safety must be an object")
        mpc = ContinuousSafetyMPC(
            step2,
            layout=StateLayout(depth_index=int(cfg.get("depth_state_index", 0)), flood_rate_index=int(cfg.get("flood_rate_state_index", 2))),
            priority_indices=priority_index,
            dt_seconds=model_step_seconds,
            per_site_flood_budget_m3=torch.as_tensor(_site_vector(cfg, "flood_budget_m3", priority), device=device),
            per_site_depth_budget_m=torch.as_tensor(_site_vector(cfg, "depth_budget_m", priority), device=device),
            flood_error_ucb_m3=torch.as_tensor(calibration.flood_error_ucb_m3, dtype=torch.float32, device=device),
            depth_error_ucb_m=torch.as_tensor(calibration.depth_error_ucb_m, dtype=torch.float32, device=device),
            forecast_safety_quantile=float(safety_cfg.get("forecast_safety_quantile", 0.95)),
            tfv_cvar_alpha=float(safety_cfg.get("tfv_cvar_alpha", 0.90)),
            safety_penalty=float(safety_cfg.get("safety_penalty", 1e3)),
            movement_tiebreak=float(safety_cfg.get("movement_tiebreak", 1e-5)),
        )
        forecast_cfg = cfg.get("forecast", {})
        if not isinstance(forecast_cfg, dict):
            raise ValueError("config forecast must be an object")
        forecast = PersistenceDecayForecast(
            decay_per_step=float(forecast_cfg.get("decay_per_step", 0.92)),
            scenario_multipliers=tuple(float(x) for x in forecast_cfg.get("scenario_multipliers", [0.75, 1.0, 1.25])),
            history_steps_for_level=int(forecast_cfg.get("history_steps_for_level", 3)),
        )
        controller_cfg_raw = cfg.get("controller", {})
        if not isinstance(controller_cfg_raw, dict):
            raise ValueError("config controller must be an object")
        controller = TorchMPCController(
            step1=step1,
            mpc=mpc,
            graph=graph,
            sensor_nodes=sensors,
            forecast=forecast,
            config=_controller_config(controller_cfg_raw, control_block_steps=control_block_steps),
            device=device,
        )
    elif args.strategy == "passive_no_rtc":
        inp_for_run = Path(args.out_dir) / f"{args.run_id}.passive_no_rtc.inp"
        write_passive_no_rtc_inp(args.inp, inp_for_run)
    elif args.strategy == "hold":
        controller = _frozen_hold_controller()
    elif args.strategy == "all_open":
        controller = _constant_controller(1.0)
    elif args.strategy == "all_closed":
        controller = _constant_controller(0.0)

    result = run_authoritative_closed_loop(
        inp_path=inp_for_run,
        output_dir=args.out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=control_start_minutes,
        control_update_seconds=control_update_seconds,
        observation_update_seconds=model_step_seconds,
        record_stride_seconds=record_stride_seconds,
        exact_global_peak=bool(cfg.get("exact_global_peak", True)),
    )
    payload = {
        "strategy": args.strategy,
        "metadata_path": result.metadata_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }
    print(json.dumps(payload, indent=2))
