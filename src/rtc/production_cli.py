from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .baselines import canonical_baseline_id, write_no_control_inp
from .calibration import SafetyCalibration
from .closed_loop import CausalObservation, ControllerAction, run_authoritative_closed_loop
from .contracts import load_priority_nodes
from .controller import ControllerConfig, TorchMPCController
from .forecast import PersistenceDecayForecast
from .graph import GraphSchema
from .models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from .tfv_mpc import ContinuousTFVFirstMPC


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


def _priority_and_calibration(
    *,
    priority_path: str | None,
    calibration_path: str | None,
    graph: GraphSchema,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | float, torch.Tensor | float]:
    if not priority_path:
        if calibration_path:
            raise ValueError("--calibration requires --priority")
        return None, 0.0, 0.0
    priority = load_priority_nodes(priority_path)
    missing = sorted(set(priority) - set(graph.node_ids))
    if missing:
        raise ValueError(
            "priority mapping is incompatible with this INP/graph; do not guess IDs. "
            f"Missing: {missing}"
        )
    pidx = torch.as_tensor([graph.node_ids.index(node) for node in priority], dtype=torch.long, device=device)
    if not calibration_path:
        return pidx, 0.0, 0.0
    calibration = SafetyCalibration.from_json(calibration_path)
    if calibration.priority_nodes != priority:
        raise ValueError("calibration priority-node order differs from priority file")
    return (
        pidx,
        torch.as_tensor(calibration.flood_error_ucb_m3, dtype=torch.float32, device=device),
        torch.as_tensor(calibration.depth_error_ucb_m, dtype=torch.float32, device=device),
    )


def run_policy_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TFV-first Proposed RTC or an explicitly separated authoritative baseline"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "proposed", "internal_rtc", "no_control", "hold", "all_open", "all_closed",
            "native_rules", "passive_no_rtc",
        ],
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--priority", help="optional soft-priority node file; Formal use requires a verified mapping")
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph")
    parser.add_argument("--step1")
    parser.add_argument("--step2")
    parser.add_argument("--calibration")
    parser.add_argument("--device")
    args = parser.parse_args()

    strategy = canonical_baseline_id(args.strategy)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_step_seconds = int(cfg["model_step_seconds"])
    control_update_seconds = int(cfg["control_update_seconds"])
    record_stride_seconds = int(cfg.get("record_stride_seconds", model_step_seconds))
    if model_step_seconds <= 0 or control_update_seconds % model_step_seconds:
        raise ValueError("control_update_seconds must be a positive integer multiple of model_step_seconds")
    control_block_steps = control_update_seconds // model_step_seconds
    control_start_minutes = int(cfg.get("control_start_minutes", 0))
    sensors = _load_lines(args.sensors)
    controller = None
    source_inp = Path(args.inp)
    inp_for_run = source_inp

    # Only Internal-RTC is allowed to execute the original [CONTROLS]. Every Python policy
    # and No-control use a controls-disabled physical copy from simulation start.
    if strategy != "internal_rtc":
        inp_for_run = Path(args.out_dir) / f"{args.run_id}.controls_disabled.inp"
        write_no_control_inp(
            source_inp,
            inp_for_run,
            swmm_threads=int(cfg.get("swmm_threads", 1)),
        )

    if strategy == "proposed":
        for name, value in (("graph", args.graph), ("step1", args.step1), ("step2", args.step2)):
            if not value:
                raise ValueError(f"--{name} is required for proposed")
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        graph = _load_graph(args.graph)
        if not set(sensors).issubset(graph.node_ids):
            raise ValueError("sensor nodes are not all present in the frozen graph schema")
        step1 = _load_step1(args.step1, device)
        step2 = _load_step2(args.step2, device)
        pidx, flood_ucb, depth_ucb = _priority_and_calibration(
            priority_path=args.priority,
            calibration_path=args.calibration,
            graph=graph,
            device=device,
        )
        objective_cfg = cfg.get("objective", {})
        if not isinstance(objective_cfg, dict):
            raise ValueError("config objective must be an object")
        mpc = ContinuousTFVFirstMPC(
            step2,
            depth_index=int(cfg.get("depth_state_index", 0)),
            flood_rate_index=int(cfg.get("flood_rate_state_index", 2)),
            priority_indices=pidx,
            dt_seconds=model_step_seconds,
            flood_error_ucb_m3=flood_ucb,
            depth_error_ucb_m=depth_ucb,
            forecast_quantile=float(objective_cfg.get("forecast_quantile", 0.95)),
            tfv_cvar_alpha=float(objective_cfg.get("tfv_cvar_alpha", 0.90)),
            tfv_near_opt_relative=float(objective_cfg.get("tfv_near_opt_relative", 0.01)),
            tfv_near_opt_absolute_m3=float(objective_cfg.get("tfv_near_opt_absolute_m3", 1.0)),
            near_opt_penalty=float(objective_cfg.get("near_opt_penalty", 1e4)),
            movement_tiebreak=float(objective_cfg.get("movement_tiebreak", 1e-6)),
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
    elif strategy == "hold":
        controller = _frozen_hold_controller()
    elif strategy == "all_open":
        controller = _constant_controller(1.0)
    elif strategy == "all_closed":
        controller = _constant_controller(0.0)
    # internal_rtc and no_control deliberately have controller=None.

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
        exact_global_peak=bool(cfg.get("exact_global_peak", False)),
    )
    payload = {
        "strategy": strategy,
        "source_inp": str(source_inp.resolve()),
        "runtime_inp": str(inp_for_run.resolve()),
        "native_controls_enabled": strategy == "internal_rtc",
        "metadata_path": result.metadata_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }
    print(json.dumps(payload, indent=2))
