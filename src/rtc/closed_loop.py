from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .inp import discover_actuators, discover_nodes
from .swmm_data import sha256_file


@dataclass(frozen=True)
class CausalObservation:
    elapsed_seconds: int
    current_time: datetime
    sensor_ids: tuple[str, ...]
    sensor_depth: np.ndarray
    sensor_head: np.ndarray
    actuator_ids: tuple[str, ...]
    actuator_target_setting: np.ndarray
    actuator_current_setting: np.ndarray
    actuator_flow: np.ndarray
    observed_rainfall: np.ndarray


@dataclass(frozen=True)
class ControllerAction:
    settings: Mapping[str, float]
    source: str = "MPC"
    diagnostics: Mapping[str, float | int | bool | str] | None = None


@dataclass(frozen=True)
class ClosedLoopResult:
    metadata_path: str
    node_path: str
    actuator_path: str
    decision_path: str
    decisions: int
    flow_routing_error_pct: float


Controller = Callable[[CausalObservation], ControllerAction | Mapping[str, float]]
RainfallObserver = Callable[[datetime], np.ndarray]


def _normalize_action(action: ControllerAction | Mapping[str, float]) -> ControllerAction:
    if isinstance(action, ControllerAction):
        return action
    return ControllerAction(settings=action)


def run_authoritative_closed_loop(
    *,
    inp_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    sensor_nodes: tuple[str, ...],
    controller: Controller | None,
    rainfall_observer: RainfallObserver | None = None,
    control_start_minutes: int = 0,
    control_update_seconds: int = 600,
    record_stride_seconds: int = 300,
) -> ClosedLoopResult:
    """Run one causally isolated closed-loop policy inside authoritative PySWMM.

    The controller receives *only* sparse sensor hydraulics, actuator readback and rainfall
    observed up to the current callback time. Full-network SWMM truth is written only to
    evaluation files and is never included in ``CausalObservation``. If ``controller`` is
    ``None``, native SWMM rules run untouched with no Python setting writes.
    """

    try:
        from pyswmm import Links, Nodes, Simulation
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc

    if control_update_seconds <= 0 or record_stride_seconds <= 0:
        raise ValueError("control/update strides must be positive")
    if control_update_seconds % record_stride_seconds:
        raise ValueError("control_update_seconds must be a multiple of record_stride_seconds")

    inp = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp)
    all_nodes = discover_nodes(inp)
    missing_sensors = sorted(set(sensor_nodes) - set(all_nodes))
    if missing_sensors:
        raise ValueError(f"sensor nodes absent from INP: {missing_sensors}")

    node_path = out / f"{run_id}.nodes.csv.gz"
    actuator_path = out / f"{run_id}.actuators.csv.gz"
    decision_path = out / f"{run_id}.decisions.jsonl"
    metadata_path = out / f"{run_id}.json"
    start_control_seconds = int(control_start_minutes * 60)
    next_decision = start_control_seconds
    decision_count = 0
    held_settings: dict[str, float] | None = None

    with Simulation(str(inp)) as sim, gzip.open(node_path, "wt", encoding="utf-8", newline="") as node_fh, gzip.open(
        actuator_path, "wt", encoding="utf-8", newline=""
    ) as act_fh, decision_path.open("w", encoding="utf-8") as decision_fh:
        sim.step_advance(record_stride_seconds)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links = Links(sim)
        nodes = Nodes(sim)
        link_obj = {aid: links[aid] for aid in catalog.ids}
        node_obj = {nid: nodes[nid] for nid in all_nodes}
        sensor_obj = {nid: node_obj[nid] for nid in sensor_nodes}
        node_writer = csv.writer(node_fh)
        act_writer = csv.writer(act_fh)
        node_writer.writerow(
            ["elapsed_seconds", "datetime", "phase", "node_id", "depth", "head", "flooding", "volume"]
        )
        act_writer.writerow(
            [
                "elapsed_seconds",
                "datetime",
                "phase",
                "actuator_id",
                "requested_setting",
                "target_setting",
                "current_setting",
                "flow",
                "decision_source",
            ]
        )

        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            is_decision = controller is not None and elapsed >= next_decision
            phase = "DECISION" if is_decision else "OBSERVATION"

            # Record authoritative truth first. This file is evaluation evidence only.
            for nid, obj in node_obj.items():
                node_writer.writerow(
                    [elapsed, sim.current_time.isoformat(), phase, nid, obj.depth, obj.head, obj.flooding, obj.volume]
                )

            action: ControllerAction | None = None
            if is_decision:
                rainfall = (
                    np.asarray(rainfall_observer(sim.current_time), dtype=float)
                    if rainfall_observer is not None
                    else np.zeros(1, dtype=float)
                )
                observation = CausalObservation(
                    elapsed_seconds=elapsed,
                    current_time=sim.current_time,
                    sensor_ids=tuple(sensor_nodes),
                    sensor_depth=np.array([sensor_obj[n].depth for n in sensor_nodes], dtype=float),
                    sensor_head=np.array([sensor_obj[n].head for n in sensor_nodes], dtype=float),
                    actuator_ids=tuple(catalog.ids),
                    actuator_target_setting=np.array([link_obj[a].target_setting for a in catalog.ids], dtype=float),
                    actuator_current_setting=np.array([link_obj[a].current_setting for a in catalog.ids], dtype=float),
                    actuator_flow=np.array([link_obj[a].flow for a in catalog.ids], dtype=float),
                    observed_rainfall=rainfall,
                )
                action = _normalize_action(controller(observation))
                supplied = set(action.settings)
                expected = set(catalog.ids)
                if supplied != expected:
                    raise ValueError(
                        f"controller must return every actuator; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}"
                    )
                held_settings = {aid: float(action.settings[aid]) for aid in catalog.ids}
                bad = {aid: value for aid, value in held_settings.items() if not 0.0 <= value <= 1.0}
                if bad:
                    raise ValueError(f"controller settings outside [0,1]: {bad}")
                for aid, value in held_settings.items():
                    link_obj[aid].target_setting = value
                decision_count += 1
                next_decision += control_update_seconds
                decision_fh.write(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed,
                            "datetime": sim.current_time.isoformat(),
                            "source": action.source,
                            "settings": held_settings,
                            "diagnostics": dict(action.diagnostics or {}),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            elif held_settings is not None:
                # Reapply the held first move at every Python callback so native rules do
                # not silently replace the selected action between control updates.
                for aid, value in held_settings.items():
                    link_obj[aid].target_setting = value

            source = action.source if action is not None else ("HELD" if held_settings is not None else "NATIVE")
            for aid, obj in link_obj.items():
                requested = "" if held_settings is None else held_settings[aid]
                act_writer.writerow(
                    [
                        elapsed,
                        sim.current_time.isoformat(),
                        phase,
                        aid,
                        requested,
                        obj.target_setting,
                        obj.current_setting,
                        obj.flow,
                        source,
                    ]
                )

        flow_error = float(sim.flow_routing_error)

    metadata = {
        "run_id": run_id,
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "controller_present": controller is not None,
        "causal_controller_observation": "sparse_sensors+actuator_readback+observed_rainfall_only",
        "sensor_nodes": list(sensor_nodes),
        "actuator_ids": list(catalog.ids),
        "control_start_minutes": int(control_start_minutes),
        "control_update_seconds": int(control_update_seconds),
        "record_stride_seconds": int(record_stride_seconds),
        "flow_units": flow_units,
        "system_units": system_units,
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "decisions": decision_count,
        "node_file": node_path.name,
        "actuator_file": actuator_path.name,
        "decision_file": decision_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ClosedLoopResult(
        metadata_path=str(metadata_path),
        node_path=str(node_path),
        actuator_path=str(actuator_path),
        decision_path=str(decision_path),
        decisions=decision_count,
        flow_routing_error_pct=flow_error,
    )
