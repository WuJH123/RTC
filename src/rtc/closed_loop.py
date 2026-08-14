from __future__ import annotations

import csv
import gzip
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .forcing import current_node_rainfall_mmhr, resolve_subcatchment_outlets
from .inp import discover_actuators, discover_nodes
from .swmm_data import STATE_CHANNELS, _node_state_si, sha256_file
from .swmm_stats import snapshot_node_statistics, write_node_statistics
from .units import flow_rate_to_m3s, length_to_m


@dataclass(frozen=True)
class CausalObservation:
    elapsed_seconds: int
    current_time: datetime
    sensor_ids: tuple[str, ...]
    sensor_depth_m: np.ndarray
    sensor_head_m: np.ndarray
    actuator_ids: tuple[str, ...]
    actuator_target_setting: np.ndarray
    actuator_current_setting: np.ndarray
    actuator_flow_m3s: np.ndarray
    rainfall_node_ids: tuple[str, ...]
    observed_rainfall_mmhr: np.ndarray


@dataclass(frozen=True)
class ControllerAction:
    settings: Mapping[str, float]
    source: str = "MPC"
    diagnostics: Mapping[str, float | int | bool | str] | None = None


@dataclass(frozen=True)
class ClosedLoopResult:
    metadata_path: str
    compact_path: str
    decision_path: str
    node_statistics_path: str
    decisions: int
    global_peak_flood_rate_m3s: float
    flow_routing_error_pct: float
    node_path: str = ""
    actuator_path: str = ""


Controller = Callable[[CausalObservation], ControllerAction | Mapping[str, float]]
RainfallObserver = Callable[[datetime, tuple[str, ...]], np.ndarray]


def _observation_time(sim: object, elapsed_seconds: int) -> datetime:
    """Use the declared start timestamp before PySWMM starts iterating."""

    if int(elapsed_seconds) == 0:
        return getattr(sim, "start_time")
    return getattr(sim, "current_time")


def _normalize_action(action: ControllerAction | Mapping[str, float]) -> ControllerAction:
    return action if isinstance(action, ControllerAction) else ControllerAction(settings=action)


def _reassert_target_latch(link_obj: object, held_settings: Mapping[str, float] | None) -> float:
    """Re-apply the Python supervisory target before a causal observation.

    SWMM can internally update a pump's target setting during a routing step even
    when native controls are disabled.  The Python controller's target is a
    supervisory latch, so the previously issued values must be written again
    before the next observation is read.  Return the largest pre-write
    discrepancy as explicit diagnostic evidence; this is not a projection of a
    newly scored command.
    """

    if held_settings is None:
        return 0.0
    before = np.asarray(
        [float(link_obj[aid].target_setting) for aid in held_settings], dtype=float
    )
    for aid, value in held_settings.items():
        link_obj[aid].target_setting = float(value)
    after = np.asarray(
        [float(link_obj[aid].target_setting) for aid in held_settings], dtype=float
    )
    return float(np.abs(after - before).max(initial=0.0))


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
    observation_update_seconds: int | None = None,
    record_stride_seconds: int = 300,
    exact_global_peak: bool = False,
    save_raw_csv: bool = False,
    keep_engine_files: bool = False,
) -> ClosedLoopResult:
    """Run one causally isolated SWMM policy and save compact SI evidence.

    The initial state at ``t=0`` is observed and recorded before any supervisory write.
    Thus 13 frames on a 5-minute grid are exactly ``0,5,...,60 min``. Python callback
    cadence is separate from SWMM's internal routing step. Formal main runs keep
    ``exact_global_peak=False`` and obtain routing-step Global Peak by frozen-decision replay.
    Authoritative PFV/TFV always come from cumulative SWMM node statistics.
    """

    try:
        from pyswmm import Links, Nodes, Simulation, Subcatchments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc

    observation_update_seconds = int(observation_update_seconds or record_stride_seconds)
    if min(control_update_seconds, observation_update_seconds, record_stride_seconds) <= 0:
        raise ValueError("all runtime cadences must be positive")
    if control_update_seconds % observation_update_seconds:
        raise ValueError("control_update_seconds must be a multiple of observation_update_seconds")
    control_start_seconds = int(control_start_minutes * 60)
    if control_start_seconds < 0 or control_start_seconds % observation_update_seconds:
        raise ValueError("control start must be non-negative and align with observation cadence")

    inp = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp)
    all_nodes = tuple(discover_nodes(inp))
    actuator_ids = tuple(catalog.ids)
    missing_sensors = sorted(set(sensor_nodes) - set(all_nodes))
    if missing_sensors:
        raise ValueError(f"sensor nodes absent from INP: {missing_sensors}")

    compact_path = out / f"{run_id}.compact.npz"
    node_path = out / f"{run_id}.nodes.csv.gz"
    actuator_path = out / f"{run_id}.actuators.csv.gz"
    decision_path = out / f"{run_id}.decisions.jsonl"
    statistics_path = out / f"{run_id}.node_statistics.csv.gz"
    metadata_path = out / f"{run_id}.json"
    report_path = out / f"{run_id}.rpt"
    engine_output_path = out / f"{run_id}.out"

    next_observation = observation_update_seconds
    next_decision = control_start_seconds if control_start_seconds > 0 else observation_update_seconds
    next_record = record_stride_seconds
    decision_count = 0
    held_settings: dict[str, float] | None = None
    global_peak_m3s = 0.0

    record_times: list[int] = []
    state_values: list[np.ndarray] = []
    rainfall_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    flow_values: list[np.ndarray] = []

    with ExitStack() as stack:
        sim = stack.enter_context(
            Simulation(str(inp), reportfile=str(report_path), outputfile=str(engine_output_path))
        )
        decision_fh = stack.enter_context(decision_path.open("w", encoding="utf-8"))
        node_writer = act_writer = None
        if save_raw_csv:
            node_fh = stack.enter_context(gzip.open(node_path, "wt", encoding="utf-8", newline=""))
            act_fh = stack.enter_context(gzip.open(actuator_path, "wt", encoding="utf-8", newline=""))
            node_writer, act_writer = csv.writer(node_fh), csv.writer(act_fh)
            node_writer.writerow(["elapsed_seconds", "phase", "node_id", *STATE_CHANNELS])
            act_writer.writerow([
                "elapsed_seconds", "phase", "actuator_id", "requested_setting",
                "target_setting", "current_setting", "flow_m3s", "decision_source",
            ])

        callback_stride = math.gcd(
            math.gcd(control_update_seconds, observation_update_seconds), record_stride_seconds
        )
        if not exact_global_peak:
            sim.step_advance(callback_stride)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links, nodes, subcatchments = Links(sim), Nodes(sim), Subcatchments(sim)
        link_obj = {aid: links[aid] for aid in actuator_ids}
        node_obj = {nid: nodes[nid] for nid in all_nodes}
        sensor_obj = {nid: node_obj[nid] for nid in sensor_nodes}
        sub_obj = {obj.subcatchmentid: obj for obj in subcatchments}
        connection = {sid: obj.connection for sid, obj in sub_obj.items()}
        resolved_outlets = resolve_subcatchment_outlets(connection, all_nodes)

        def observed_rainfall() -> np.ndarray:
            if rainfall_observer is None:
                return current_node_rainfall_mmhr(sub_obj, resolved_outlets, all_nodes, system_units)
            rain = np.asarray(rainfall_observer(sim.current_time, all_nodes), dtype=float).reshape(-1)
            if rain.size == 1:
                rain = np.repeat(rain, len(all_nodes))
            if rain.size != len(all_nodes) or np.any(rain < 0) or not np.isfinite(rain).all():
                raise ValueError("rainfall_observer must return finite non-negative scalar/node vector")
            return rain.astype(np.float32)

        def build_observation(
            elapsed: int, rain: np.ndarray, *, current_time: datetime | None = None
        ) -> CausalObservation:
            return CausalObservation(
                elapsed_seconds=elapsed,
                current_time=(
                    _observation_time(sim, elapsed)
                    if current_time is None
                    else current_time
                ),
                sensor_ids=tuple(sensor_nodes),
                sensor_depth_m=length_to_m(
                    np.array([sensor_obj[n].depth for n in sensor_nodes], dtype=float), system_units
                ),
                sensor_head_m=length_to_m(
                    np.array([sensor_obj[n].head for n in sensor_nodes], dtype=float), system_units
                ),
                actuator_ids=actuator_ids,
                actuator_target_setting=np.array([link_obj[a].target_setting for a in actuator_ids], dtype=float),
                actuator_current_setting=np.array([link_obj[a].current_setting for a in actuator_ids], dtype=float),
                actuator_flow_m3s=flow_rate_to_m3s(
                    np.array([link_obj[a].flow for a in actuator_ids], dtype=float), flow_units
                ),
                rainfall_node_ids=all_nodes,
                observed_rainfall_mmhr=rain,
            )

        def append_record(elapsed: int, rain: np.ndarray, phase: str, source: str) -> None:
            state = _node_state_si(node_obj, all_nodes, system_units=system_units, flow_units=flow_units)
            target = np.array([link_obj[a].target_setting for a in actuator_ids], dtype=np.float32)
            current = np.array([link_obj[a].current_setting for a in actuator_ids], dtype=np.float32)
            flow = flow_rate_to_m3s(
                np.array([link_obj[a].flow for a in actuator_ids], dtype=float), flow_units
            ).astype(np.float32)
            record_times.append(elapsed)
            state_values.append(state)
            rainfall_values.append(rain[:, None].astype(np.float32))
            target_values.append(target)
            current_values.append(current)
            flow_values.append(flow)
            if save_raw_csv:
                assert node_writer is not None and act_writer is not None
                for ni, nid in enumerate(all_nodes):
                    node_writer.writerow([elapsed, phase, nid, *state[ni].tolist()])
                for ai, aid in enumerate(actuator_ids):
                    requested = "" if held_settings is None else held_settings[aid]
                    act_writer.writerow(
                        [elapsed, phase, aid, requested, target[ai], current[ai], flow[ai], source]
                    )

        # t=0 belongs to the causal information set. No supervisory command has yet been sent.
        rain_zero = observed_rainfall()
        if controller is not None and hasattr(controller, "observe"):
            controller.observe(  # type: ignore[attr-defined]
                build_observation(0, rain_zero, current_time=sim.start_time)
            )
        append_record(0, rain_zero, "INITIAL", "NATIVE")
        initial_total_flooding = sum(max(0.0, float(obj.flooding)) for obj in node_obj.values())
        global_peak_m3s = float(flow_rate_to_m3s(initial_total_flooding, flow_units))

        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            # Reassert the prior supervisory target before reading the next
            # decision observation.  This prevents SWMM's internal pump-state
            # update from being mistaken for a failed Python target write.
            target_latch_reasserted_max = _reassert_target_latch(link_obj, held_settings)
            total_flooding_native = sum(max(0.0, float(obj.flooding)) for obj in node_obj.values())
            global_peak_m3s = max(
                global_peak_m3s, float(flow_rate_to_m3s(total_flooding_native, flow_units))
            )

            observation_due = controller is not None and elapsed >= next_observation
            decision_due = controller is not None and elapsed >= next_decision
            obs: CausalObservation | None = None
            rain_now: np.ndarray | None = None
            if observation_due or decision_due:
                rain_now = observed_rainfall()
                obs = build_observation(elapsed, rain_now)
            if observation_due:
                if hasattr(controller, "observe"):
                    controller.observe(obs)  # type: ignore[attr-defined,union-attr]
                while next_observation <= elapsed:
                    next_observation += observation_update_seconds

            action: ControllerAction | None = None
            if decision_due:
                if obs is None:
                    raise RuntimeError("decision is due without a current causal observation")
                if hasattr(controller, "decide"):
                    action = _normalize_action(
                        controller.decide(  # type: ignore[attr-defined]
                            obs, observation_already_recorded=observation_due
                        )
                    )
                else:
                    action = _normalize_action(controller(obs))
                expected, supplied = set(actuator_ids), set(action.settings)
                if supplied != expected:
                    raise ValueError(
                        f"controller must return every actuator; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}"
                    )
                held_settings = {aid: float(action.settings[aid]) for aid in actuator_ids}
                bad = {aid: v for aid, v in held_settings.items() if not 0.0 <= v <= 1.0}
                if bad:
                    raise ValueError(f"controller settings outside [0,1]: {bad}")
                for aid, value in held_settings.items():
                    link_obj[aid].target_setting = value
                decision_count += 1
                while next_decision <= elapsed:
                    next_decision += control_update_seconds
                decision_fh.write(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed,
                            "datetime": sim.current_time.isoformat(),
                            "source": action.source,
                            "settings": held_settings,
                            "diagnostics": {
                                **dict(action.diagnostics or {}),
                                "target_latch_reasserted_max": target_latch_reasserted_max,
                                "target_latch_reasserted": bool(
                                    target_latch_reasserted_max > 1e-9
                                ),
                            },
                        },
                        sort_keys=True,
                    ) + "\n"
                )
            elif held_settings is not None:
                for aid, value in held_settings.items():
                    link_obj[aid].target_setting = value

            if elapsed >= next_record:
                rain_record = rain_now if rain_now is not None else observed_rainfall()
                phase = "DECISION" if action is not None else "OBSERVATION"
                source = action.source if action is not None else (
                    "HELD" if held_settings is not None else "NATIVE"
                )
                append_record(elapsed, rain_record, phase, source)
                while next_record <= elapsed:
                    next_record += record_stride_seconds

        end_statistics = snapshot_node_statistics(node_obj)
        flow_error = float(sim.flow_routing_error)

    np.savez_compressed(
        compact_path,
        # V2 tensor names/shapes remain backward compatible; only the time grid now includes t=0.
        schema_version=np.asarray("RTC_COMPACT_CLOSED_LOOP_V2_T0_CAUSAL"),
        elapsed_seconds=np.asarray(record_times, dtype=np.int64),
        node_ids=np.asarray(all_nodes),
        state_si=np.stack(state_values).astype(np.float32),
        state_channels=np.asarray(STATE_CHANNELS),
        rainfall_mmhr=np.stack(rainfall_values).astype(np.float32),
        actuator_ids=np.asarray(actuator_ids),
        target_setting=np.stack(target_values).astype(np.float32),
        current_setting=np.stack(current_values).astype(np.float32),
        actuator_flow_m3s=np.stack(flow_values).astype(np.float32),
    )
    write_node_statistics(
        statistics_path,
        end_statistics=end_statistics,
        system_units=system_units,
        flow_units=flow_units,
    )
    if not keep_engine_files:
        report_path.unlink(missing_ok=True)
        engine_output_path.unlink(missing_ok=True)

    metadata = {
        "run_id": run_id,
        # Keep the established metadata contract so baseline/checkpoint readers remain compatible.
        "data_contract": "CLOSED_LOOP_COMPACT_V2",
        "causal_timing_revision": "T0_INCLUDED_V1",
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "controller_present": controller is not None,
        "causal_controller_observation": "sparse_SI_hydraulics+SI_actuator_readback+realised_rainfall_only",
        "initial_observation_elapsed_seconds": 0,
        "sensor_nodes": list(sensor_nodes),
        "actuator_ids": list(actuator_ids),
        "control_start_minutes": int(control_start_minutes),
        "control_update_seconds": int(control_update_seconds),
        "observation_update_seconds": int(observation_update_seconds),
        "record_stride_seconds": int(record_stride_seconds),
        "exact_global_peak": bool(exact_global_peak),
        "global_peak_flood_rate_m3s": float(global_peak_m3s),
        "flow_units": flow_units,
        "system_units": system_units,
        "controller_units": {"depth": "m", "head": "m", "flow": "m3/s", "rainfall": "mm/h"},
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "decisions": decision_count,
        "compact_file": compact_path.name,
        "decision_file": decision_path.name,
        "node_statistics_file": statistics_path.name,
        "raw_csv_saved": bool(save_raw_csv),
        "node_file": node_path.name if save_raw_csv else None,
        "actuator_file": actuator_path.name if save_raw_csv else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ClosedLoopResult(
        metadata_path=str(metadata_path),
        compact_path=str(compact_path),
        decision_path=str(decision_path),
        node_statistics_path=str(statistics_path),
        decisions=decision_count,
        global_peak_flood_rate_m3s=float(global_peak_m3s),
        flow_routing_error_pct=flow_error,
        node_path=str(node_path) if save_raw_csv else "",
        actuator_path=str(actuator_path) if save_raw_csv else "",
    )
