from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .forcing import current_node_rainfall_mmhr, resolve_subcatchment_outlets
from .inp import discover_actuators, discover_nodes
from .inp_runtime import section_has_payload
from .swmm_data import STATE_CHANNELS, _node_state_si, sha256_file
from .swmm_stats import snapshot_node_statistics, write_node_statistics
from .units import flow_rate_to_m3s


@dataclass(frozen=True)
class HydraulicTrajectoryResult:
    metadata_path: str
    compact_path: str
    node_statistics_path: str
    flow_routing_error_pct: float


def run_hydraulic_trajectory(
    *,
    inp_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    record_stride_seconds: int = 300,
    keep_engine_files: bool = False,
) -> HydraulicTrajectoryResult:
    """Generate one compact full-event D0/D1 authoritative trajectory.

    The same function is valid for Internal-RTC (original INP) and No-control (controls-
    disabled runtime INP); policy semantics are recorded in metadata. Only node-level
    causal rainfall is retained for model input. Per-subcatchment runoff/rainfall CSVs are
    intentionally not persisted by default.
    """

    try:
        from pyswmm import Links, Nodes, Simulation, Subcatchments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc
    if record_stride_seconds <= 0:
        raise ValueError("record_stride_seconds must be positive")

    inp = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp)
    node_ids = tuple(discover_nodes(inp))
    actuator_ids = tuple(catalog.ids)
    compact_path = out / f"{run_id}.compact.npz"
    statistics_path = out / f"{run_id}.node_statistics.csv.gz"
    metadata_path = out / f"{run_id}.json"
    report_path = out / f"{run_id}.rpt"
    engine_output_path = out / f"{run_id}.out"

    elapsed_values: list[int] = []
    state_values: list[np.ndarray] = []
    rainfall_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    flow_values: list[np.ndarray] = []

    with Simulation(str(inp), reportfile=str(report_path), outputfile=str(engine_output_path)) as sim:
        sim.step_advance(record_stride_seconds)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links, nodes, subcatchments = Links(sim), Nodes(sim), Subcatchments(sim)
        link_obj = {aid: links[aid] for aid in actuator_ids}
        node_obj = {nid: nodes[nid] for nid in node_ids}
        sub_obj = {obj.subcatchmentid: obj for obj in subcatchments}
        connection = {sid: tuple(obj.connection) for sid, obj in sub_obj.items()}
        resolved = resolve_subcatchment_outlets(connection, node_ids)

        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            elapsed_values.append(elapsed)
            state_values.append(
                _node_state_si(node_obj, node_ids, system_units=system_units, flow_units=flow_units)
            )
            rainfall_values.append(
                current_node_rainfall_mmhr(sub_obj, resolved, node_ids, system_units)[:, None]
            )
            target_values.append(np.array([link_obj[a].target_setting for a in actuator_ids], dtype=np.float32))
            current_values.append(np.array([link_obj[a].current_setting for a in actuator_ids], dtype=np.float32))
            flow_values.append(
                flow_rate_to_m3s(
                    np.array([link_obj[a].flow for a in actuator_ids], dtype=float), flow_units
                ).astype(np.float32)
            )
        end_statistics = snapshot_node_statistics(node_obj)
        flow_error = float(sim.flow_routing_error)

    np.savez_compressed(
        compact_path,
        schema_version=np.asarray("RTC_COMPACT_TRAJECTORY_V2"),
        elapsed_seconds=np.asarray(elapsed_values, dtype=np.int64),
        node_ids=np.asarray(node_ids),
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
        "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
        "data_role": "D0_D1_HYDRAULIC_TRAJECTORY",
        "python_actuator_writes": False,
        "native_controls_enabled": bool(section_has_payload(inp, "CONTROLS")),
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "record_stride_seconds": int(record_stride_seconds),
        "flow_units": flow_units,
        "system_units": system_units,
        "controller_tensor_units": "SI",
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "compact_file": compact_path.name,
        "node_statistics_file": statistics_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HydraulicTrajectoryResult(
        metadata_path=str(metadata_path),
        compact_path=str(compact_path),
        node_statistics_path=str(statistics_path),
        flow_routing_error_pct=flow_error,
    )
