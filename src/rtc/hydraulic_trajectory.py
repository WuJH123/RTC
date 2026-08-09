from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .inp import discover_actuators, discover_nodes
from .swmm_data import sha256_file


@dataclass(frozen=True)
class HydraulicTrajectoryResult:
    metadata_path: str
    node_path: str
    actuator_path: str
    subcatchment_path: str
    flow_routing_error_pct: float


def run_hydraulic_trajectory(
    *,
    inp_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    record_stride_seconds: int = 300,
) -> HydraulicTrajectoryResult:
    """Generate D0/D1 full-event truth under the INP's own frozen operating semantics.

    No Python actuator writes occur. Use the frozen INP for native-rule trajectories or a
    separately hashed ``PASSIVE_NO_RTC`` INP produced by ``write_passive_no_rtc_inp``.
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
    node_ids = discover_nodes(inp)
    node_path = out / f"{run_id}.nodes.csv.gz"
    actuator_path = out / f"{run_id}.actuators.csv.gz"
    subcatchment_path = out / f"{run_id}.subcatchments.csv.gz"
    metadata_path = out / f"{run_id}.json"

    with Simulation(str(inp)) as sim, gzip.open(node_path, "wt", encoding="utf-8", newline="") as node_fh, gzip.open(
        actuator_path, "wt", encoding="utf-8", newline=""
    ) as act_fh, gzip.open(subcatchment_path, "wt", encoding="utf-8", newline="") as sub_fh:
        sim.step_advance(record_stride_seconds)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links, nodes, subcatchments = Links(sim), Nodes(sim), Subcatchments(sim)
        link_obj = {aid: links[aid] for aid in catalog.ids}
        node_obj = {nid: nodes[nid] for nid in node_ids}
        sub_obj = {obj.subcatchmentid: obj for obj in subcatchments}
        sub_connection = {sid: tuple(obj.connection) for sid, obj in sub_obj.items()}
        node_writer, act_writer, sub_writer = csv.writer(node_fh), csv.writer(act_fh), csv.writer(sub_fh)
        node_writer.writerow(["elapsed_seconds", "datetime", "node_id", "depth", "head", "flooding", "volume"])
        act_writer.writerow(["elapsed_seconds", "datetime", "actuator_id", "target_setting", "current_setting", "flow"])
        sub_writer.writerow([
            "elapsed_seconds", "datetime", "subcatchment_id", "outlet_connection_type", "outlet_id", "rainfall", "runoff"
        ])
        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            for nid, obj in node_obj.items():
                node_writer.writerow([elapsed, sim.current_time.isoformat(), nid, obj.depth, obj.head, obj.flooding, obj.volume])
            for aid, obj in link_obj.items():
                act_writer.writerow([elapsed, sim.current_time.isoformat(), aid, obj.target_setting, obj.current_setting, obj.flow])
            for sid, obj in sub_obj.items():
                connection_type, outlet_id = sub_connection[sid]
                sub_writer.writerow([
                    elapsed, sim.current_time.isoformat(), sid, connection_type, outlet_id, obj.rainfall, obj.runoff
                ])
        flow_error = float(sim.flow_routing_error)

    metadata = {
        "run_id": run_id,
        "data_role": "D0_D1_HYDRAULIC_TRAJECTORY",
        "python_actuator_writes": False,
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "record_stride_seconds": int(record_stride_seconds),
        "flow_units": flow_units,
        "system_units": system_units,
        "rainfall_units": "mm/hr_if_SI_else_in/hr",
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "node_file": node_path.name,
        "actuator_file": actuator_path.name,
        "subcatchment_file": subcatchment_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HydraulicTrajectoryResult(
        metadata_path=str(metadata_path),
        node_path=str(node_path),
        actuator_path=str(actuator_path),
        subcatchment_path=str(subcatchment_path),
        flow_routing_error_pct=flow_error,
    )
