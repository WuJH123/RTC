from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .data_design import canonical_sequence_sha
from .inp import discover_actuators, discover_nodes
from .swmm_data import sha256_file


@dataclass(frozen=True)
class SequenceBranchResult:
    branch_id: str
    metadata_path: str
    node_path: str
    actuator_path: str
    subcatchment_path: str
    flow_routing_error_pct: float


def run_control_sequence_branch(
    *,
    inp_path: str | Path,
    checkpoint_minutes: int,
    settings_sequence: list[dict[str, float]],
    control_block_seconds: int,
    output_dir: str | Path,
    branch_id: str,
    python_intervention_seconds: int = 300,
) -> SequenceBranchResult:
    """Run one fresh same-prefix D3 multi-actuator continuous sequence in PySWMM."""

    try:
        from pyswmm import Links, Nodes, Simulation, Subcatchments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc
    if not settings_sequence:
        raise ValueError("settings_sequence cannot be empty")
    if control_block_seconds <= 0 or python_intervention_seconds <= 0:
        raise ValueError("control block and Python stride must be positive")
    if control_block_seconds % python_intervention_seconds:
        raise ValueError("control_block_seconds must be a multiple of Python stride")
    if (checkpoint_minutes * 60) % python_intervention_seconds:
        raise ValueError("checkpoint must align with Python stride")

    inp = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp)
    node_ids = discover_nodes(inp)
    expected = set(catalog.ids)
    normalized: list[dict[str, float]] = []
    for step, settings in enumerate(settings_sequence):
        supplied = set(settings)
        if supplied != expected:
            raise ValueError(f"sequence step {step} does not specify every actuator")
        values = {aid: float(settings[aid]) for aid in catalog.ids}
        bad = {aid: v for aid, v in values.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(f"sequence step {step} outside [0,1]: {bad}")
        normalized.append(values)

    checkpoint_seconds = int(checkpoint_minutes * 60)
    stop_seconds = checkpoint_seconds + len(normalized) * control_block_seconds
    node_path = out / f"{branch_id}.nodes.csv.gz"
    actuator_path = out / f"{branch_id}.actuators.csv.gz"
    subcatchment_path = out / f"{branch_id}.subcatchments.csv.gz"
    metadata_path = out / f"{branch_id}.json"

    with Simulation(str(inp)) as sim, gzip.open(node_path, "wt", encoding="utf-8", newline="") as node_fh, gzip.open(
        actuator_path, "wt", encoding="utf-8", newline=""
    ) as act_fh, gzip.open(subcatchment_path, "wt", encoding="utf-8", newline="") as sub_fh:
        sim.step_advance(python_intervention_seconds)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links, nodes, subcatchments = Links(sim), Nodes(sim), Subcatchments(sim)
        link_obj = {aid: links[aid] for aid in catalog.ids}
        node_obj = {nid: nodes[nid] for nid in node_ids}
        sub_obj = {obj.subcatchmentid: obj for obj in subcatchments}
        sub_connection = {sid: tuple(obj.connection) for sid, obj in sub_obj.items()}
        node_writer, act_writer, sub_writer = csv.writer(node_fh), csv.writer(act_fh), csv.writer(sub_fh)
        node_writer.writerow(["elapsed_seconds", "datetime", "phase", "node_id", "depth", "head", "flooding", "volume"])
        act_writer.writerow([
            "elapsed_seconds", "datetime", "phase", "sequence_step", "actuator_id",
            "requested_setting", "target_setting", "current_setting", "flow",
        ])
        sub_writer.writerow([
            "elapsed_seconds", "datetime", "phase", "subcatchment_id",
            "outlet_connection_type", "outlet_id", "rainfall", "runoff",
        ])
        checkpoint_recorded = False
        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            if elapsed < checkpoint_seconds:
                continue
            if elapsed > stop_seconds:
                sim.terminate_simulation()
                break
            relative = elapsed - checkpoint_seconds
            is_checkpoint = elapsed == checkpoint_seconds
            is_boundary = relative % control_block_seconds == 0
            phase = "PRE_ACTION_CHECKPOINT" if is_checkpoint else ("BLOCK_BOUNDARY" if is_boundary else "POST_ACTION")
            action_step = min(relative // control_block_seconds, len(normalized) - 1)

            for nid, obj in node_obj.items():
                node_writer.writerow([elapsed, sim.current_time.isoformat(), phase, nid, obj.depth, obj.head, obj.flooding, obj.volume])
            for aid, obj in link_obj.items():
                requested = "" if elapsed >= stop_seconds else normalized[action_step][aid]
                act_writer.writerow([
                    elapsed, sim.current_time.isoformat(), phase, action_step, aid,
                    requested, obj.target_setting, obj.current_setting, obj.flow,
                ])
            for sid, obj in sub_obj.items():
                connection_type, outlet_id = sub_connection[sid]
                sub_writer.writerow([
                    elapsed, sim.current_time.isoformat(), phase, sid,
                    connection_type, outlet_id, obj.rainfall, obj.runoff,
                ])

            if is_checkpoint:
                checkpoint_recorded = True
            if elapsed < stop_seconds:
                for aid, value in normalized[action_step].items():
                    link_obj[aid].target_setting = value

        if not checkpoint_recorded:
            raise RuntimeError("simulation did not reach the requested checkpoint")
        flow_error = float(sim.flow_routing_error)

    metadata = {
        "branch_id": branch_id,
        "data_role": "D3_MULTI_ACTUATOR_SEQUENCE",
        "sequence_sha256": canonical_sequence_sha(normalized),
        "settings_sequence": normalized,
        "control_block_seconds": int(control_block_seconds),
        "horizon_steps": len(normalized),
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "checkpoint_minutes": int(checkpoint_minutes),
        "python_intervention_seconds": int(python_intervention_seconds),
        "prefix_policy": "native_swmm_no_python_override_until_checkpoint",
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
    return SequenceBranchResult(
        branch_id=branch_id,
        metadata_path=str(metadata_path),
        node_path=str(node_path),
        actuator_path=str(actuator_path),
        subcatchment_path=str(subcatchment_path),
        flow_routing_error_pct=flow_error,
    )
