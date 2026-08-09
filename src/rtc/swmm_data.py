from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .data_design import canonical_action_sha
from .inp import discover_actuators, discover_nodes
from .swmm_stats import snapshot_node_statistics, write_node_statistics


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    node_path: str
    actuator_path: str
    subcatchment_path: str
    node_statistics_path: str
    metadata_path: str
    flow_routing_error_pct: float


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_independent_control_branch(
    *,
    inp_path: str | Path,
    checkpoint_minutes: int,
    horizon_minutes: int,
    candidate_settings: dict[str, float],
    output_dir: str | Path,
    branch_id: str,
    python_intervention_seconds: int = 300,
) -> BranchResult:
    """Run one authoritative same-prefix single-action SWMM counterfactual branch."""

    try:
        from pyswmm import Links, Nodes, Simulation, Subcatchments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc

    inp_path = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp_path)
    node_ids = discover_nodes(inp_path)
    expected = set(catalog.ids)
    supplied = set(candidate_settings)
    if supplied != expected:
        raise ValueError(
            f"candidate must specify every actuator; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}"
        )
    for aid, value in candidate_settings.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{aid} setting outside [0,1]: {value}")
    if checkpoint_minutes <= 0 or horizon_minutes <= 0:
        raise ValueError("checkpoint and horizon must be positive")
    if python_intervention_seconds <= 0 or (checkpoint_minutes * 60) % python_intervention_seconds:
        raise ValueError("checkpoint must align with a positive Python intervention stride")

    node_path = out / f"{branch_id}.nodes.csv.gz"
    actuator_path = out / f"{branch_id}.actuators.csv.gz"
    subcatchment_path = out / f"{branch_id}.subcatchments.csv.gz"
    statistics_path = out / f"{branch_id}.node_statistics.csv.gz"
    metadata_path = out / f"{branch_id}.json"
    checkpoint_seconds = checkpoint_minutes * 60
    stop_seconds = (checkpoint_minutes + horizon_minutes) * 60
    start_statistics: dict[str, dict[str, float]] | None = None

    with Simulation(str(inp_path)) as sim, gzip.open(
        node_path, "wt", encoding="utf-8", newline=""
    ) as node_fh, gzip.open(actuator_path, "wt", encoding="utf-8", newline="") as act_fh, gzip.open(
        subcatchment_path, "wt", encoding="utf-8", newline=""
    ) as sub_fh:
        import csv

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
        node_writer.writerow(
            ["elapsed_seconds", "datetime", "phase", "node_id", "depth", "head", "flooding", "volume"]
        )
        act_writer.writerow(
            ["elapsed_seconds", "datetime", "phase", "actuator_id", "requested_setting", "target_setting", "current_setting", "flow"]
        )
        sub_writer.writerow(
            ["elapsed_seconds", "datetime", "phase", "subcatchment_id", "outlet_connection_type", "outlet_id", "rainfall", "runoff"]
        )
        checkpoint_recorded = False
        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            if elapsed < checkpoint_seconds:
                continue
            if elapsed > stop_seconds:
                sim.terminate_simulation()
                break
            phase = "PRE_ACTION_CHECKPOINT" if elapsed == checkpoint_seconds else "POST_ACTION"
            for nid, obj in node_obj.items():
                node_writer.writerow(
                    [elapsed, sim.current_time.isoformat(), phase, nid, obj.depth, obj.head, obj.flooding, obj.volume]
                )
            for aid, obj in link_obj.items():
                act_writer.writerow(
                    [
                        elapsed,
                        sim.current_time.isoformat(),
                        phase,
                        aid,
                        "" if phase == "PRE_ACTION_CHECKPOINT" else candidate_settings[aid],
                        obj.target_setting,
                        obj.current_setting,
                        obj.flow,
                    ]
                )
            for sid, obj in sub_obj.items():
                connection_type, outlet_id = sub_connection[sid]
                sub_writer.writerow(
                    [elapsed, sim.current_time.isoformat(), phase, sid, connection_type, outlet_id, obj.rainfall, obj.runoff]
                )
            if elapsed == checkpoint_seconds and not checkpoint_recorded:
                checkpoint_recorded = True
                start_statistics = snapshot_node_statistics(node_obj)
            for aid, value in candidate_settings.items():
                link_obj[aid].target_setting = float(value)

        if not checkpoint_recorded or start_statistics is None:
            raise RuntimeError("simulation did not reach the requested checkpoint")
        end_statistics = snapshot_node_statistics(node_obj)
        flow_error = float(sim.flow_routing_error)

    write_node_statistics(
        statistics_path,
        start_statistics=start_statistics,
        end_statistics=end_statistics,
        system_units=system_units,
        flow_units=flow_units,
    )
    metadata = {
        "branch_id": branch_id,
        "candidate_action_sha256": canonical_action_sha(candidate_settings),
        "inp_path": str(inp_path.resolve()),
        "inp_sha256": sha256_file(inp_path),
        "checkpoint_minutes": checkpoint_minutes,
        "horizon_minutes": horizon_minutes,
        "python_intervention_seconds": python_intervention_seconds,
        "actuator_count": len(catalog.actuators),
        "subcatchment_count": len(sub_obj),
        "candidate_settings": {k: float(v) for k, v in sorted(candidate_settings.items())},
        "prefix_policy": "native_swmm_no_python_override_until_checkpoint",
        "flow_units": flow_units,
        "system_units": system_units,
        "rainfall_units": "mm/hr_if_SI_else_in/hr",
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "node_file": node_path.name,
        "actuator_file": actuator_path.name,
        "subcatchment_file": subcatchment_path.name,
        "node_statistics_file": statistics_path.name,
        "post_action_flood_volume_truth": "SWMM_node_statistics_delta",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BranchResult(
        branch_id=branch_id,
        node_path=str(node_path),
        actuator_path=str(actuator_path),
        subcatchment_path=str(subcatchment_path),
        node_statistics_path=str(statistics_path),
        metadata_path=str(metadata_path),
        flow_routing_error_pct=flow_error,
    )
