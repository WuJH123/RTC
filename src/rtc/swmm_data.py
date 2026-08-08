from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .inp import discover_actuators, discover_nodes


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    node_path: str
    actuator_path: str
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
    """Run one authoritative same-prefix SWMM counterfactual branch.

    Scientific contract:
    - a new Simulation is created for every branch;
    - no Python controls are applied before the checkpoint, so all branches replay the
      exact same native prefix for a fixed event INP;
    - the pre-action checkpoint state is recorded before candidate settings are written;
    - requested target setting, actual current setting, facility flow and node hydraulics
      are recorded after the checkpoint;
    - ``step_advance`` is only an intervention/output stride. It does not alter SWMM's
      internal routing step.

    This intentionally favours causal correctness over speed. A verified hot-start cache
    can be introduced later without changing the data contract.
    """

    try:
        from pyswmm import Links, Nodes, Simulation
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc

    inp_path = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp_path)
    node_ids = discover_nodes(inp_path)
    expected = set(catalog.ids)
    supplied = set(candidate_settings)
    if supplied != expected:
        missing, extra = sorted(expected - supplied), sorted(supplied - expected)
        raise ValueError(f"candidate must specify every actuator; missing={missing}, extra={extra}")
    for aid, value in candidate_settings.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{aid} setting outside [0,1]: {value}")
    if checkpoint_minutes <= 0 or horizon_minutes <= 0:
        raise ValueError("checkpoint and horizon must be positive")
    if (checkpoint_minutes * 60) % python_intervention_seconds:
        raise ValueError("checkpoint must align with the Python intervention stride")

    node_path = out / f"{branch_id}.nodes.csv.gz"
    actuator_path = out / f"{branch_id}.actuators.csv.gz"
    metadata_path = out / f"{branch_id}.json"
    checkpoint_seconds = checkpoint_minutes * 60
    stop_seconds = (checkpoint_minutes + horizon_minutes) * 60

    with Simulation(str(inp_path)) as sim, gzip.open(
        node_path, "wt", encoding="utf-8", newline=""
    ) as node_fh, gzip.open(actuator_path, "wt", encoding="utf-8", newline="") as act_fh:
        import csv

        sim.step_advance(python_intervention_seconds)
        links = Links(sim)
        nodes = Nodes(sim)
        link_obj = {aid: links[aid] for aid in catalog.ids}
        node_obj = {nid: nodes[nid] for nid in node_ids}
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
            ]
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
                        candidate_settings[aid] if elapsed >= checkpoint_seconds else "",
                        obj.target_setting,
                        obj.current_setting,
                        obj.flow,
                    ]
                )

            if elapsed == checkpoint_seconds and not checkpoint_recorded:
                checkpoint_recorded = True
            # Apply after recording the exact pre-action checkpoint. Reapply at every
            # intervention so Python controls have priority over native rules thereafter.
            for aid, value in candidate_settings.items():
                link_obj[aid].target_setting = float(value)

        flow_error = float(sim.flow_routing_error)

    metadata = {
        "branch_id": branch_id,
        "inp_path": str(inp_path.resolve()),
        "inp_sha256": sha256_file(inp_path),
        "checkpoint_minutes": checkpoint_minutes,
        "horizon_minutes": horizon_minutes,
        "python_intervention_seconds": python_intervention_seconds,
        "actuator_count": len(catalog.actuators),
        "candidate_settings": {k: float(v) for k, v in sorted(candidate_settings.items())},
        "prefix_policy": "native_swmm_no_python_override_until_checkpoint",
        "flow_routing_error_pct": flow_error,
        "node_file": node_path.name,
        "actuator_file": actuator_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BranchResult(
        branch_id=branch_id,
        node_path=str(node_path),
        actuator_path=str(actuator_path),
        metadata_path=str(metadata_path),
        flow_routing_error_pct=flow_error,
    )
