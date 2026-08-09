from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_design import canonical_sequence_sha
from .forcing import current_node_rainfall_mmhr, resolve_subcatchment_outlets
from .inp import discover_actuators, discover_nodes
from .inp_runtime import assert_native_controls_disabled
from .swmm_data import STATE_CHANNELS, _node_state_si, sha256_file
from .swmm_stats import snapshot_node_statistics, write_node_statistics
from .units import flow_rate_to_m3s


@dataclass(frozen=True)
class SequenceBranchResult:
    branch_id: str
    metadata_path: str
    compact_path: str
    node_statistics_path: str
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
    keep_engine_files: bool = False,
) -> SequenceBranchResult:
    """Run one exact D3 multi-actuator sequence on a controls-disabled physical base."""

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
    assert_native_controls_disabled(inp)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp)
    node_ids = tuple(discover_nodes(inp))
    actuator_ids = tuple(catalog.ids)
    expected = set(actuator_ids)
    normalized: list[dict[str, float]] = []
    for step, settings in enumerate(settings_sequence):
        if set(settings) != expected:
            raise ValueError(f"sequence step {step} does not specify every actuator")
        values = {aid: float(settings[aid]) for aid in actuator_ids}
        bad = {aid: v for aid, v in values.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(f"sequence step {step} outside [0,1]: {bad}")
        normalized.append(values)

    checkpoint_seconds = int(checkpoint_minutes * 60)
    stop_seconds = checkpoint_seconds + len(normalized) * control_block_seconds
    compact_path = out / f"{branch_id}.compact.npz"
    stats_path = out / f"{branch_id}.node_statistics.csv.gz"
    metadata_path = out / f"{branch_id}.json"
    report_path = out / f"{branch_id}.rpt"
    engine_output_path = out / f"{branch_id}.out"

    elapsed_values: list[int] = []
    state_values: list[np.ndarray] = []
    rainfall_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    flow_values: list[np.ndarray] = []
    commanded_values: list[np.ndarray] = []
    start_statistics: dict[str, dict[str, float]] | None = None
    end_statistics: dict[str, dict[str, float]] | None = None

    with Simulation(str(inp), reportfile=str(report_path), outputfile=str(engine_output_path)) as sim:
        sim.step_advance(python_intervention_seconds)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links, nodes, subcatchments = Links(sim), Nodes(sim), Subcatchments(sim)
        link_obj = {aid: links[aid] for aid in actuator_ids}
        node_obj = {nid: nodes[nid] for nid in node_ids}
        sub_obj = {obj.subcatchmentid: obj for obj in subcatchments}
        connection = {sid: tuple(obj.connection) for sid, obj in sub_obj.items()}
        resolved = resolve_subcatchment_outlets(connection, node_ids)
        checkpoint_recorded = False

        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            if elapsed < checkpoint_seconds:
                continue
            if elapsed > stop_seconds:
                raise RuntimeError("SWMM callback stepped beyond aligned D3 horizon endpoint")
            relative = elapsed - checkpoint_seconds
            action_step = min(max(relative // control_block_seconds, 0), len(normalized) - 1)
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
            commanded_values.append(
                np.asarray([normalized[action_step][a] for a in actuator_ids], dtype=np.float32)
            )
            elapsed_values.append(elapsed)

            if elapsed == checkpoint_seconds and not checkpoint_recorded:
                checkpoint_recorded = True
                start_statistics = snapshot_node_statistics(node_obj)

            if elapsed == stop_seconds:
                end_statistics = snapshot_node_statistics(node_obj)
                sim.terminate_simulation()
                break

            # Action k governs [t_k, t_{k+1}); record state before the write, then apply.
            for aid, value in normalized[action_step].items():
                link_obj[aid].target_setting = value

        if not checkpoint_recorded or start_statistics is None or end_statistics is None:
            raise RuntimeError("simulation did not produce exact D3 checkpoint/end statistics")
        flow_error = float(sim.flow_routing_error)

    np.savez_compressed(
        compact_path,
        schema_version=np.asarray("RTC_COMPACT_BRANCH_V2"),
        elapsed_seconds=np.asarray(elapsed_values, dtype=np.int64),
        node_ids=np.asarray(node_ids),
        state_si=np.stack(state_values).astype(np.float32),
        state_channels=np.asarray(STATE_CHANNELS),
        rainfall_mmhr=np.stack(rainfall_values).astype(np.float32),
        actuator_ids=np.asarray(actuator_ids),
        target_setting=np.stack(target_values).astype(np.float32),
        current_setting=np.stack(current_values).astype(np.float32),
        actuator_flow_m3s=np.stack(flow_values).astype(np.float32),
        commanded_setting=np.stack(commanded_values).astype(np.float32),
    )
    write_node_statistics(
        stats_path,
        start_statistics=start_statistics,
        end_statistics=end_statistics,
        system_units=system_units,
        flow_units=flow_units,
    )
    if not keep_engine_files:
        report_path.unlink(missing_ok=True)
        engine_output_path.unlink(missing_ok=True)

    metadata = {
        "branch_id": branch_id,
        "data_contract": "D3_CONTROLS_DISABLED_COMPACT_V2",
        "data_role": "D3_MULTI_ACTUATOR_SEQUENCE",
        "sequence_sha256": canonical_sequence_sha(normalized),
        "settings_sequence": normalized,
        "control_block_seconds": int(control_block_seconds),
        "horizon_steps": len(normalized),
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "native_controls_enabled": False,
        "checkpoint_minutes": int(checkpoint_minutes),
        "python_intervention_seconds": int(python_intervention_seconds),
        "flow_units": flow_units,
        "system_units": system_units,
        "controller_tensor_units": "SI",
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "compact_file": compact_path.name,
        "node_statistics_file": stats_path.name,
        "post_action_flood_volume_truth": "SWMM_node_statistics_delta_exact_horizon",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SequenceBranchResult(
        branch_id=branch_id,
        metadata_path=str(metadata_path),
        compact_path=str(compact_path),
        node_statistics_path=str(stats_path),
        flow_routing_error_pct=flow_error,
    )
