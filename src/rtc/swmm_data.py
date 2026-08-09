from __future__ import annotations

import csv
import gzip
import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_design import canonical_action_sha
from .forcing import current_node_rainfall_mmhr, resolve_subcatchment_outlets
from .inp import discover_actuators, discover_nodes
from .inp_runtime import assert_native_controls_disabled
from .replay_prefix import load_checkpoint_reference, verify_replayed_checkpoint
from .swmm_stats import snapshot_node_statistics, write_node_statistics
from .units import flow_rate_to_m3s, length_to_m, volume_to_m3


STATE_CHANNELS = (
    "depth_m",
    "head_m",
    "flooding_m3s",
    "volume_m3",
    "total_inflow_m3s",
    "total_outflow_m3s",
)


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    compact_path: str
    node_statistics_path: str
    metadata_path: str
    flow_routing_error_pct: float
    node_path: str = ""
    actuator_path: str = ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _node_state_si(node_obj: dict[str, object], node_ids: tuple[str, ...], *, system_units: str, flow_units: str) -> np.ndarray:
    depth = length_to_m(np.array([node_obj[n].depth for n in node_ids], dtype=float), system_units)
    head = length_to_m(np.array([node_obj[n].head for n in node_ids], dtype=float), system_units)
    flooding = flow_rate_to_m3s(np.array([node_obj[n].flooding for n in node_ids], dtype=float), flow_units)
    volume = volume_to_m3(np.array([node_obj[n].volume for n in node_ids], dtype=float), system_units)
    inflow = flow_rate_to_m3s(np.array([node_obj[n].total_inflow for n in node_ids], dtype=float), flow_units)
    outflow = flow_rate_to_m3s(np.array([node_obj[n].total_outflow for n in node_ids], dtype=float), flow_units)
    return np.stack([depth, head, flooding, volume, inflow, outflow], axis=-1).astype(np.float32)


def run_independent_control_branch(
    *,
    inp_path: str | Path,
    checkpoint_minutes: int,
    horizon_minutes: int,
    candidate_settings: dict[str, float],
    output_dir: str | Path,
    branch_id: str,
    python_intervention_seconds: int = 300,
    reference_trajectory_metadata_path: str | Path | None = None,
    replay_state_atol: float = 1e-5,
    replay_setting_atol: float = 1e-7,
    save_raw_csv: bool = False,
    keep_engine_files: bool = False,
) -> BranchResult:
    """Run one exact same-prefix D2 counterfactual on a controls-disabled INP.

    Formal D2 supplies the saved No-control trajectory used to select the checkpoint. Before
    the candidate is written, the full six-channel hydraulic state and all actuator readbacks
    are compared to that saved checkpoint. Thus an engine/runtime/prefix drift cannot silently
    masquerade as an action effect.
    """

    try:
        from pyswmm import Links, Nodes, Simulation, Subcatchments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc

    inp_path = Path(inp_path)
    assert_native_controls_disabled(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = discover_actuators(inp_path)
    node_ids = tuple(discover_nodes(inp_path))
    actuator_ids = tuple(catalog.ids)
    expected = set(actuator_ids)
    supplied = set(candidate_settings)
    if supplied != expected:
        raise ValueError(
            f"candidate must specify every actuator; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}"
        )
    for aid, value in candidate_settings.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{aid} setting outside [0,1]: {value}")
    if checkpoint_minutes <= 0 or horizon_minutes <= 0 or python_intervention_seconds <= 0:
        raise ValueError("checkpoint, horizon and intervention stride must be positive")
    checkpoint_seconds = checkpoint_minutes * 60
    horizon_seconds = horizon_minutes * 60
    if checkpoint_seconds % python_intervention_seconds or horizon_seconds % python_intervention_seconds:
        raise ValueError("checkpoint and horizon must align with Python intervention stride")
    stop_seconds = checkpoint_seconds + horizon_seconds
    reference = (
        load_checkpoint_reference(
            reference_trajectory_metadata_path, elapsed_seconds=checkpoint_seconds
        )
        if reference_trajectory_metadata_path is not None
        else None
    )

    compact_path = out / f"{branch_id}.compact.npz"
    statistics_path = out / f"{branch_id}.node_statistics.csv.gz"
    metadata_path = out / f"{branch_id}.json"
    node_path = out / f"{branch_id}.nodes.csv.gz"
    actuator_path = out / f"{branch_id}.actuators.csv.gz"
    report_path = out / f"{branch_id}.rpt"
    engine_output_path = out / f"{branch_id}.out"

    elapsed_values: list[int] = []
    state_values: list[np.ndarray] = []
    rainfall_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    flow_values: list[np.ndarray] = []
    start_statistics: dict[str, dict[str, float]] | None = None
    end_statistics: dict[str, dict[str, float]] | None = None
    prefix_verification: dict[str, object] | None = None

    with ExitStack() as stack:
        sim = stack.enter_context(
            Simulation(str(inp_path), reportfile=str(report_path), outputfile=str(engine_output_path))
        )
        node_fh = act_fh = None
        node_writer = act_writer = None
        if save_raw_csv:
            node_fh = stack.enter_context(gzip.open(node_path, "wt", encoding="utf-8", newline=""))
            act_fh = stack.enter_context(gzip.open(actuator_path, "wt", encoding="utf-8", newline=""))
            node_writer, act_writer = csv.writer(node_fh), csv.writer(act_fh)
            node_writer.writerow(["elapsed_seconds", "phase", "node_id", *STATE_CHANNELS])
            act_writer.writerow([
                "elapsed_seconds", "phase", "actuator_id", "requested_setting",
                "target_setting", "current_setting", "flow_m3s",
            ])

        sim.step_advance(python_intervention_seconds)
        flow_units = str(sim.flow_units)
        system_units = str(sim.system_units)
        engine_version = str(sim.engine_version)
        links, nodes, subcatchments = Links(sim), Nodes(sim), Subcatchments(sim)
        link_obj = {aid: links[aid] for aid in actuator_ids}
        node_obj = {nid: nodes[nid] for nid in node_ids}
        sub_obj = {obj.subcatchmentid: obj for obj in subcatchments}
        connection = {sid: tuple(obj.connection) for sid, obj in sub_obj.items()}
        resolved_outlets = resolve_subcatchment_outlets(connection, node_ids)
        checkpoint_recorded = False

        for _ in sim:
            elapsed = int((sim.current_time - sim.start_time).total_seconds())
            if elapsed < checkpoint_seconds:
                continue
            if elapsed > stop_seconds:
                raise RuntimeError("SWMM callback stepped beyond an aligned D2 horizon endpoint")

            phase = "PRE_ACTION_CHECKPOINT" if elapsed == checkpoint_seconds else "POST_ACTION"
            state = _node_state_si(node_obj, node_ids, system_units=system_units, flow_units=flow_units)
            rainfall = current_node_rainfall_mmhr(
                sub_obj, resolved_outlets, node_ids, system_units
            )[:, None]
            target = np.array([link_obj[a].target_setting for a in actuator_ids], dtype=np.float32)
            current = np.array([link_obj[a].current_setting for a in actuator_ids], dtype=np.float32)
            flow = flow_rate_to_m3s(
                np.array([link_obj[a].flow for a in actuator_ids], dtype=float), flow_units
            ).astype(np.float32)
            elapsed_values.append(elapsed)
            state_values.append(state)
            rainfall_values.append(rainfall.astype(np.float32))
            target_values.append(target)
            current_values.append(current)
            flow_values.append(flow)

            if save_raw_csv:
                assert node_writer is not None and act_writer is not None
                for ni, nid in enumerate(node_ids):
                    node_writer.writerow([elapsed, phase, nid, *state[ni].tolist()])
                for ai, aid in enumerate(actuator_ids):
                    act_writer.writerow([
                        elapsed, phase, aid,
                        "" if phase == "PRE_ACTION_CHECKPOINT" else candidate_settings[aid],
                        target[ai], current[ai], flow[ai],
                    ])

            if elapsed == checkpoint_seconds and not checkpoint_recorded:
                checkpoint_recorded = True
                if reference is not None:
                    prefix_verification = dict(
                        verify_replayed_checkpoint(
                            reference,
                            elapsed_seconds=elapsed,
                            node_ids=node_ids,
                            actuator_ids=actuator_ids,
                            state_si=state,
                            current_setting=current,
                            swmm_engine_version=engine_version,
                            state_atol=replay_state_atol,
                            setting_atol=replay_setting_atol,
                        )
                    )
                start_statistics = snapshot_node_statistics(node_obj)

            if elapsed == stop_seconds:
                end_statistics = snapshot_node_statistics(node_obj)
                sim.terminate_simulation()
                break

            for aid, value in candidate_settings.items():
                link_obj[aid].target_setting = float(value)

        if not checkpoint_recorded or start_statistics is None or end_statistics is None:
            raise RuntimeError("simulation did not produce exact checkpoint/end statistics")
        flow_error = float(sim.flow_routing_error)

    np.savez_compressed(
        compact_path,
        schema_version=np.asarray("RTC_COMPACT_BRANCH_V3_PREFIX_VERIFIED"),
        elapsed_seconds=np.asarray(elapsed_values, dtype=np.int64),
        node_ids=np.asarray(node_ids),
        state_si=np.stack(state_values).astype(np.float32),
        state_channels=np.asarray(STATE_CHANNELS),
        rainfall_mmhr=np.stack(rainfall_values).astype(np.float32),
        actuator_ids=np.asarray(actuator_ids),
        target_setting=np.stack(target_values).astype(np.float32),
        current_setting=np.stack(current_values).astype(np.float32),
        actuator_flow_m3s=np.stack(flow_values).astype(np.float32),
        candidate_setting=np.asarray([candidate_settings[a] for a in actuator_ids], dtype=np.float32),
    )
    write_node_statistics(
        statistics_path,
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
        "data_contract": "D2_CONTROLS_DISABLED_COMPACT_V3_PREFIX_VERIFIED",
        "candidate_action_sha256": canonical_action_sha(candidate_settings),
        "inp_path": str(inp_path.resolve()),
        "inp_sha256": sha256_file(inp_path),
        "native_controls_enabled": False,
        "checkpoint_minutes": checkpoint_minutes,
        "horizon_minutes": horizon_minutes,
        "python_intervention_seconds": python_intervention_seconds,
        "actuator_count": len(actuator_ids),
        "node_count": len(node_ids),
        "candidate_settings": {k: float(v) for k, v in sorted(candidate_settings.items())},
        "flow_units": flow_units,
        "system_units": system_units,
        "controller_tensor_units": "SI",
        "swmm_engine_version": engine_version,
        "flow_routing_error_pct": flow_error,
        "compact_file": compact_path.name,
        "node_statistics_file": statistics_path.name,
        "post_action_flood_volume_truth": "SWMM_node_statistics_delta_exact_horizon",
        "same_prefix_required": reference is not None,
        "same_prefix_verification": prefix_verification,
        "raw_csv_saved": bool(save_raw_csv),
        "node_file": node_path.name if save_raw_csv else None,
        "actuator_file": actuator_path.name if save_raw_csv else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BranchResult(
        branch_id=branch_id,
        compact_path=str(compact_path),
        node_statistics_path=str(statistics_path),
        metadata_path=str(metadata_path),
        flow_routing_error_pct=flow_error,
        node_path=str(node_path) if save_raw_csv else "",
        actuator_path=str(actuator_path) if save_raw_csv else "",
    )
