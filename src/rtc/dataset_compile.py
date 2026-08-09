from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .units import flow_rate_to_m3s, length_to_m, rainfall_rate_to_mmhr, volume_to_m3


@dataclass(frozen=True)
class BranchTensors:
    initial_state: np.ndarray
    rainfall: np.ndarray
    settings: np.ndarray
    previous_actuator_flow: np.ndarray
    target_states: np.ndarray
    target_actuator_flows: np.ndarray
    elapsed_seconds: np.ndarray
    node_ids: tuple[str, ...]
    actuator_ids: tuple[str, ...]
    action_or_sequence_sha256: str


def _pivot(frame: pd.DataFrame, *, times: np.ndarray, ids: tuple[str, ...], id_col: str, value_col: str) -> np.ndarray:
    table = frame.pivot(index="elapsed_seconds", columns=id_col, values=value_col).reindex(index=times, columns=list(ids))
    if table.isna().any().any():
        raise ValueError(f"missing values while pivoting {value_col}")
    return table.to_numpy(dtype=float)


def _settings_for_intervals(meta: dict[str, object], times: np.ndarray, actuator_ids: tuple[str, ...]) -> tuple[np.ndarray, str]:
    interval_times = times[:-1]
    if "candidate_settings" in meta:
        values = np.array([float(meta["candidate_settings"][aid]) for aid in actuator_ids], dtype=float)  # type: ignore[index]
        return np.repeat(values[None, :], interval_times.size, axis=0), str(meta["candidate_action_sha256"])
    if "settings_sequence" in meta:
        sequence = meta["settings_sequence"]
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("invalid settings_sequence metadata")
        checkpoint_seconds = int(meta["checkpoint_minutes"]) * 60
        block = int(meta["control_block_seconds"])
        rows: list[list[float]] = []
        for elapsed in interval_times:
            step = min(max(0, int((int(elapsed) - checkpoint_seconds) // block)), len(sequence) - 1)
            action = sequence[step]
            rows.append([float(action[aid]) for aid in actuator_ids])
        return np.asarray(rows, dtype=float), str(meta["sequence_sha256"])
    raise ValueError("metadata has neither candidate_settings nor settings_sequence")


def _resolve_outlet(
    sid: str,
    connection: dict[str, tuple[int, str]],
    valid_nodes: set[str],
) -> str | None:
    seen: set[str] = set()
    current = sid
    while current not in seen:
        seen.add(current)
        kind, target = connection[current]
        if int(kind) == 2:
            return target if target in valid_nodes else None
        if int(kind) != 1 or target not in connection:
            return None
        current = target
    raise ValueError(f"subcatchment outlet cycle detected at {sid}")


def _node_rainfall(
    sub: pd.DataFrame,
    *,
    times: np.ndarray,
    node_ids: tuple[str, ...],
    system_units: str,
) -> np.ndarray:
    """Map realised rainfall at the interval start to ultimate receiving nodes."""

    node_index = {nid: i for i, nid in enumerate(node_ids)}
    valid_nodes = set(node_ids)
    first = sub.sort_values("elapsed_seconds").groupby("subcatchment_id", sort=False).first()
    connection = {
        str(sid): (int(row["outlet_connection_type"]), str(row["outlet_id"]))
        for sid, row in first.iterrows()
    }
    outlet = {sid: _resolve_outlet(sid, connection, valid_nodes) for sid in connection}
    forcing = np.zeros((times.size, len(node_ids), 1), dtype=float)
    for ti, elapsed in enumerate(times):
        rows = sub[sub["elapsed_seconds"] == elapsed]
        if rows.empty:
            raise ValueError(f"missing subcatchment rainfall at elapsed={elapsed}")
        rainfall_by_node: dict[str, list[float]] = {}
        for _, row in rows.iterrows():
            receiving_node = outlet.get(str(row["subcatchment_id"]))
            if receiving_node is not None:
                rainfall_by_node.setdefault(receiving_node, []).append(float(row["rainfall"]))
        for receiving_node, values in rainfall_by_node.items():
            forcing[ti, node_index[receiving_node], 0] = float(
                rainfall_rate_to_mmhr(np.mean(values), system_units)
            )
    return forcing


def compile_branch_tensors(metadata_path: str | Path) -> BranchTensors:
    """Compile one authoritative D2/D3 branch to causal SI Step2 arrays.

    State layout is ``[depth_m, head_m, flooding_m3s, volume_m3]``. The only exogenous
    formal forcing is rainfall observed/forecast causally; authoritative SWMM runoff is
    retained in raw evidence for diagnostics but excluded from model inputs.
    """

    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    root = meta_path.parent
    node = pd.read_csv(root / meta["node_file"], compression="infer")
    act = pd.read_csv(root / meta["actuator_file"], compression="infer")
    sub = pd.read_csv(root / meta["subcatchment_file"], compression="infer")
    flow_units = str(meta["flow_units"])
    system_units = str(meta["system_units"])

    node_ids = tuple(node["node_id"].astype(str).drop_duplicates().tolist())
    actuator_ids = tuple(act["actuator_id"].astype(str).drop_duplicates().tolist())
    times = np.array(sorted(node["elapsed_seconds"].unique()), dtype=int)
    if times.size < 2:
        raise ValueError("branch needs checkpoint plus at least one post-action sample")
    if node.loc[node["elapsed_seconds"] == times[0], "phase"].iloc[0] != "PRE_ACTION_CHECKPOINT":
        raise ValueError("first branch sample must be PRE_ACTION_CHECKPOINT")

    depth = length_to_m(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="depth"), system_units)
    head = length_to_m(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="head"), system_units)
    flooding = flow_rate_to_m3s(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="flooding"), flow_units)
    volume = volume_to_m3(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="volume"), system_units)
    state = np.stack([depth, head, flooding, volume], axis=-1)

    flow = flow_rate_to_m3s(_pivot(act, times=times, ids=actuator_ids, id_col="actuator_id", value_col="flow"), flow_units)
    settings, action_sha = _settings_for_intervals(meta, times, actuator_ids)
    rainfall = _node_rainfall(sub, times=times[:-1], node_ids=node_ids, system_units=system_units)

    return BranchTensors(
        initial_state=state[0],
        rainfall=rainfall,
        settings=settings,
        previous_actuator_flow=flow[0],
        target_states=state[1:],
        target_actuator_flows=flow[1:],
        elapsed_seconds=times,
        node_ids=node_ids,
        actuator_ids=actuator_ids,
        action_or_sequence_sha256=action_sha,
    )


def compile_branches_to_npz(metadata_paths: list[str | Path], output_path: str | Path) -> Path:
    branches = [compile_branch_tensors(path) for path in metadata_paths]
    if not branches:
        raise ValueError("no branch metadata supplied")
    node_ids = branches[0].node_ids
    actuator_ids = branches[0].actuator_ids
    horizon = branches[0].settings.shape[0]
    for branch in branches[1:]:
        if branch.node_ids != node_ids or branch.actuator_ids != actuator_ids:
            raise ValueError("branch topology/actuator ordering mismatch")
        if branch.settings.shape[0] != horizon:
            raise ValueError("branch horizons must match before stacking")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        initial_state=np.stack([b.initial_state for b in branches]),
        rainfall=np.stack([b.rainfall for b in branches]),
        settings=np.stack([b.settings for b in branches]),
        previous_actuator_flow=np.stack([b.previous_actuator_flow for b in branches]),
        target_states=np.stack([b.target_states for b in branches]),
        target_actuator_flows=np.stack([b.target_actuator_flows for b in branches]),
        node_ids=np.asarray(node_ids),
        actuator_ids=np.asarray(actuator_ids),
        action_or_sequence_sha256=np.asarray([b.action_or_sequence_sha256 for b in branches]),
    )
    return out
