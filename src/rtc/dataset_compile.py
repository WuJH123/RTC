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
    rainfall_runoff: np.ndarray
    settings: np.ndarray
    previous_actuator_flow: np.ndarray
    target_states: np.ndarray
    target_actuator_flows: np.ndarray
    elapsed_seconds: np.ndarray
    node_ids: tuple[str, ...]
    actuator_ids: tuple[str, ...]
    candidate_action_sha256: str


def _pivot(frame: pd.DataFrame, *, times: np.ndarray, ids: tuple[str, ...], id_col: str, value_col: str) -> np.ndarray:
    table = frame.pivot(index="elapsed_seconds", columns=id_col, values=value_col).reindex(index=times, columns=list(ids))
    if table.isna().any().any():
        raise ValueError(f"missing values while pivoting {value_col}")
    return table.to_numpy(dtype=float)


def compile_branch_tensors(metadata_path: str | Path) -> BranchTensors:
    """Compile one authoritative branch to SI free-rollout training arrays.

    State layout is ``[depth_m, head_m, flooding_m3s, volume_m3]``. Exogenous forcing is
    node-local ``[rainfall_mmhr, runoff_m3s]``; subcatchment runoff is accumulated at its
    SWMM outlet node. The pre-action checkpoint is the initial state and only later samples
    are targets, preserving action causality.
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
    settings_vector = np.array([float(meta["candidate_settings"][aid]) for aid in actuator_ids], dtype=float)
    settings = np.repeat(settings_vector[None, :], times.size - 1, axis=0)

    node_index = {nid: i for i, nid in enumerate(node_ids)}
    forcing = np.zeros((times.size - 1, len(node_ids), 2), dtype=float)
    post_times = times[1:]
    for ti, elapsed in enumerate(post_times):
        rows = sub[sub["elapsed_seconds"] == elapsed]
        if rows.empty:
            raise ValueError(f"missing subcatchment forcing at elapsed={elapsed}")
        rainfall_by_node: dict[str, list[float]] = {}
        runoff_by_node: dict[str, float] = {}
        for _, row in rows.iterrows():
            # PySWMM connection type 2 means the subcatchment loads directly to a node.
            if int(row["outlet_connection_type"]) != 2:
                continue
            outlet = str(row["outlet_id"])
            if outlet not in node_index:
                continue
            rainfall_by_node.setdefault(outlet, []).append(float(row["rainfall"]))
            runoff_by_node[outlet] = runoff_by_node.get(outlet, 0.0) + float(row["runoff"])
        for outlet, values in rainfall_by_node.items():
            forcing[ti, node_index[outlet], 0] = float(rainfall_rate_to_mmhr(np.mean(values), system_units))
        for outlet, value in runoff_by_node.items():
            forcing[ti, node_index[outlet], 1] = float(flow_rate_to_m3s(value, flow_units))

    return BranchTensors(
        initial_state=state[0],
        rainfall_runoff=forcing,
        settings=settings,
        previous_actuator_flow=flow[0],
        target_states=state[1:],
        target_actuator_flows=flow[1:],
        elapsed_seconds=times,
        node_ids=node_ids,
        actuator_ids=actuator_ids,
        candidate_action_sha256=str(meta["candidate_action_sha256"]),
    )


def compile_branches_to_npz(metadata_paths: list[str | Path], output_path: str | Path) -> Path:
    """Stack same-schema branches into one compressed Step2 tensor dataset."""

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
        rainfall=np.stack([b.rainfall_runoff for b in branches]),
        settings=np.stack([b.settings for b in branches]),
        previous_actuator_flow=np.stack([b.previous_actuator_flow for b in branches]),
        target_states=np.stack([b.target_states for b in branches]),
        target_actuator_flows=np.stack([b.target_actuator_flows for b in branches]),
        node_ids=np.asarray(node_ids),
        actuator_ids=np.asarray(actuator_ids),
        candidate_action_sha256=np.asarray([b.candidate_action_sha256 for b in branches]),
    )
    return out
