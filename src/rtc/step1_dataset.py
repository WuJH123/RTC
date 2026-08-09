from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .units import flow_rate_to_m3s, length_to_m, rainfall_rate_to_mmhr, volume_to_m3


@dataclass(frozen=True)
class Step1Windows:
    observed_history: np.ndarray
    observation_mask: np.ndarray
    context_history: np.ndarray
    target_state: np.ndarray
    event_ids: np.ndarray
    rainfall_groups: np.ndarray
    scientific_splits: np.ndarray
    target_elapsed_seconds: np.ndarray
    node_ids: tuple[str, ...]
    actuator_ids: tuple[str, ...]
    context_feature_names: tuple[str, ...]


def _pivot(frame: pd.DataFrame, *, times: np.ndarray, ids: tuple[str, ...], id_col: str, value_col: str) -> np.ndarray:
    table = frame.pivot(index="elapsed_seconds", columns=id_col, values=value_col).reindex(index=times, columns=list(ids))
    if table.isna().any().any():
        raise ValueError(f"missing values while pivoting {value_col}")
    return table.to_numpy(dtype=float)


def _load_trajectory(metadata_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    root = meta_path.parent
    node = pd.read_csv(root / str(meta["node_file"]), compression="infer")
    act = pd.read_csv(root / str(meta["actuator_file"]), compression="infer")
    sub = pd.read_csv(root / str(meta["subcatchment_file"]), compression="infer")
    flow_units = str(meta["flow_units"])
    system_units = str(meta["system_units"])

    node_ids = tuple(node["node_id"].astype(str).drop_duplicates().tolist())
    actuator_ids = tuple(act["actuator_id"].astype(str).drop_duplicates().tolist())
    times = np.array(sorted(node["elapsed_seconds"].unique()), dtype=int)
    if times.size < 2:
        raise ValueError("trajectory has too few samples")

    depth = length_to_m(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="depth"), system_units)
    head = length_to_m(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="head"), system_units)
    flooding = flow_rate_to_m3s(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="flooding"), flow_units)
    volume = volume_to_m3(_pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="volume"), system_units)
    full_state = np.stack([depth, head, flooding, volume], axis=-1)

    target_setting = _pivot(act, times=times, ids=actuator_ids, id_col="actuator_id", value_col="target_setting")
    current_setting = _pivot(act, times=times, ids=actuator_ids, id_col="actuator_id", value_col="current_setting")
    actuator_flow = flow_rate_to_m3s(
        _pivot(act, times=times, ids=actuator_ids, id_col="actuator_id", value_col="flow"), flow_units
    )

    rainfall_mean = np.zeros(times.size, dtype=float)
    rainfall_max = np.zeros(times.size, dtype=float)
    for i, elapsed in enumerate(times):
        rows = sub[sub["elapsed_seconds"] == elapsed]
        if rows.empty:
            raise ValueError(f"missing subcatchment rainfall at elapsed={elapsed}")
        rainfall_mean[i] = float(rainfall_rate_to_mmhr(rows["rainfall"].astype(float).mean(), system_units))
        rainfall_max[i] = float(rainfall_rate_to_mmhr(rows["rainfall"].astype(float).max(), system_units))

    # Formal Step1 context contains only information available at decision time:
    # observed rainfall and actuator target/current/flow readback. SWMM runoff is retained
    # in authoritative files for diagnostics but is deliberately excluded from model input.
    context = np.concatenate(
        [
            rainfall_mean[:, None],
            rainfall_max[:, None],
            target_setting,
            current_setting,
            actuator_flow,
        ],
        axis=1,
    )
    feature_names = (
        "rainfall_mean_mmhr",
        "rainfall_max_mmhr",
        *(f"target_setting:{aid}" for aid in actuator_ids),
        *(f"current_setting:{aid}" for aid in actuator_ids),
        *(f"actuator_flow_m3s:{aid}" for aid in actuator_ids),
    )
    return times, full_state, context, np.stack([depth, head], axis=-1), node_ids, actuator_ids, tuple(feature_names)


def compile_step1_windows(
    run_index: pd.DataFrame,
    *,
    sensor_nodes: tuple[str, ...],
    history_steps: int = 13,
    stride_steps: int = 1,
) -> Step1Windows:
    """Build causal Step1 examples from D0/D1 full-event authoritative trajectories.

    ``run_index`` must provide ``metadata_path``, ``event_id``, ``rainfall_group`` and
    ``scientific_split``. The target is the same current time as the final history frame,
    not a future state. Only listed sensors are visible in ``observed_history``; full
    network SWMM truth appears only in the supervised ``target_state``.
    """

    required = {"metadata_path", "event_id", "rainfall_group", "scientific_split"}
    missing = sorted(required - set(run_index.columns))
    if missing:
        raise ValueError(f"run_index missing columns: {missing}")
    if history_steps < 2 or stride_steps <= 0:
        raise ValueError("history_steps must be >=2 and stride_steps positive")

    observed_windows: list[np.ndarray] = []
    mask_windows: list[np.ndarray] = []
    context_windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    event_ids: list[str] = []
    rainfall_groups: list[str] = []
    splits: list[str] = []
    target_times: list[int] = []
    canonical_nodes: tuple[str, ...] | None = None
    canonical_actuators: tuple[str, ...] | None = None
    canonical_context_names: tuple[str, ...] | None = None

    for _, row in run_index.iterrows():
        times, full_state, context, sensor_state_source, node_ids, actuator_ids, context_names = _load_trajectory(str(row["metadata_path"]))
        if canonical_nodes is None:
            canonical_nodes, canonical_actuators, canonical_context_names = node_ids, actuator_ids, context_names
        elif node_ids != canonical_nodes or actuator_ids != canonical_actuators or context_names != canonical_context_names:
            raise ValueError("D0/D1 trajectories do not share an identical model schema")
        node_index = {node: i for i, node in enumerate(node_ids)}
        missing_sensors = sorted(set(sensor_nodes) - set(node_ids))
        if missing_sensors:
            raise ValueError(f"sensor nodes absent from trajectory: {missing_sensors}")
        sensor_idx = np.array([node_index[n] for n in sensor_nodes], dtype=int)

        observed = np.zeros_like(sensor_state_source, dtype=float)
        mask = np.zeros_like(sensor_state_source, dtype=float)
        observed[:, sensor_idx, :] = sensor_state_source[:, sensor_idx, :]
        mask[:, sensor_idx, :] = 1.0

        for end in range(history_steps - 1, len(times), stride_steps):
            start = end - history_steps + 1
            observed_windows.append(observed[start : end + 1])
            mask_windows.append(mask[start : end + 1])
            context_windows.append(context[start : end + 1])
            targets.append(full_state[end])
            event_ids.append(str(row["event_id"]))
            rainfall_groups.append(str(row["rainfall_group"]))
            splits.append(str(row["scientific_split"]))
            target_times.append(int(times[end]))

    if not observed_windows or canonical_nodes is None or canonical_actuators is None or canonical_context_names is None:
        raise ValueError("no Step1 windows were generated")
    return Step1Windows(
        observed_history=np.stack(observed_windows).astype(np.float32),
        observation_mask=np.stack(mask_windows).astype(np.float32),
        context_history=np.stack(context_windows).astype(np.float32),
        target_state=np.stack(targets).astype(np.float32),
        event_ids=np.asarray(event_ids),
        rainfall_groups=np.asarray(rainfall_groups),
        scientific_splits=np.asarray(splits),
        target_elapsed_seconds=np.asarray(target_times, dtype=np.int64),
        node_ids=canonical_nodes,
        actuator_ids=canonical_actuators,
        context_feature_names=canonical_context_names,
    )


def save_step1_windows(windows: Step1Windows, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observed_history=windows.observed_history,
        observation_mask=windows.observation_mask,
        context_history=windows.context_history,
        target_state=windows.target_state,
        event_ids=windows.event_ids,
        rainfall_groups=windows.rainfall_groups,
        scientific_splits=windows.scientific_splits,
        target_elapsed_seconds=windows.target_elapsed_seconds,
        node_ids=np.asarray(windows.node_ids),
        actuator_ids=np.asarray(windows.actuator_ids),
        context_feature_names=np.asarray(windows.context_feature_names),
    )
    return out
