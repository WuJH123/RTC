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
    swmm_engine_version: str
    exact_node_flood_volume_m3: np.ndarray | None = None

    @property
    def model_step_seconds(self) -> int:
        dt = np.diff(self.elapsed_seconds.astype(np.int64))
        if not dt.size or np.any(dt <= 0) or not np.all(dt == dt[0]):
            raise ValueError("Step2 branch requires one regular positive time step")
        return int(dt[0])

    @property
    def horizon_steps(self) -> int:
        return int(self.settings.shape[0])


def _pivot(
    frame: pd.DataFrame,
    *,
    times: np.ndarray,
    ids: tuple[str, ...],
    id_col: str,
    value_col: str,
) -> np.ndarray:
    table = frame.pivot(index="elapsed_seconds", columns=id_col, values=value_col).reindex(
        index=times, columns=list(ids)
    )
    if table.isna().any().any():
        raise ValueError(f"missing values while pivoting {value_col}")
    return table.to_numpy(dtype=float)


def _exact_node_flood(
    meta_path: Path, meta: dict[str, object], node_ids: tuple[str, ...]
) -> np.ndarray | None:
    name = meta.get("node_statistics_file")
    if not name:
        return None
    frame = pd.read_csv(meta_path.parent / str(name), compression="infer")
    if not {"node_id", "delta_flooding_volume_m3"}.issubset(frame.columns):
        raise ValueError("node statistics file lacks exact flooding-volume columns")
    table = frame.assign(node_id=frame["node_id"].astype(str)).set_index("node_id")
    missing = [node for node in node_ids if node not in table.index]
    if missing:
        raise ValueError(f"node statistics missing nodes: {missing[:20]}")
    return table.reindex(node_ids)["delta_flooding_volume_m3"].to_numpy(dtype=np.float32)


def _engine(meta: dict[str, object], meta_path: Path) -> str:
    value = str(meta.get("swmm_engine_version", "")).strip()
    if not value:
        raise ValueError(f"Formal Step2 branch lacks SWMM engine version: {meta_path}")
    return value


def _compile_compact(meta_path: Path, meta: dict[str, object]) -> BranchTensors:
    with np.load(meta_path.parent / str(meta["compact_file"]), allow_pickle=False) as raw:
        times = raw["elapsed_seconds"].astype(np.int64)
        if times.size < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("compact branch requires strictly increasing checkpoint+horizon samples")
        state = raw["state_si"].astype(np.float32)
        rainfall_all = raw["rainfall_mmhr"].astype(np.float32)
        flow = raw["actuator_flow_m3s"].astype(np.float32)
        node_ids = tuple(raw["node_ids"].astype(str).tolist())
        actuator_ids = tuple(raw["actuator_ids"].astype(str).tolist())
        if (
            state.shape[0] != times.size
            or rainfall_all.shape[0] != times.size
            or flow.shape[0] != times.size
        ):
            raise ValueError("compact branch time dimensions do not align")

        if "candidate_setting" in raw.files:
            candidate = raw["candidate_setting"].astype(np.float32)
            settings = np.repeat(candidate[None, :], times.size - 1, axis=0)
            action_sha = str(meta["candidate_action_sha256"])
        elif "commanded_setting" in raw.files:
            settings = raw["commanded_setting"][:-1].astype(np.float32)
            action_sha = str(meta["sequence_sha256"])
        else:
            raise ValueError("compact branch lacks candidate/commanded setting")

    branch = BranchTensors(
        initial_state=state[0],
        rainfall=rainfall_all[:-1],
        settings=settings,
        previous_actuator_flow=flow[0],
        target_states=state[1:],
        target_actuator_flows=flow[1:],
        elapsed_seconds=times,
        node_ids=node_ids,
        actuator_ids=actuator_ids,
        action_or_sequence_sha256=action_sha,
        swmm_engine_version=_engine(meta, meta_path),
        exact_node_flood_volume_m3=_exact_node_flood(meta_path, meta, node_ids),
    )
    _ = branch.model_step_seconds
    return branch


def _resolve_outlet(
    sid: str, connection: dict[str, tuple[int, str]], valid_nodes: set[str]
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
    return None


def _node_rainfall(
    sub: pd.DataFrame,
    *,
    times: np.ndarray,
    node_ids: tuple[str, ...],
    system_units: str,
) -> np.ndarray:
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    valid_nodes = set(node_ids)
    first = sub.sort_values("elapsed_seconds").groupby("subcatchment_id", sort=False).first()
    connection = {
        str(sid): (int(row["outlet_connection_type"]), str(row["outlet_id"]))
        for sid, row in first.iterrows()
    }
    outlet = {sid: _resolve_outlet(sid, connection, valid_nodes) for sid in connection}
    forcing = np.zeros((times.size, len(node_ids), 1), dtype=np.float32)
    for ti, elapsed in enumerate(times):
        rows = sub[sub["elapsed_seconds"] == elapsed]
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


def _compile_legacy_raw(meta_path: Path, meta: dict[str, object]) -> BranchTensors:
    root = meta_path.parent
    node = pd.read_csv(root / str(meta["node_file"]), compression="infer")
    act = pd.read_csv(root / str(meta["actuator_file"]), compression="infer")
    sub = pd.read_csv(root / str(meta["subcatchment_file"]), compression="infer")
    flow_units, system_units = str(meta["flow_units"]), str(meta["system_units"])
    node_ids = tuple(node["node_id"].astype(str).drop_duplicates().tolist())
    actuator_ids = tuple(act["actuator_id"].astype(str).drop_duplicates().tolist())
    times = np.array(sorted(node["elapsed_seconds"].unique()), dtype=int)
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("legacy branch time grid invalid")
    depth = length_to_m(
        _pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="depth"),
        system_units,
    )
    head = length_to_m(
        _pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="head"),
        system_units,
    )
    flooding = flow_rate_to_m3s(
        _pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="flooding"),
        flow_units,
    )
    volume = volume_to_m3(
        _pivot(node, times=times, ids=node_ids, id_col="node_id", value_col="volume"),
        system_units,
    )
    extras = []
    for name in ("total_inflow", "total_outflow"):
        if name in node.columns:
            extras.append(
                flow_rate_to_m3s(
                    _pivot(node, times=times, ids=node_ids, id_col="node_id", value_col=name),
                    flow_units,
                )
            )
    state = np.stack([depth, head, flooding, volume, *extras], axis=-1).astype(np.float32)
    flow = flow_rate_to_m3s(
        _pivot(act, times=times, ids=actuator_ids, id_col="actuator_id", value_col="flow"),
        flow_units,
    ).astype(np.float32)
    if "candidate_settings" in meta:
        setting = np.array(
            [float(meta["candidate_settings"][aid]) for aid in actuator_ids], dtype=np.float32  # type: ignore[index]
        )
        settings = np.repeat(setting[None, :], times.size - 1, axis=0)
        action_sha = str(meta["candidate_action_sha256"])
    else:
        sequence = meta["settings_sequence"]
        block = int(meta["control_block_seconds"])
        checkpoint_seconds = int(meta["checkpoint_minutes"]) * 60
        settings = np.asarray(
            [
                [
                    float(
                        sequence[
                            min(max((int(t) - checkpoint_seconds) // block, 0), len(sequence) - 1)
                        ][aid]
                    )
                    for aid in actuator_ids
                ]
                for t in times[:-1]
            ],
            dtype=np.float32,
        )
        action_sha = str(meta["sequence_sha256"])
    rainfall = _node_rainfall(
        sub, times=times[:-1], node_ids=node_ids, system_units=system_units
    )
    branch = BranchTensors(
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
        swmm_engine_version=_engine(meta, meta_path),
        exact_node_flood_volume_m3=_exact_node_flood(meta_path, meta, node_ids),
    )
    _ = branch.model_step_seconds
    return branch


def compile_branch_tensors(metadata_path: str | Path) -> BranchTensors:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("compact_file"):
        return _compile_compact(meta_path, meta)
    return _compile_legacy_raw(meta_path, meta)


def compile_branches_to_npz(
    metadata_paths: list[str | Path],
    output_path: str | Path,
    *,
    provenance: pd.DataFrame | None = None,
) -> Path:
    branches = [compile_branch_tensors(path) for path in metadata_paths]
    if not branches:
        raise ValueError("no branch metadata supplied")
    if provenance is not None and len(provenance) != len(branches):
        raise ValueError("provenance rows must align one-for-one with metadata paths")
    node_ids, actuator_ids = branches[0].node_ids, branches[0].actuator_ids
    horizon = branches[0].horizon_steps
    state_dim = branches[0].target_states.shape[-1]
    model_step_seconds = branches[0].model_step_seconds
    swmm_engine_version = branches[0].swmm_engine_version
    for branch in branches[1:]:
        if branch.node_ids != node_ids or branch.actuator_ids != actuator_ids:
            raise ValueError("branch topology/actuator ordering mismatch")
        if branch.horizon_steps != horizon or branch.target_states.shape[-1] != state_dim:
            raise ValueError("branch horizon/state schema mismatch")
        if branch.model_step_seconds != model_step_seconds:
            raise ValueError(
                "Step2 branch time-step mismatch: Phase-0/high-frequency and production branches "
                "must never be mixed in one surrogate dataset"
            )
        if branch.swmm_engine_version != swmm_engine_version:
            raise ValueError(
                "Step2 branches from different SWMM engine versions cannot be mixed: "
                f"{branch.swmm_engine_version} != {swmm_engine_version}"
            )
    payload: dict[str, np.ndarray] = {
        "initial_state": np.stack([b.initial_state for b in branches]),
        "rainfall": np.stack([b.rainfall for b in branches]),
        "settings": np.stack([b.settings for b in branches]),
        "previous_actuator_flow": np.stack([b.previous_actuator_flow for b in branches]),
        "target_states": np.stack([b.target_states for b in branches]),
        "target_actuator_flows": np.stack([b.target_actuator_flows for b in branches]),
        "elapsed_seconds": np.stack([b.elapsed_seconds for b in branches]),
        "node_ids": np.asarray(node_ids),
        "actuator_ids": np.asarray(actuator_ids),
        "action_or_sequence_sha256": np.asarray(
            [b.action_or_sequence_sha256 for b in branches]
        ),
        "model_step_seconds": np.asarray(model_step_seconds, dtype=np.int64),
        "horizon_steps": np.asarray(horizon, dtype=np.int64),
        "swmm_engine_version": np.asarray(swmm_engine_version),
    }
    if all(b.exact_node_flood_volume_m3 is not None for b in branches):
        payload["exact_node_flood_volume_m3"] = np.stack(
            [
                b.exact_node_flood_volume_m3
                for b in branches
                if b.exact_node_flood_volume_m3 is not None
            ]
        )
    if provenance is not None:
        for column in (
            "event_id",
            "rainfall_group",
            "scientific_split",
            "development_fold",
            "data_role",
            "checkpoint_id",
            "actuator_id",
            "base_setting",
            "requested_setting",
            "base_action_sha256",
            "source_kind",
        ):
            if column in provenance.columns:
                # Keep shard provenance readable with allow_pickle=False.  Pandas
                # may return an object array for ``Series.astype(str).to_numpy()``;
                # object arrays force NumPy readers to enable pickle and made the
                # V6 training loader fail before the first epoch.
                payload[column] = provenance[column].fillna("").astype(str).to_numpy(
                    dtype=str
                )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    return out
