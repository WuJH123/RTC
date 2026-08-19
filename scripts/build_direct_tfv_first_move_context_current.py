"""Build candidate-free causal first-move contexts from completed no-control D0 prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.context_features import build_node_context
from rtc.direct_tfv_first_move_context import DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import _load_graph, _load_lines
from rtc.step1_runtime_v127 import load_frozen_step1_v127


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _metadata_compact(metadata_path: Path) -> tuple[dict, Path]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("data_contract") != "D0_D1_COMPACT_TRAJECTORY_V3_T0_CAUSAL":
        raise ValueError(f"first-move context requires current D0 metadata: {metadata_path}")
    if meta.get("python_actuator_writes") is not False or meta.get("native_controls_enabled") is not False:
        raise ValueError("first-move calibration prefix must be no-control with no Python/native RTC")
    compact = metadata_path.parent / str(meta.get("compact_file", ""))
    if not compact.is_file():
        raise FileNotFoundError(f"D0 compact missing: {compact}")
    hashes = meta.get("generated_artifact_sha256", {})
    expected = str(hashes.get("compact_file", "")) if isinstance(hashes, dict) else ""
    if expected and _sha(compact) != expected:
        raise ValueError("D0 compact SHA differs from stamped metadata")
    return meta, compact


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-index", required=True)
    p.add_argument("--event-registry", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--development-fold")
    p.add_argument("--checkpoint-elapsed-seconds", type=int, default=10800)
    p.add_argument("--history-steps", type=int, default=13)
    p.add_argument("--rainfall-horizon-steps", type=int, default=72)
    args = p.parse_args()
    if args.history_steps < 2 or args.rainfall_horizon_steps <= 0:
        raise ValueError("invalid causal history/forecast horizon")
    if args.checkpoint_elapsed_seconds <= 0 or args.checkpoint_elapsed_seconds % 300:
        raise ValueError("first-move checkpoint must be a positive 300-s grid time")

    runs = pd.read_csv(args.run_index)
    events = pd.read_csv(args.event_registry)
    required_runs = {"event_id", "rainfall_group", "strategy", "metadata_path"}
    required_events = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    if missing := sorted(required_runs - set(runs.columns)):
        raise ValueError(f"D0 run index missing columns: {missing}")
    if missing := sorted(required_events - set(events.columns)):
        raise ValueError(f"event registry missing columns: {missing}")
    runs = runs[runs["strategy"].astype(str) == "no_control"].copy()
    if len(runs) == 0:
        raise ValueError("first-move context found no no-control D0 runs")
    merged = runs.merge(
        events,
        on=["event_id", "rainfall_group"],
        suffixes=("_run", "_event"),
        validate="one_to_one",
    )
    if args.development_fold is not None:
        if "development_fold" not in merged:
            raise ValueError("event registry lacks development_fold")
        merged = merged[merged["development_fold"].astype(str) == str(args.development_fold)]
    if len(merged) < 24:
        raise ValueError("first-move context requires at least 24 role-disjoint Development groups")
    if any(str(value).lower() != "development" for value in merged["scientific_split"]):
        raise ValueError("first-move context may use Development events only")
    if merged["rainfall_group"].astype(str).duplicated().any():
        raise ValueError("first-move context requires one checkpoint per rainfall group")

    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    if len(set(sensors)) != len(sensors):
        raise ValueError("sensor list contains duplicates")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    step1 = load_frozen_step1_v127(args.step1, device)
    node_index = {str(node): i for i, node in enumerate(graph.node_ids)}
    missing_sensors = [node for node in sensors if node not in node_index]
    if missing_sensors:
        raise ValueError(f"first-move sensors absent from graph: {missing_sensors}")
    sensor_index = np.asarray([node_index[node] for node in sensors], dtype=int)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)

    event_ids = []
    rainfall_groups = []
    checkpoint_ids = []
    elapsed_seconds = []
    inp_paths = []
    trajectory_metadata_paths = []
    scientific_splits = []
    development_folds = []
    states = []
    rainfall_histories = []
    default_forecasts = []
    targets = []
    flows = []
    prefix_shas = []
    context_shas = []

    for row in merged.sort_values("rainfall_group", kind="mergesort").itertuples(index=False):
        metadata_path = Path(str(row.metadata_path)).resolve()
        meta, compact_path = _metadata_compact(metadata_path)
        with np.load(compact_path, allow_pickle=False) as raw:
            times = raw["elapsed_seconds"].astype(np.int64)
            nodes = tuple(raw["node_ids"].astype(str).tolist())
            actuators = tuple(raw["actuator_ids"].astype(str).tolist())
            state_si = raw["state_si"].astype(np.float32)
            rain = raw["rainfall_mmhr"].astype(np.float32)
            current = raw["current_setting"].astype(np.float32)
            target = raw["target_setting"].astype(np.float32)
            flow = raw["actuator_flow_m3s"].astype(np.float32)
        if nodes != tuple(str(x) for x in graph.node_ids):
            raise ValueError(f"{row.event_id}: D0 node order differs from graph")
        if actuators != tuple(str(x) for x in graph.actuator_ids):
            raise ValueError(f"{row.event_id}: D0 actuator order differs from graph")
        matches = np.flatnonzero(times == int(args.checkpoint_elapsed_seconds))
        if len(matches) != 1:
            raise ValueError(f"{row.event_id}: D0 lacks exact checkpoint {args.checkpoint_elapsed_seconds}s")
        end = int(matches[0])
        start = end - int(args.history_steps) + 1
        if start < 0:
            raise ValueError(f"{row.event_id}: insufficient causal history before checkpoint")
        history_times = times[start : end + 1]
        if not np.all(np.diff(history_times) == 300):
            raise ValueError(f"{row.event_id}: causal history is not a fixed 300-s grid")
        n = len(nodes)
        observed_frames = []
        mask_frames = []
        context_frames = []
        for ti in range(start, end + 1):
            observed = np.zeros((n, 2), dtype=np.float32)
            mask = np.zeros((n, 2), dtype=np.float32)
            observed[sensor_index, 0] = state_si[ti, sensor_index, 0]
            observed[sensor_index, 1] = state_si[ti, sensor_index, 1]
            mask[sensor_index, :] = 1.0
            context = build_node_context(
                rainfall_mmhr=rain[ti],
                actuator_setting=current[ti],
                actuator_flow_m3s=flow[ti],
                actuator_upstream=graph.actuator_upstream,
                actuator_downstream=graph.actuator_downstream,
                node_count=n,
            )
            observed_frames.append(observed)
            mask_frames.append(mask)
            context_frames.append(context)
        with torch.no_grad():
            reconstructed = step1(
                torch.as_tensor(np.stack(observed_frames)[None], dtype=torch.float32, device=device),
                torch.as_tensor(np.stack(mask_frames)[None], dtype=torch.float32, device=device),
                static,
                edges,
                torch.as_tensor(np.stack(context_frames)[None], dtype=torch.float32, device=device),
            ).detach().cpu().numpy()[0].astype(np.float32)
        rain_history = rain[start : end + 1].astype(np.float32)
        default_forecast = PersistenceDecayForecast(
            decay_per_step=0.92,
            scenario_multipliers=(1.0,),
            history_steps_for_level=1,
        ).forecast(rain_history, horizon_steps=int(args.rainfall_horizon_steps)).astype(np.float32)
        active = target[end].astype(np.float32)
        previous_flow = flow[end].astype(np.float32)
        prefix_digest = hashlib.sha256()
        prefix_digest.update(_sha(compact_path).encode("utf-8"))
        prefix_digest.update(np.asarray(times[: end + 1], dtype=np.int64).tobytes())
        prefix_digest.update(state_si[: end + 1].tobytes())
        prefix_digest.update(target[: end + 1].tobytes())
        prefix_digest.update(current[: end + 1].tobytes())
        prefix_digest.update(flow[: end + 1].tobytes())
        prefix_sha = prefix_digest.hexdigest()
        context_digest = hashlib.sha256()
        for array in (reconstructed, rain_history, default_forecast, active, previous_flow):
            context_digest.update(_array_sha(array).encode("utf-8"))

        event_ids.append(str(row.event_id))
        rainfall_groups.append(str(row.rainfall_group))
        checkpoint_ids.append(f"{row.event_id}__t{int(args.checkpoint_elapsed_seconds)}")
        elapsed_seconds.append(int(args.checkpoint_elapsed_seconds))
        inp_paths.append(str(Path(str(row.inp_path)).resolve()))
        trajectory_metadata_paths.append(str(metadata_path))
        scientific_splits.append(str(row.scientific_split))
        development_folds.append(str(getattr(row, "development_fold", "")))
        states.append(reconstructed)
        rainfall_histories.append(rain_history)
        default_forecasts.append(default_forecast)
        targets.append(active)
        flows.append(previous_flow)
        prefix_shas.append(prefix_sha)
        context_shas.append(context_digest.hexdigest())

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path = out.with_suffix(".npz")
    np.savez_compressed(
        data_path,
        contract=np.asarray(DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT),
        event_ids=np.asarray(event_ids),
        rainfall_groups=np.asarray(rainfall_groups),
        checkpoint_ids=np.asarray(checkpoint_ids),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.int64),
        inp_paths=np.asarray(inp_paths),
        trajectory_metadata_paths=np.asarray(trajectory_metadata_paths),
        scientific_splits=np.asarray(scientific_splits),
        development_folds=np.asarray(development_folds),
        current_state=np.stack(states),
        rainfall_history=np.stack(rainfall_histories),
        default_rainfall_forecast=np.stack(default_forecasts),
        active_target=np.stack(targets),
        previous_actuator_flow=np.stack(flows),
        prefix_sha256=np.asarray(prefix_shas),
        context_sha256=np.asarray(context_shas),
        graph_sha256=np.asarray(_sha(args.graph)),
        step1_sha256=np.asarray(_sha(args.step1)),
        sensors_sha256=np.asarray(_sha(args.sensors)),
        candidate_rows_used=np.asarray(False),
        generic_d3_candidate_dependency=np.asarray(False),
        causal_future_rainfall_used=np.asarray(False),
    )
    payload = {
        "contract": DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT,
        "development_only": True,
        "rainfall_group_count": len(rainfall_groups),
        "checkpoint_elapsed_seconds": int(args.checkpoint_elapsed_seconds),
        "history_steps": int(args.history_steps),
        "rainfall_horizon_steps": int(args.rainfall_horizon_steps),
        "candidate_rows_used": False,
        "generic_d3_candidate_dependency": False,
        "causal_future_rainfall_used": False,
        "data_path": data_path.name,
        "data_sha256": _sha(data_path),
        "run_index_sha256": _sha(args.run_index),
        "event_registry_sha256": _sha(args.event_registry),
        "graph_sha256": _sha(args.graph),
        "step1_sha256": _sha(args.step1),
        "sensors_sha256": _sha(args.sensors),
        "rainfall_groups": rainfall_groups,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
