from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .context_features import build_node_context
from .graph import GraphSchema


@dataclass(frozen=True)
class Step1SampleRef:
    trajectory_index: int
    end_index: int
    event_id: str
    rainfall_group: str


@dataclass(frozen=True)
class _Trajectory:
    compact_path: str
    event_id: str
    rainfall_group: str


class CausalStep1TrajectoryDataset(Dataset):
    """Lazy Step1 windows backed by compact trajectories, never duplicated on disk."""

    def __init__(
        self,
        run_index: pd.DataFrame,
        *,
        graph: GraphSchema,
        sensor_nodes: tuple[str, ...],
        history_steps: int,
        model_step_seconds: int,
        scientific_split: str = "development",
        development_fold: str | None = "train",
        cache_trajectories: int = 2,
    ):
        if history_steps < 2 or model_step_seconds <= 0 or cache_trajectories <= 0:
            raise ValueError("invalid history/model-step/cache configuration")
        required = {"metadata_path", "event_id", "rainfall_group", "scientific_split"}
        missing = sorted(required - set(run_index.columns))
        if missing:
            raise ValueError(f"run index missing columns: {missing}")
        frame = run_index[run_index["scientific_split"].astype(str) == scientific_split].copy()
        if development_fold is not None and scientific_split == "development":
            if "development_fold" not in frame.columns:
                raise ValueError("development Step1 index requires development_fold")
            frame = frame[frame["development_fold"].astype(str) == development_fold]
        if frame.empty:
            raise ValueError("no Step1 trajectories remain after split/fold filtering")
        missing_sensors = sorted(set(sensor_nodes) - set(graph.node_ids))
        if missing_sensors:
            raise ValueError(f"sensor nodes absent from graph: {missing_sensors}")
        self.graph = graph
        self.sensor_idx = np.asarray([graph.node_ids.index(n) for n in sensor_nodes], dtype=np.int64)
        self.history_steps = int(history_steps)
        self.model_step_seconds = int(model_step_seconds)
        self.cache_limit = int(cache_trajectories)
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.trajectories: list[_Trajectory] = []
        self.samples: list[Step1SampleRef] = []

        for _, row in frame.iterrows():
            meta_path = Path(str(row["metadata_path"]))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            compact_name = meta.get("compact_file")
            if not compact_name:
                raise ValueError(f"Formal Step1 requires compact trajectory V2: {meta_path}")
            compact = meta_path.parent / str(compact_name)
            with np.load(compact, allow_pickle=False) as raw:
                times = raw["elapsed_seconds"].astype(np.int64)
                nodes = tuple(raw["node_ids"].astype(str).tolist())
                actuators = tuple(raw["actuator_ids"].astype(str).tolist())
                if nodes != graph.node_ids or actuators != graph.actuator_ids:
                    raise ValueError(f"trajectory schema differs from locked graph: {compact}")
                if times.size < history_steps:
                    continue
                dt = np.diff(times)
                if not np.all(dt == model_step_seconds):
                    raise ValueError(
                        f"trajectory time grid differs from frozen model step {model_step_seconds}s: {compact}"
                    )
            ti = len(self.trajectories)
            self.trajectories.append(
                _Trajectory(str(compact), str(row["event_id"]), str(row["rainfall_group"]))
            )
            for end in range(history_steps - 1, times.size):
                self.samples.append(
                    Step1SampleRef(ti, end, str(row["event_id"]), str(row["rainfall_group"]))
                )
        if not self.samples:
            raise ValueError("no causal Step1 windows were available")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, index: int) -> dict[str, np.ndarray]:
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        path = self.trajectories[index].compact_path
        with np.load(path, allow_pickle=False) as raw:
            value = {
                "state": raw["state_si"].astype(np.float32),
                "rain": raw["rainfall_mmhr"].astype(np.float32),
                "setting": raw["current_setting"].astype(np.float32),
                "flow": raw["actuator_flow_m3s"].astype(np.float32),
            }
        self.cache[index] = value
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return value

    def __getitem__(self, index: int):
        ref = self.samples[index]
        data = self._load(ref.trajectory_index)
        start = ref.end_index - self.history_steps + 1
        state = data["state"][start : ref.end_index + 1]
        n = state.shape[1]
        observed = np.zeros((self.history_steps, n, 2), dtype=np.float32)
        mask = np.zeros_like(observed)
        observed[:, self.sensor_idx] = state[:, self.sensor_idx, :2]
        mask[:, self.sensor_idx] = 1.0
        context = build_node_context(
            rainfall_mmhr=data["rain"][start : ref.end_index + 1],
            actuator_setting=data["setting"][start : ref.end_index + 1],
            actuator_flow_m3s=data["flow"][start : ref.end_index + 1],
            actuator_upstream=self.graph.actuator_upstream,
            actuator_downstream=self.graph.actuator_downstream,
            node_count=n,
        )
        target = data["state"][ref.end_index]
        return (
            torch.from_numpy(observed),
            torch.from_numpy(mask),
            torch.from_numpy(context),
            torch.from_numpy(target),
        )


class TrajectoryBatchSampler(Sampler[list[int]]):
    """Shuffle trajectories, then batch within each trajectory to preserve cache locality."""

    def __init__(self, dataset: CausalStep1TrajectoryDataset, *, batch_size: int, seed: int, shuffle: bool = True):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        groups: dict[int, list[int]] = {}
        for i, ref in enumerate(dataset.samples):
            groups.setdefault(ref.trajectory_index, []).append(i)
        self.groups = groups
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        trajectory_ids = np.asarray(list(self.groups), dtype=int)
        if self.shuffle:
            rng.shuffle(trajectory_ids)
        for tid in trajectory_ids:
            values = np.asarray(self.groups[int(tid)], dtype=int)
            if self.shuffle:
                rng.shuffle(values)
            for start in range(0, len(values), self.batch_size):
                yield values[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.groups.values())
