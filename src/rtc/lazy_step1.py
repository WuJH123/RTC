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
from .swmm_data import STATE_CHANNELS


STEP1_STRATA = ("dry", "wet", "flood_or_high_depth")


@dataclass(frozen=True)
class Step1SampleRef:
    trajectory_index: int
    end_index: int
    event_id: str
    rainfall_group: str
    stratum: str = "dry"


@dataclass(frozen=True)
class _Trajectory:
    compact_path: str
    event_id: str
    rainfall_group: str
    swmm_engine_version: str


class CausalStep1TrajectoryDataset(Dataset):
    """Lazy t=0-inclusive Step1 windows backed by compact trajectories.

    Every causal window is retained in ``samples``. A current-state-only hydraulic stratum is
    attached to each reference so training can avoid being dominated by long dry/recession
    prefixes while held-out acceptance can still evaluate every window.
    """

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
        wet_fill_threshold: float = 0.25,
        high_fill_threshold: float = 0.75,
        flood_threshold_m3s: float = 1e-9,
    ):
        if history_steps < 2 or model_step_seconds <= 0 or cache_trajectories <= 0:
            raise ValueError("invalid history/model-step/cache configuration")
        if not 0.0 <= wet_fill_threshold < high_fill_threshold:
            raise ValueError("Step1 hydraulic fill thresholds are invalid")
        if flood_threshold_m3s < 0:
            raise ValueError("Step1 flooding threshold must be non-negative")
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
        if "max_depth_m" not in graph.static_node_feature_names:
            raise ValueError("Step1 stratification requires graph max_depth_m")
        self.graph = graph
        self.sensor_idx = np.asarray([graph.node_ids.index(n) for n in sensor_nodes], dtype=np.int64)
        self.history_steps = int(history_steps)
        self.model_step_seconds = int(model_step_seconds)
        self.cache_limit = int(cache_trajectories)
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.trajectories: list[_Trajectory] = []
        self.samples: list[Step1SampleRef] = []
        self.wet_fill_threshold = float(wet_fill_threshold)
        self.high_fill_threshold = float(high_fill_threshold)
        self.flood_threshold_m3s = float(flood_threshold_m3s)
        engine_versions: set[str] = set()

        max_depth_col = graph.static_node_feature_names.index("max_depth_m")
        max_depth = np.asarray(graph.static_node_features[:, max_depth_col], dtype=np.float32)
        valid_depth = max_depth > 1e-6
        depth_index = STATE_CHANNELS.index("depth_m")
        flooding_index = STATE_CHANNELS.index("flooding_m3s")

        for _, row in frame.iterrows():
            meta_path = Path(str(row["metadata_path"]))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            compact_name = meta.get("compact_file")
            if not compact_name:
                raise ValueError(f"Formal Step1 requires compact trajectory evidence: {meta_path}")
            engine_version = str(meta.get("swmm_engine_version", "")).strip()
            if not engine_version:
                raise ValueError(f"Formal Step1 trajectory lacks SWMM engine version: {meta_path}")
            compact = meta_path.parent / str(compact_name)
            with np.load(compact, allow_pickle=False) as raw:
                times = raw["elapsed_seconds"].astype(np.int64)
                nodes = tuple(raw["node_ids"].astype(str).tolist())
                actuators = tuple(raw["actuator_ids"].astype(str).tolist())
                channels = tuple(raw["state_channels"].astype(str).tolist())
                if nodes != graph.node_ids or actuators != graph.actuator_ids:
                    raise ValueError(f"trajectory schema differs from locked graph: {compact}")
                if channels != STATE_CHANNELS:
                    raise ValueError(f"trajectory state channels differ from locked Step1 schema: {compact}")
                if times.size < history_steps:
                    continue
                if int(times[0]) != 0:
                    raise ValueError(
                        f"Formal Step1 trajectory must include the causal t=0 frame: {compact}"
                    )
                dt = np.diff(times)
                if not np.all(dt == model_step_seconds):
                    raise ValueError(
                        f"trajectory time grid differs from frozen model step {model_step_seconds}s: {compact}"
                    )
                state = raw["state_si"].astype(np.float32)
                rain = raw["rainfall_mmhr"].astype(np.float32)
                if state.shape[:2] != (times.size, len(graph.node_ids)):
                    raise ValueError(f"Step1 state array shape differs from graph/time schema: {compact}")
                depth = np.maximum(state[:, :, depth_index], 0.0)
                fill = np.zeros_like(depth)
                if np.any(valid_depth):
                    fill[:, valid_depth] = depth[:, valid_depth] / max_depth[None, valid_depth]
                max_fill = np.max(fill, axis=1)
                positive_flood = np.maximum(state[:, :, flooding_index], 0.0).sum(axis=1)
                rain_active = np.max(rain.reshape(times.size, -1), axis=1) > 1e-9

            engine_versions.add(engine_version)
            ti = len(self.trajectories)
            event_id = str(row["event_id"])
            rainfall_group = str(row["rainfall_group"])
            self.trajectories.append(
                _Trajectory(str(compact), event_id, rainfall_group, engine_version)
            )
            for end in range(history_steps - 1, times.size):
                if positive_flood[end] > self.flood_threshold_m3s or max_fill[end] >= self.high_fill_threshold:
                    stratum = "flood_or_high_depth"
                elif rain_active[end] or max_fill[end] >= self.wet_fill_threshold:
                    stratum = "wet"
                else:
                    stratum = "dry"
                self.samples.append(
                    Step1SampleRef(ti, end, event_id, rainfall_group, stratum)
                )
        if not self.samples:
            raise ValueError("no causal Step1 windows were available")
        if len(engine_versions) != 1:
            raise ValueError(
                f"Formal Step1 cannot mix SWMM engine versions: {sorted(engine_versions)}"
            )
        self.swmm_engine_version = next(iter(engine_versions))
        self.stratum_counts = {
            stratum: int(sum(ref.stratum == stratum for ref in self.samples))
            for stratum in STEP1_STRATA
        }

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
    """Trajectory-local batching with optional hydraulic-stratified fixed epoch budgets.

    Validation uses the default unstratified mode and therefore evaluates every causal window.
    Training can request stratification so long dry/recovery periods do not dominate wet/high
    hydraulic states. Sampling remains inside development/train and never changes rainfall-group
    splits.
    """

    def __init__(
        self,
        dataset: CausalStep1TrajectoryDataset,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        stratified: bool = False,
        samples_per_trajectory: int | None = None,
        stratum_weights: dict[str, float] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if samples_per_trajectory is not None and samples_per_trajectory <= 0:
            raise ValueError("samples_per_trajectory must be positive or None")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.stratified = bool(stratified)
        self.samples_per_trajectory = samples_per_trajectory
        weights = stratum_weights or {
            "dry": 0.25,
            "wet": 0.35,
            "flood_or_high_depth": 0.40,
        }
        if set(weights) != set(STEP1_STRATA) or any(float(v) < 0 for v in weights.values()):
            raise ValueError("Step1 stratum weights must define non-negative dry/wet/high weights")
        if sum(float(v) for v in weights.values()) <= 0:
            raise ValueError("Step1 stratum weights cannot all be zero")
        self.stratum_weights = {k: float(v) for k, v in weights.items()}
        groups: dict[int, list[int]] = {}
        group_strata: dict[int, dict[str, list[int]]] = {}
        for i, ref in enumerate(dataset.samples):
            groups.setdefault(ref.trajectory_index, []).append(i)
            group_strata.setdefault(
                ref.trajectory_index, {stratum: [] for stratum in STEP1_STRATA}
            )[ref.stratum].append(i)
        self.groups = groups
        self.group_strata = group_strata
        self.epoch = 0

    def _sample_stratified(self, tid: int, rng: np.random.Generator) -> np.ndarray:
        strata = self.group_strata[int(tid)]
        available = [s for s in STEP1_STRATA if strata[s]]
        if not available:
            return np.asarray([], dtype=int)
        total_unique = sum(len(strata[s]) for s in available)
        budget = int(self.samples_per_trajectory or total_unique)
        active_weight = sum(self.stratum_weights[s] for s in available)
        raw_quota = {
            s: budget * self.stratum_weights[s] / active_weight for s in available
        }
        quota = {s: int(np.floor(raw_quota[s])) for s in available}
        remaining = budget - sum(quota.values())
        order = sorted(
            available,
            key=lambda s: (raw_quota[s] - quota[s], self.stratum_weights[s]),
            reverse=True,
        )
        for i in range(remaining):
            quota[order[i % len(order)]] += 1
        selected: list[np.ndarray] = []
        for stratum in available:
            q = quota[stratum]
            if q <= 0:
                continue
            values = np.asarray(strata[stratum], dtype=int)
            selected.append(
                rng.choice(values, size=q, replace=len(values) < q).astype(int)
            )
        if not selected:
            return np.asarray([], dtype=int)
        result = np.concatenate(selected)
        if self.shuffle:
            rng.shuffle(result)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        trajectory_ids = np.asarray(list(self.groups), dtype=int)
        if self.shuffle:
            rng.shuffle(trajectory_ids)
        for tid in trajectory_ids:
            if self.stratified:
                values = self._sample_stratified(int(tid), rng)
            else:
                values = np.asarray(self.groups[int(tid)], dtype=int)
                if self.shuffle:
                    rng.shuffle(values)
            for start in range(0, len(values), self.batch_size):
                yield values[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        if self.stratified:
            total = 0
            for tid, values in self.groups.items():
                n = int(self.samples_per_trajectory or len(values))
                total += (n + self.batch_size - 1) // self.batch_size
            return total
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.groups.values())
