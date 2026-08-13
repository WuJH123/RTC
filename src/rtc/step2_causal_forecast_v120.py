"""Causal sparse-observation view of the existing Step2 cache for V120.

Authoritative SWMM outcomes remain unchanged. Model inputs are restricted to the
same information available online:

* current depth/head at the frozen sparse sensor nodes;
* current rainfall persisted/decayed causally;
* current actuator flow and the candidate/reference actions.

Future realised rainfall and authoritative unobserved-node checkpoint state are
never presented to the V120 Value model. Step1 remains the accepted full-state
reconstruction/monitoring module, but the TFV policy no longer depends on a
true-state-trained/reconstructed-state-runtime domain shift.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache

V120_CAUSAL_RAINFALL_CONTRACT = "PROJECT7_V120_CURRENT_RAIN_PERSISTENCE_DECAY_TRAIN_INPUT_V1"
V120_STATE_INPUT_CONTRACT = "PROJECT7_V120_SPARSE_SENSOR_DEPTH_HEAD_MEAN_IMPUTED_V1"


def causal_rainfall_from_checkpoint_v120(
    realised_future_rainfall: np.ndarray,
    *,
    decay_per_step: float = 0.92,
) -> np.ndarray:
    """Create an online-legal central forecast from checkpoint rainfall only."""
    rainfall = np.asarray(realised_future_rainfall, dtype=np.float32)
    if rainfall.ndim != 3 or rainfall.shape[0] < 1:
        raise ValueError("V120 rainfall must be [H,N,R]")
    if not 0.0 <= float(decay_per_step) <= 1.0:
        raise ValueError("V120 rainfall decay must lie in [0,1]")
    current = np.clip(rainfall[0], 0.0, None)
    scale = np.power(float(decay_per_step), np.arange(rainfall.shape[0], dtype=np.float32))
    return (scale[:, None, None] * current[None]).astype(np.float32)


def load_sensor_nodes_v120(path: str | Path) -> tuple[str, ...]:
    nodes = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("V120 sensor layout must be non-empty and unique")
    return nodes


def sensor_indices_v120(
    node_ids: Sequence[str], sensor_nodes: Sequence[str]
) -> np.ndarray:
    index = {str(node): i for i, node in enumerate(node_ids)}
    missing = [str(node) for node in sensor_nodes if str(node) not in index]
    if missing:
        raise ValueError(f"V120 sensor nodes absent from graph/cache: {missing[:20]}")
    return np.asarray([index[str(node)] for node in sensor_nodes], dtype=np.int64)


def sparse_observable_state_v120(
    full_state: np.ndarray,
    sensor_indices: np.ndarray,
) -> np.ndarray:
    """Build [N,6] from sensor depth/head only.

    Unsensed node depth/head receive the *current sensor mean*. This gives the
    existing V7 endpoint/global encoders a causal neutral value without exposing
    authoritative unobserved-node state. Channels that are not directly sensed
    (flooding, volume, total inflow/outflow) are identically zero.
    """
    state = np.asarray(full_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 6:
        raise ValueError("V120 state must be [N,6]")
    idx = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
    if not idx.size or np.any(idx < 0) or np.any(idx >= state.shape[0]):
        raise ValueError("V120 sensor indices are invalid")
    observed = state[idx, :2]
    if not np.isfinite(observed).all():
        raise ValueError("V120 observed sensor depth/head is non-finite")
    result = np.zeros_like(state, dtype=np.float32)
    result[:, :2] = observed.mean(axis=0)[None, :]
    result[idx, :2] = observed
    return result


def sparse_runtime_state_v120(
    *,
    node_count: int,
    sensor_indices: np.ndarray,
    sensor_depth_m: Sequence[float],
    sensor_head_m: Sequence[float],
) -> np.ndarray:
    idx = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
    depth = np.asarray(sensor_depth_m, dtype=np.float32).reshape(-1)
    head = np.asarray(sensor_head_m, dtype=np.float32).reshape(-1)
    if depth.size != idx.size or head.size != idx.size:
        raise ValueError("V120 runtime sensor observation length mismatch")
    if not np.isfinite(depth).all() or not np.isfinite(head).all():
        raise ValueError("V120 runtime sensor depth/head is non-finite")
    state = np.zeros((int(node_count), 6), dtype=np.float32)
    state[idx, 0] = depth
    state[idx, 1] = head
    return sparse_observable_state_v120(state, idx)


def _channel_stats(
    values: list[np.ndarray], channels: int
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(channels, dtype=np.float64)
    square = np.zeros(channels, dtype=np.float64)
    count = 0
    for value in values:
        x = np.asarray(value, dtype=np.float64).reshape(-1, channels)
        total += x.sum(axis=0)
        square += np.square(x).sum(axis=0)
        count += x.shape[0]
    if count <= 0:
        raise ValueError("V120 cannot derive normalization from no values")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 1.0e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def derive_sparse_causal_input_normalization_v120(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    *,
    sensor_indices: np.ndarray,
    decay_per_step: float = 0.92,
) -> InputNormalizationV60:
    """Derive TrainFit-only normalization under the online information budget."""
    names = list(fit_names)
    if not names:
        raise ValueError("V120 normalization requires TrainFit groups")
    entries = [cache.entry(name) for name in names]
    first = entries[0].arrays
    state_values = [
        sparse_observable_state_v120(
            np.asarray(entry.arrays["initial_state"][entry.reference_index], np.float32),
            sensor_indices,
        )
        for entry in entries
    ]
    flow_values = [
        np.asarray(entry.arrays["previous_actuator_flow"][entry.reference_index], np.float32)
        for entry in entries
    ]
    rainfall_values = [
        causal_rainfall_from_checkpoint_v120(
            np.asarray(entry.arrays["rainfall"][entry.reference_index], np.float32),
            decay_per_step=decay_per_step,
        )
        for entry in entries
    ]
    sm, ss = _channel_stats(state_values, int(first["initial_state"].shape[-1]))
    rm, rs = _channel_stats(rainfall_values, int(first["rainfall"].shape[-1]))
    fm, fs = _channel_stats(flow_values, int(first["previous_actuator_flow"].shape[-1]))
    return InputNormalizationV60(sm, ss, rm, rs, fm, fs)


def derive_causal_input_normalization_v120(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    *,
    decay_per_step: float = 0.92,
) -> InputNormalizationV60:
    """Legacy helper retained for audit; canonical V120 uses sparse-state helper."""
    names = list(fit_names)
    if not names:
        raise ValueError("V120 normalization requires TrainFit groups")
    entries = [cache.entry(name) for name in names]
    first = entries[0].arrays
    state_values = [
        np.asarray(entry.arrays["initial_state"][entry.reference_index], np.float32)
        for entry in entries
    ]
    flow_values = [
        np.asarray(entry.arrays["previous_actuator_flow"][entry.reference_index], np.float32)
        for entry in entries
    ]
    rainfall_values = [
        causal_rainfall_from_checkpoint_v120(
            np.asarray(entry.arrays["rainfall"][entry.reference_index], np.float32),
            decay_per_step=decay_per_step,
        )
        for entry in entries
    ]
    sm, ss = _channel_stats(state_values, int(first["initial_state"].shape[-1]))
    rm, rs = _channel_stats(rainfall_values, int(first["rainfall"].shape[-1]))
    fm, fs = _channel_stats(flow_values, int(first["previous_actuator_flow"].shape[-1]))
    return InputNormalizationV60(sm, ss, rm, rs, fm, fs)


class CausalForecastCacheV120:
    """Legacy rainfall-only cache view retained for diagnostics."""

    def __init__(self, base: V60TrainCache, *, decay_per_step: float = 0.92) -> None:
        self.base = base
        self.decay_per_step = float(decay_per_step)

    def entry(self, name: str):
        return self.base.entry(name)

    def batch(
        self,
        name: str,
        normalization: InputNormalizationV60,
        device: torch.device | str,
    ) -> V60GroupBatch:
        batch = self.base.batch(name, normalization, device)
        entry = self.base.entry(name)
        raw = np.asarray(entry.arrays["rainfall"][entry.reference_index], np.float32)
        causal = causal_rainfall_from_checkpoint_v120(
            raw, decay_per_step=self.decay_per_step
        )
        normalized = (
            causal - np.asarray(normalization.rainfall_mean, np.float32)
        ) / np.maximum(np.asarray(normalization.rainfall_std, np.float32), 1.0e-6)
        rainfall = torch.from_numpy(np.ascontiguousarray(normalized))[None].to(device)
        return replace(batch, rainfall=rainfall)


class SparseCausalCacheV120:
    """Canonical cache view: sparse current state + causal rainfall."""

    def __init__(
        self,
        base: V60TrainCache,
        *,
        sensor_indices: np.ndarray,
        decay_per_step: float = 0.92,
    ) -> None:
        self.base = base
        self.sensor_indices = np.asarray(sensor_indices, dtype=np.int64)
        self.causal = CausalForecastCacheV120(base, decay_per_step=decay_per_step)

    def entry(self, name: str):
        return self.base.entry(name)

    def batch(
        self,
        name: str,
        normalization: InputNormalizationV60,
        device: torch.device | str,
    ) -> V60GroupBatch:
        batch = self.causal.batch(name, normalization, device)
        entry = self.base.entry(name)
        raw = np.asarray(entry.arrays["initial_state"][entry.reference_index], np.float32)
        sparse = sparse_observable_state_v120(raw, self.sensor_indices)
        normalized = (
            sparse - np.asarray(normalization.state_mean, np.float32)
        ) / np.maximum(np.asarray(normalization.state_std, np.float32), 1.0e-6)
        initial = torch.from_numpy(np.ascontiguousarray(normalized))[None].to(device)
        return replace(batch, initial_state=initial)


__all__ = [
    "CausalForecastCacheV120",
    "SparseCausalCacheV120",
    "V120_CAUSAL_RAINFALL_CONTRACT",
    "V120_STATE_INPUT_CONTRACT",
    "causal_rainfall_from_checkpoint_v120",
    "derive_causal_input_normalization_v120",
    "derive_sparse_causal_input_normalization_v120",
    "load_sensor_nodes_v120",
    "sensor_indices_v120",
    "sparse_observable_state_v120",
    "sparse_runtime_state_v120",
]
