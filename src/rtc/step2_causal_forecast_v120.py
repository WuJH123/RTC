"""Causal-rainfall view of the existing Step2 cache for V120 value learning.

The authoritative branch outcome and current hydraulic state remain unchanged.
Only the rainfall *model input* is changed: future realised SWMM rainfall is
replaced by the same persistence/decay forecast that is legal online.  The online
current state continues to come from the accepted Step1 sparse-sensor
reconstruction, preserving the intended Step1 -> Step2 -> Step3 architecture.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import torch

from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache

V120_CAUSAL_RAINFALL_CONTRACT = "PROJECT7_V120_CURRENT_RAIN_PERSISTENCE_DECAY_TRAIN_INPUT_V1"


def causal_rainfall_from_checkpoint_v120(
    realised_future_rainfall: np.ndarray,
    *,
    decay_per_step: float = 0.92,
) -> np.ndarray:
    """Create an online-legal central forecast from the checkpoint rainfall only."""
    rainfall = np.asarray(realised_future_rainfall, dtype=np.float32)
    if rainfall.ndim != 3 or rainfall.shape[0] < 1:
        raise ValueError("V120 rainfall must be [H,N,R]")
    if not 0.0 <= float(decay_per_step) <= 1.0:
        raise ValueError("V120 rainfall decay must lie in [0,1]")
    current = np.clip(rainfall[0], 0.0, None)
    scale = np.power(float(decay_per_step), np.arange(rainfall.shape[0], dtype=np.float32))
    return (scale[:, None, None] * current[None]).astype(np.float32)


def _channel_stats(values: list[np.ndarray], channels: int) -> tuple[np.ndarray, np.ndarray]:
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


def derive_causal_input_normalization_v120(
    cache: V60TrainCache,
    fit_names: Sequence[str],
    *,
    decay_per_step: float = 0.92,
) -> InputNormalizationV60:
    """Derive TrainFit normalization with causal rainfall inputs only."""
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
    """Drop-in V60TrainCache view with causal forecast rainfall in every batch."""

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


__all__ = [
    "CausalForecastCacheV120",
    "V120_CAUSAL_RAINFALL_CONTRACT",
    "causal_rainfall_from_checkpoint_v120",
    "derive_causal_input_normalization_v120",
]
