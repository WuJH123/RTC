"""Causal rainfall semantics for Project7 V12.3 Value training.

Legacy D2/D3 shards contain the *realised future* rainfall trajectory because that
forcing was needed to run authoritative SWMM branches.  Passing that tensor directly to
the Value model during training creates an oracle-forcing/train-runtime mismatch: online
RTC only has rainfall observed up to the decision time.

V12.3 therefore requires an explicit checkpoint-keyed causal forecast store.  Labels
remain authoritative future SWMM outcomes; model inputs are generated only from the
pre-decision rainfall history using the same frozen forecast algorithm as runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache

V123_CAUSAL_RAINFALL_CONTRACT = "PROJECT7_V123_VALUE_INPUT_CAUSAL_FORECAST_V1"


def causal_forecast_from_history_v123(
    observed_history_mmhr: np.ndarray,
    *,
    horizon_steps: int,
    decay_per_step: float = 0.92,
) -> np.ndarray:
    """Build the single central forecast from the last observed rainfall frame.

    The history is retained for lineage and future forecaster upgrades, but the
    frozen V123 contract uses ``history_steps_for_level=1``: only the current
    observed frame may seed the forecast.  No realised future array is accepted by
    this helper.
    """
    history = np.asarray(observed_history_mmhr, dtype=np.float32)
    if history.ndim != 3 or history.shape[0] < 1 or history.shape[-1] != 1:
        raise ValueError("V123 observed rainfall history must be [history,node,1]")
    if horizon_steps <= 0:
        raise ValueError("V123 forecast horizon must be positive")
    if not 0.0 <= float(decay_per_step) <= 1.0:
        raise ValueError("V123 rainfall decay must lie in [0,1]")
    current = np.clip(history[-1], 0.0, None)
    scale = np.power(float(decay_per_step), np.arange(int(horizon_steps), dtype=np.float32))
    forecast = scale[:, None, None] * current[None]
    if not np.isfinite(forecast).all():
        raise ValueError("V123 causal forecast is non-finite")
    return forecast.astype(np.float32)


@dataclass(frozen=True)
class CausalForecastStoreV123:
    group_names: tuple[str, ...]
    event_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    checkpoint_elapsed_seconds: np.ndarray
    forecast_mmhr: np.ndarray
    history_sha256: tuple[str, ...]
    forecast_sha256: tuple[str, ...]
    forecast_contract: str
    future_realized_rainfall_not_used: bool

    def validate(self) -> None:
        count = len(self.group_names)
        if count <= 0 or len(set(self.group_names)) != count:
            raise ValueError("V123 causal rainfall store has empty/duplicate groups")
        if len(self.event_ids) != count or any(not str(value) for value in self.event_ids):
            raise ValueError("V123 causal rainfall store lacks event/checkpoint lineage")
        if len(self.checkpoint_ids) != count or any(not str(value) for value in self.checkpoint_ids):
            raise ValueError("V123 causal rainfall store lacks event/checkpoint lineage")
        checkpoints = np.asarray(self.checkpoint_elapsed_seconds)
        if checkpoints.shape != (count,) or not np.issubdtype(checkpoints.dtype, np.integer):
            raise ValueError("V123 event/checkpoint lineage is invalid")
        if np.any(checkpoints < 0):
            raise ValueError("V123 checkpoint elapsed seconds cannot be negative")
        if self.forecast_mmhr.ndim != 4 or self.forecast_mmhr.shape[0] != count:
            raise ValueError("V123 causal forecast must be [group,H,node,rain_channel]")
        if self.forecast_mmhr.shape[-1] != 1:
            raise ValueError("V123 causal forecast expects one rainfall channel")
        if not np.isfinite(self.forecast_mmhr).all() or np.any(self.forecast_mmhr < -1e-9):
            raise ValueError("V123 causal rainfall forecast is non-finite/negative")
        if len(self.history_sha256) != count or any(not value for value in self.history_sha256):
            raise ValueError("V123 causal forecast store lacks history hashes")
        if len(self.forecast_sha256) != count or any(not value for value in self.forecast_sha256):
            raise ValueError("V123 causal forecast store lacks forecast hashes")
        if not self.forecast_contract:
            raise ValueError("V123 causal forecast store lacks runtime forecast contract")
        if not bool(self.future_realized_rainfall_not_used):
            raise ValueError("V123 store must assert future-realised-rainfall-not-used")

    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.group_names)}


def load_causal_forecast_store_v123(path: str | Path) -> CausalForecastStoreV123:
    with np.load(path, allow_pickle=False) as raw:
        contract = str(raw["contract"].item())
        if contract != V123_CAUSAL_RAINFALL_CONTRACT:
            raise ValueError("not a V123 causal rainfall forecast store")
        result = CausalForecastStoreV123(
            group_names=tuple(raw["group_names"].astype(str).tolist()),
            event_ids=tuple(raw["event_ids"].astype(str).tolist()),
            checkpoint_ids=tuple(raw["checkpoint_ids"].astype(str).tolist()),
            checkpoint_elapsed_seconds=raw["checkpoint_elapsed_seconds"].astype(np.int64),
            forecast_mmhr=raw["forecast_mmhr"].astype(np.float32),
            history_sha256=tuple(raw["history_sha256"].astype(str).tolist()),
            forecast_sha256=tuple(raw["forecast_sha256"].astype(str).tolist()),
            forecast_contract=str(raw["forecast_contract"].item()),
            future_realized_rainfall_not_used=bool(raw["future_realized_rainfall_not_used"].item()),
        )
    result.validate()
    return result


class CausalForecastValueCacheV123:
    """Replace oracle realised-future rainfall input with causal forecast input."""

    def __init__(self, base: V60TrainCache, store: CausalForecastStoreV123) -> None:
        store.validate()
        self.base = base
        self.store = store
        self._index = store.index()
        missing = sorted(set(base.names()) - set(self._index))
        if missing:
            raise ValueError(f"V123 causal forecast store misses Step2 groups: {missing[:20]}")

    @property
    def manifest_path(self):
        return self.base.manifest_path

    def names(self, source: str | None = None) -> list[str]:
        return self.base.names(source)

    def entry(self, name: str):
        return self.base.entry(name)

    def targeted_d3_names(self) -> list[str]:
        return self.base.targeted_d3_names()

    def legacy_d3_names(self) -> list[str]:
        return self.base.legacy_d3_names()

    def batch(
        self,
        name: str,
        normalization: InputNormalizationV60,
        device: torch.device | str,
    ) -> V60GroupBatch:
        original = self.base.batch(name, normalization, device)
        raw = np.asarray(self.store.forecast_mmhr[self._index[name]], dtype=np.float32)
        expected = tuple(original.rainfall.shape[1:])
        if raw.shape != expected:
            raise ValueError(f"V123 causal forecast shape {raw.shape} != Step2 expected {expected}")
        normalized = (raw - normalization.rainfall_mean) / np.maximum(normalization.rainfall_std, 1e-6)
        rainfall = torch.as_tensor(
            normalized,
            dtype=original.rainfall.dtype,
            device=torch.device(device),
        )[None]
        return V60GroupBatch(
            source_kind=original.source_kind,
            group_name=original.group_name,
            initial_state=original.initial_state,
            rainfall=rainfall,
            reference_settings=original.reference_settings,
            candidate_settings=original.candidate_settings,
            previous_actuator_flow=original.previous_actuator_flow,
            elapsed_seconds=original.elapsed_seconds,
            true_reference_states=original.true_reference_states,
            true_candidate_states=original.true_candidate_states,
            true_reference_flows=original.true_reference_flows,
            true_candidate_flows=original.true_candidate_flows,
            true_delta_tfv_m3=original.true_delta_tfv_m3,
        )


__all__ = [
    "CausalForecastStoreV123",
    "CausalForecastValueCacheV123",
    "V123_CAUSAL_RAINFALL_CONTRACT",
    "causal_forecast_from_history_v123",
    "load_causal_forecast_store_v123",
]
