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


@dataclass(frozen=True)
class CausalForecastStoreV123:
    group_names: tuple[str, ...]
    forecast_mmhr: np.ndarray
    history_sha256: tuple[str, ...]
    forecast_contract: str

    def validate(self) -> None:
        count = len(self.group_names)
        if count <= 0 or len(set(self.group_names)) != count:
            raise ValueError("V123 causal rainfall store has empty/duplicate groups")
        if self.forecast_mmhr.ndim != 4 or self.forecast_mmhr.shape[0] != count:
            raise ValueError("V123 causal forecast must be [group,H,node,rain_channel]")
        if self.forecast_mmhr.shape[-1] != 1:
            raise ValueError("V123 causal forecast expects one rainfall channel")
        if not np.isfinite(self.forecast_mmhr).all() or np.any(self.forecast_mmhr < -1e-9):
            raise ValueError("V123 causal rainfall forecast is non-finite/negative")
        if len(self.history_sha256) != count or any(not value for value in self.history_sha256):
            raise ValueError("V123 causal forecast store lacks history hashes")
        if not self.forecast_contract:
            raise ValueError("V123 causal forecast store lacks runtime forecast contract")

    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.group_names)}


def load_causal_forecast_store_v123(path: str | Path) -> CausalForecastStoreV123:
    with np.load(path, allow_pickle=False) as raw:
        contract = str(raw["contract"].item())
        if contract != V123_CAUSAL_RAINFALL_CONTRACT:
            raise ValueError("not a V123 causal rainfall forecast store")
        result = CausalForecastStoreV123(
            group_names=tuple(raw["group_names"].astype(str).tolist()),
            forecast_mmhr=raw["forecast_mmhr"].astype(np.float32),
            history_sha256=tuple(raw["history_sha256"].astype(str).tolist()),
            forecast_contract=str(raw["forecast_contract"].item()),
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
    "load_causal_forecast_store_v123",
]
