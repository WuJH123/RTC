"""Causal rainfall semantics for Project7 V12.3 Value training.

Authoritative D2/D3 shards retain realised future rainfall because SWMM needs that
forcing to generate labels.  That realised future is *not* an admissible Value-model
input online.  V12.3 therefore consumes a checkpoint-keyed causal forecast store made
from rainfall observed no later than the decision time with the same frozen forecaster
used by runtime.

The store is deliberately fail-closed on lineage.  A forecast tensor without event,
checkpoint, history and forecast hashes is not acceptable evidence, even if its shape
happens to match the training cache.  Rainfall normalization is also derived from the
TrainFit causal forecasts themselves; oracle realised-future rainfall statistics are
not allowed to leak back through normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch

from .step2_train_response_v60 import (
    InputNormalizationV60,
    V60GroupBatch,
    V60TrainCache,
    derive_input_normalization_v60,
)

V123_CAUSAL_RAINFALL_CONTRACT = "PROJECT7_V123_VALUE_INPUT_CAUSAL_FORECAST_V3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CausalForecastStoreV123:
    group_names: tuple[str, ...]
    event_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    forecast_mmhr: np.ndarray
    history_sha256: tuple[str, ...]
    forecast_sha256: tuple[str, ...]
    forecast_contract: str
    future_realized_rainfall_not_used: bool

    def validate(self) -> None:
        count = len(self.group_names)
        if count <= 0 or len(set(self.group_names)) != count:
            raise ValueError("V123 causal rainfall store has empty/duplicate groups")
        for name, values in (
            ("event", self.event_ids),
            ("checkpoint", self.checkpoint_ids),
            ("history SHA", self.history_sha256),
            ("forecast SHA", self.forecast_sha256),
        ):
            if len(values) != count or any(not str(value) for value in values):
                raise ValueError(f"V123 causal forecast store lacks {name} lineage")
        if any(
            _SHA256.fullmatch(str(value).lower()) is None
            for value in self.history_sha256
        ):
            raise ValueError("V123 causal history hashes are not canonical SHA256 values")
        if any(
            _SHA256.fullmatch(str(value).lower()) is None
            for value in self.forecast_sha256
        ):
            raise ValueError("V123 causal forecast hashes are not canonical SHA256 values")
        if self.forecast_mmhr.ndim != 4 or self.forecast_mmhr.shape[0] != count:
            raise ValueError("V123 causal forecast must be [group,H,node,rain_channel]")
        if self.forecast_mmhr.shape[-1] != 1:
            raise ValueError("V123 causal forecast expects one rainfall channel")
        if not np.isfinite(self.forecast_mmhr).all() or np.any(
            self.forecast_mmhr < -1e-9
        ):
            raise ValueError("V123 causal rainfall forecast is non-finite/negative")
        if not self.forecast_contract:
            raise ValueError("V123 causal forecast store lacks runtime forecast contract")
        if self.future_realized_rainfall_not_used is not True:
            raise ValueError(
                "V123 causal store does not prove future realised rainfall exclusion"
            )

    def index(self) -> dict[str, int]:
        self.validate()
        return {name: i for i, name in enumerate(self.group_names)}


def _required(raw: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in raw.files:
        raise ValueError(f"V123 causal forecast store missing required field: {name}")
    return raw[name]


def load_causal_forecast_store_v123(path: str | Path) -> CausalForecastStoreV123:
    with np.load(path, allow_pickle=False) as raw:
        contract = str(_required(raw, "contract").item())
        if contract != V123_CAUSAL_RAINFALL_CONTRACT:
            raise ValueError("not a V123 causal rainfall forecast V3 store")
        result = CausalForecastStoreV123(
            group_names=tuple(_required(raw, "group_names").astype(str).tolist()),
            event_ids=tuple(_required(raw, "event_ids").astype(str).tolist()),
            checkpoint_ids=tuple(_required(raw, "checkpoint_ids").astype(str).tolist()),
            forecast_mmhr=_required(raw, "forecast_mmhr").astype(np.float32),
            history_sha256=tuple(
                _required(raw, "history_sha256").astype(str).tolist()
            ),
            forecast_sha256=tuple(
                _required(raw, "forecast_sha256").astype(str).tolist()
            ),
            forecast_contract=str(_required(raw, "forecast_contract").item()),
            future_realized_rainfall_not_used=bool(
                _required(raw, "future_realized_rainfall_not_used").item()
            ),
        )
    result.validate()
    return result


def derive_causal_input_normalization_v123(
    base: V60TrainCache,
    store: CausalForecastStoreV123,
    fit_names: Sequence[str],
) -> InputNormalizationV60:
    """Fit all input normalization from causal/TrainFit information only.

    Current state and previous managed-flow statistics remain valid from the base cache.
    Rainfall statistics are replaced by statistics of the frozen *causal forecasts* for
    the supplied TrainFit groups.  Holdout/Validation/Final groups must not be supplied.
    """
    names = tuple(str(name) for name in fit_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("V123 causal normalization requires unique TrainFit groups")
    store.validate()
    index = store.index()
    missing = [name for name in names if name not in index]
    if missing:
        raise ValueError(f"V123 causal normalization misses groups: {missing[:20]}")

    base_norm = derive_input_normalization_v60(base, names)
    rain = np.concatenate(
        [
            np.asarray(store.forecast_mmhr[index[name]], dtype=np.float64).reshape(
                -1, store.forecast_mmhr.shape[-1]
            )
            for name in names
        ],
        axis=0,
    )
    if rain.size == 0 or not np.isfinite(rain).all():
        raise ValueError("V123 causal normalization rainfall sample is invalid")
    rain_mean = rain.mean(axis=0)
    rain_std = np.sqrt(np.maximum(np.square(rain).mean(axis=0) - rain_mean**2, 1e-12))
    return InputNormalizationV60(
        state_mean=np.asarray(base_norm.state_mean, dtype=np.float32),
        state_std=np.asarray(base_norm.state_std, dtype=np.float32),
        rainfall_mean=rain_mean.astype(np.float32),
        rainfall_std=rain_std.astype(np.float32),
        flow_mean=np.asarray(base_norm.flow_mean, dtype=np.float32),
        flow_std=np.asarray(base_norm.flow_std, dtype=np.float32),
    )


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
        extras = sorted(set(self._index) - set(base.names()))
        if extras:
            raise ValueError(
                f"V123 causal forecast store contains unbound groups: {extras[:20]}"
            )

        for name in base.names():
            i = self._index[name]
            entry = base.entry(name)
            if str(entry.event_id) != self.store.event_ids[i]:
                raise ValueError(f"{name}: causal forecast event lineage mismatch")
            if str(entry.checkpoint_id) != self.store.checkpoint_ids[i]:
                raise ValueError(f"{name}: causal forecast checkpoint lineage mismatch")

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
        if name not in self._index:
            raise KeyError(f"V123 causal forecast has no group {name}")
        original = self.base.batch(name, normalization, device)
        raw = np.asarray(self.store.forecast_mmhr[self._index[name]], dtype=np.float32)
        expected = tuple(original.rainfall.shape[1:])
        if raw.shape != expected:
            raise ValueError(
                f"V123 causal forecast shape {raw.shape} != Step2 expected {expected}"
            )
        normalized = (raw - normalization.rainfall_mean) / np.maximum(
            normalization.rainfall_std, 1e-6
        )
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
    "derive_causal_input_normalization_v123",
    "load_causal_forecast_store_v123",
]
