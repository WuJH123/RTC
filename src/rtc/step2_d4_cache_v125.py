"""Isolated D4 cache/index adapters for the V125 data-support ablation.

D4 never enters the historical V60 role guard. It is compiled as its own source kind
(`D4`) with the causal Sparse-RBC anchor as explicit reference. FIT and AUDIT are
materialised as separate caches. A composite read-only view lets the unchanged V124
trainer consume base D2/D3 plus D4-FIT without altering old cache semantics.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .data_index import standardize_d3_run_index
from .step2_causal_rainfall_v123 import CausalForecastStoreV123
from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache

D4_CACHE_CONTRACT_V125 = "PROJECT7_V125_D4_ISOLATED_CACHE_V1"
D4_SOURCE_KIND = "D4"
D4_REFERENCE_ROLE = "reference"
D4_CANDIDATE_ROLE = "D4_V125_ANCHOR_NEIGHBOURHOOD_CANDIDATE"


def build_d4_run_index_v125(
    execution_manifest: pd.DataFrame,
    run_summary: pd.DataFrame,
    *,
    split_role: str,
) -> pd.DataFrame:
    role = str(split_role).lower()
    if role not in {"fit", "audit"}:
        raise ValueError("V125 D4 split_role must be fit or audit")
    required = {
        "plan_row_id",
        "d4_split_role",
        "event_id",
        "rainfall_group",
        "checkpoint_id",
        "sequence_sha256",
        "candidate_family",
    }
    missing = sorted(required - set(execution_manifest.columns))
    if missing:
        raise ValueError(f"V125 D4 execution manifest lacks provenance: {missing}")
    designed = execution_manifest.copy()
    if set(designed["d4_split_role"].astype(str)) - {"fit", "audit"}:
        raise ValueError("V125 D4 execution manifest has invalid split roles")
    rain_roles = designed.groupby("rainfall_group")["d4_split_role"].nunique()
    if bool((rain_roles != 1).any()):
        raise ValueError("V125 D4 split leaks within rainfall group")
    designed = designed[designed["d4_split_role"].astype(str) == role].copy()
    if designed.empty:
        raise ValueError(f"V125 D4 {role} execution manifest is empty")

    runs = standardize_d3_run_index(run_summary)
    keys = ["checkpoint_id", "action_or_sequence_sha256"]
    provenance = designed.rename(columns={"sequence_sha256": "action_or_sequence_sha256"})[
        [
            "checkpoint_id",
            "action_or_sequence_sha256",
            "plan_row_id",
            "d4_split_role",
            "event_id",
            "rainfall_group",
            "candidate_family",
        ]
    ]
    joined = runs.merge(
        provenance,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_run", "_plan"),
    )
    if len(joined) != len(provenance):
        raise ValueError(
            f"V125 D4 {role} run summary is incomplete/stale: "
            f"matched {len(joined)} of {len(provenance)} frozen rows"
        )
    for column in ("event_id", "rainfall_group"):
        run = joined[f"{column}_run"].astype(str)
        plan = joined[f"{column}_plan"].astype(str)
        if bool((run != plan).any()):
            raise ValueError(f"V125 D4 run/plan lineage mismatch in {column}")
        joined[column] = plan

    result = pd.DataFrame()
    for column in runs.columns:
        if column in joined.columns:
            result[column] = joined[column]
        elif f"{column}_run" in joined.columns:
            result[column] = joined[f"{column}_run"]
        else:
            raise ValueError(f"V125 D4 lost standardized run column {column}")
    result["event_id"] = joined["event_id"]
    result["rainfall_group"] = joined["rainfall_group"]
    result["source_kind"] = D4_SOURCE_KIND
    result["data_role"] = np.where(
        joined["candidate_family"].astype(str) == "anchor_scale_1.00",
        D4_REFERENCE_ROLE,
        D4_CANDIDATE_ROLE,
    )
    result["d4_split_role"] = role
    result["plan_row_id"] = joined["plan_row_id"].astype(str)
    result["candidate_family"] = joined["candidate_family"].astype(str)
    if set(result["scientific_split"].astype(str).str.lower()) != {"development"}:
        raise ValueError("V125 D4 cache is development-only")
    if set(result["development_fold"].astype(str).str.lower()) != {"train"}:
        raise ValueError("V125 D4 cache is Train-only")
    if result["metadata_path"].duplicated().any():
        raise ValueError("V125 D4 cache duplicates SWMM branch metadata")
    for key, group in result.groupby(
        ["rainfall_group", "event_id", "checkpoint_id"], sort=False, dropna=False
    ):
        refs = int((group["data_role"].astype(str) == D4_REFERENCE_ROLE).sum())
        if refs != 1:
            raise ValueError(f"V125 D4 group {key} requires one anchor reference, got {refs}")
        if len(group) < 2:
            raise ValueError(f"V125 D4 group {key} has no anchor-neighbourhood candidate")
    return result.sort_values(
        ["rainfall_group", "event_id", "checkpoint_id", "data_role", "action_or_sequence_sha256"]
    ).reset_index(drop=True)


class D4CausalForecastValueCacheV125:
    """Reuse the frozen V123 causal forecast at the same event/checkpoint for D4."""

    def __init__(self, base: V60TrainCache, store: CausalForecastStoreV123) -> None:
        store.validate()
        self.base = base
        self.store = store
        # V123 store intentionally does not carry rainfall_group as a separate field.
        # D2 and D3 may both bind the same event/checkpoint, so require all matching
        # causal forecasts to be exactly identical before reusing one for D4.
        lookup: dict[tuple[str, str], list[int]] = {}
        for i, (event, checkpoint) in enumerate(
            zip(store.event_ids, store.checkpoint_ids, strict=True)
        ):
            lookup.setdefault((str(event), str(checkpoint)), []).append(i)
        self._index: dict[str, int] = {}
        for name in base.names(D4_SOURCE_KIND):
            entry = base.entry(name)
            key = (str(entry.event_id), str(entry.checkpoint_id))
            matches = lookup.get(key, [])
            if not matches:
                raise ValueError(f"{name}: no frozen causal rainfall forecast at D4 checkpoint")
            first = np.asarray(store.forecast_mmhr[matches[0]], dtype=np.float32)
            for other in matches[1:]:
                if not np.array_equal(
                    first, np.asarray(store.forecast_mmhr[other], dtype=np.float32)
                ):
                    raise ValueError(
                        f"{name}: base causal store disagrees within event/checkpoint"
                    )
            self._index[name] = int(matches[0])

    @property
    def manifest_path(self):
        return self.base.manifest_path

    def names(self, source: str | None = None) -> list[str]:
        return self.base.names(source)

    def entry(self, name: str):
        return self.base.entry(name)

    def batch(
        self,
        name: str,
        normalization: InputNormalizationV60,
        device: torch.device | str,
    ) -> V60GroupBatch:
        original = self.base.batch(name, normalization, device)
        raw = np.asarray(self.store.forecast_mmhr[self._index[name]], dtype=np.float32)
        if raw.shape != tuple(original.rainfall.shape[1:]):
            raise ValueError(f"{name}: D4 causal rainfall shape mismatch")
        normalized = (raw - normalization.rainfall_mean) / np.maximum(
            normalization.rainfall_std, 1e-6
        )
        rain = torch.as_tensor(
            normalized, dtype=original.rainfall.dtype, device=torch.device(device)
        )[None]
        return V60GroupBatch(
            source_kind=original.source_kind,
            group_name=original.group_name,
            initial_state=original.initial_state,
            rainfall=rain,
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


class CompositeValueCacheV125:
    """Read-only union of non-overlapping causal group caches."""

    def __init__(self, caches: Sequence[Any]) -> None:
        if not caches:
            raise ValueError("V125 composite cache cannot be empty")
        self.caches = tuple(caches)
        owner: dict[str, Any] = {}
        for cache in self.caches:
            for name in cache.names():
                if name in owner:
                    raise ValueError(f"V125 composite cache duplicates group {name}")
                owner[name] = cache
        self._owner = owner

    @property
    def manifest_path(self) -> str:
        return "COMPOSITE_V125_READ_ONLY"

    def names(self, source: str | None = None) -> list[str]:
        names = sorted(self._owner)
        if source is None:
            return names
        prefix = str(source).upper() + "::"
        return [name for name in names if name.startswith(prefix)]

    def entry(self, name: str):
        return self._owner[name].entry(name)

    def batch(
        self,
        name: str,
        normalization: InputNormalizationV60,
        device: torch.device | str,
    ):
        return self._owner[name].batch(name, normalization, device)


__all__ = [
    "CompositeValueCacheV125",
    "D4CausalForecastValueCacheV125",
    "D4_CACHE_CONTRACT_V125",
    "D4_CANDIDATE_ROLE",
    "D4_REFERENCE_ROLE",
    "D4_SOURCE_KIND",
    "build_d4_run_index_v125",
]
