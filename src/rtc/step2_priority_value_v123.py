"""Exact priority-flood-volume (PFV) labels for Project7 V12.3.

No new SWMM simulation is required: V6/V7 shards already store authoritative per-node
post-checkpoint cumulative flood volume. V12.3 sums the frozen priority subset and can
present those deltas through the existing V7 scalar-Value training interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache

V123_PFV_LABEL_CONTRACT = "PROJECT7_V123_EXACT_PRIORITY_NODE_FLOOD_VOLUME_LABEL_V2"


@dataclass(frozen=True)
class PriorityVolumeLabelsV123:
    reference_pfv_m3: float
    candidate_pfv_m3: np.ndarray
    delta_pfv_m3: np.ndarray

    def validate(self) -> None:
        candidate = np.asarray(self.candidate_pfv_m3, dtype=np.float64)
        delta = np.asarray(self.delta_pfv_m3, dtype=np.float64)
        if candidate.ndim != 1 or delta.shape != candidate.shape or candidate.size == 0:
            raise ValueError("V123 PFV candidate/delta vectors are invalid")
        if not np.isfinite(candidate).all() or not np.isfinite(delta).all() or not np.isfinite(self.reference_pfv_m3):
            raise ValueError("V123 PFV labels contain non-finite values")
        if self.reference_pfv_m3 < -1e-8 or np.any(candidate < -1e-8):
            raise ValueError("V123 PFV physical volumes cannot be negative")


def priority_node_indices_v123(node_ids: Sequence[str], priority_nodes: Sequence[str]) -> np.ndarray:
    ids = tuple(str(x) for x in node_ids)
    priority = tuple(str(x) for x in priority_nodes)
    if not priority or len(set(priority)) != len(priority):
        raise ValueError("V123 requires a non-empty unique frozen priority-node list")
    lookup = {node: i for i, node in enumerate(ids)}
    missing = [node for node in priority if node not in lookup]
    if missing:
        raise ValueError(f"priority nodes absent from Step2 node ordering: {missing}")
    return np.asarray([lookup[node] for node in priority], dtype=np.int64)


def exact_pfv_labels_v123(
    exact_node_flood_volume_m3: np.ndarray,
    *,
    reference_index: int,
    candidate_indices: Sequence[int],
    priority_indices: np.ndarray,
) -> PriorityVolumeLabelsV123:
    flood = np.asarray(exact_node_flood_volume_m3, dtype=np.float64)
    if flood.ndim != 2:
        raise ValueError("V123 exact node flood volume must be [branch,node]")
    pidx = np.asarray(priority_indices, dtype=np.int64).reshape(-1)
    if not pidx.size or np.any(pidx < 0) or np.any(pidx >= flood.shape[1]):
        raise ValueError("V123 priority indices are invalid")
    candidates = np.asarray(list(candidate_indices), dtype=np.int64)
    if candidates.size == 0 or np.any(candidates < 0) or np.any(candidates >= flood.shape[0]):
        raise ValueError("V123 candidate indices are invalid")
    if not 0 <= int(reference_index) < flood.shape[0]:
        raise ValueError("V123 reference index is invalid")
    reference = float(flood[int(reference_index), pidx].sum())
    candidate = flood[candidates][:, pidx].sum(axis=1)
    result = PriorityVolumeLabelsV123(
        reference_pfv_m3=reference,
        candidate_pfv_m3=candidate.astype(np.float64),
        delta_pfv_m3=(candidate - reference).astype(np.float64),
    )
    result.validate()
    return result


def group_pfv_labels_v123(entry, priority_nodes: Sequence[str]) -> PriorityVolumeLabelsV123:
    arrays = entry.arrays
    if "exact_node_flood_volume_m3" not in arrays or "node_ids" not in arrays:
        raise ValueError("Step2 cache lacks exact node flood volume/node ordering for PFV")
    candidates = [i for i in entry.indices if i != entry.reference_index]
    pidx = priority_node_indices_v123(np.asarray(arrays["node_ids"]).astype(str).tolist(), priority_nodes)
    return exact_pfv_labels_v123(
        arrays["exact_node_flood_volume_m3"],
        reference_index=entry.reference_index,
        candidate_indices=candidates,
        priority_indices=pidx,
    )


class PriorityValueCacheV123:
    """Read-only adapter that reuses V7 training code with exact PFV as scalar target."""

    def __init__(self, base: V60TrainCache, priority_nodes: Sequence[str]) -> None:
        self.base = base
        self.priority_nodes = tuple(str(node) for node in priority_nodes)
        if not self.priority_nodes:
            raise ValueError("V123 PFV cache requires frozen priority nodes")
        # Fail early before a long training run.
        for name in base.names()[:1]:
            _ = group_pfv_labels_v123(base.entry(name), self.priority_nodes)

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
        labels = group_pfv_labels_v123(self.base.entry(name), self.priority_nodes)
        truth = torch.as_tensor(
            labels.delta_pfv_m3.astype(np.float32),
            dtype=original.true_delta_tfv_m3.dtype,
            device=torch.device(device),
        )[None]
        if truth.shape != original.true_delta_tfv_m3.shape:
            raise RuntimeError("V123 PFV labels do not align with candidate batch")
        return V60GroupBatch(
            source_kind=original.source_kind,
            group_name=original.group_name,
            initial_state=original.initial_state,
            rainfall=original.rainfall,
            reference_settings=original.reference_settings,
            candidate_settings=original.candidate_settings,
            previous_actuator_flow=original.previous_actuator_flow,
            elapsed_seconds=original.elapsed_seconds,
            true_reference_states=original.true_reference_states,
            true_candidate_states=original.true_candidate_states,
            true_reference_flows=original.true_reference_flows,
            true_candidate_flows=original.true_candidate_flows,
            true_delta_tfv_m3=truth,
        )


__all__ = [
    "PriorityValueCacheV123",
    "PriorityVolumeLabelsV123",
    "V123_PFV_LABEL_CONTRACT",
    "exact_pfv_labels_v123",
    "group_pfv_labels_v123",
    "priority_node_indices_v123",
]
