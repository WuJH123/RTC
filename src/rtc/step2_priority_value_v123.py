"""Exact priority-flood-volume (PFV) labels for Project7 V12.3.

No new SWMM simulation is required to construct these labels: V6/V7 shards already
store authoritative per-node cumulative post-checkpoint flood volume.  V12.3 simply
sums the frozen priority-node subset and forms candidate-minus-reference deltas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

V123_PFV_LABEL_CONTRACT = "PROJECT7_V123_EXACT_PRIORITY_NODE_FLOOD_VOLUME_LABEL_V1"


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


def priority_node_indices_v123(
    node_ids: Sequence[str], priority_nodes: Sequence[str]
) -> np.ndarray:
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
    """Extract PFV labels from a V60TrainCache group entry without mutating the cache."""
    arrays = entry.arrays
    if "exact_node_flood_volume_m3" not in arrays or "node_ids" not in arrays:
        raise ValueError("Step2 cache lacks exact node flood volume/node ordering for PFV")
    candidates = [i for i in entry.indices if i != entry.reference_index]
    pidx = priority_node_indices_v123(
        np.asarray(arrays["node_ids"]).astype(str).tolist(), priority_nodes
    )
    return exact_pfv_labels_v123(
        arrays["exact_node_flood_volume_m3"],
        reference_index=entry.reference_index,
        candidate_indices=candidates,
        priority_indices=pidx,
    )


__all__ = [
    "PriorityVolumeLabelsV123",
    "V123_PFV_LABEL_CONTRACT",
    "exact_pfv_labels_v123",
    "group_pfv_labels_v123",
    "priority_node_indices_v123",
]
