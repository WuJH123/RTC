"""Current V128 objective truth partition with canonical float32 semantics.

The authoritative SWMM node-volume labels are stored as float32 in the training cache and
live CUDA losses reduce them in float32.  The historical pair census re-summed the same
labels in NumPy float64, which can move a pair across the frozen 1 m3 informative threshold
and produce impossible coverage failures such as 544/542.  Current training therefore uses
one canonical float32 predicate for the pair census, first-pass report, and live gradients.
"""
from __future__ import annotations

import numpy as np

from . import step2_train_v128_exact as _exact

V128_TRUTH_PARTITION_CONTRACT = "PROJECT7_V128_CANONICAL_FLOAT32_TRUTH_PARTITION_V1"
V128_OBJECTIVE_TRAINING_CONTRACT = _exact.V128_OBJECTIVE_TRAINING_CONTRACT


def canonical_float32_tfv_and_delta(truth_node_volume_m3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(truth_node_volume_m3, dtype=np.float32)
    if truth.ndim != 2 or truth.shape[0] < 2:
        raise ValueError("V128 truth node-volume matrix must be [reference+candidate,node]")
    tfv = np.sum(truth, axis=1, dtype=np.float32).astype(np.float32, copy=False)
    delta = (tfv[1:] - tfv[0]).astype(np.float32, copy=False)
    return tfv, delta


def informative_pair_totals_float32(true_delta: np.ndarray, *, threshold: float) -> tuple[int, int, int]:
    values = np.asarray(true_delta, dtype=np.float32).reshape(-1)
    threshold32 = np.float32(threshold)
    reference = int(np.sum(np.abs(values) > threshold32))
    if len(values) < 2:
        return reference, 0, reference
    ii, jj = np.triu_indices(len(values), k=1)
    candidate = int(np.sum(np.abs(values[ii] - values[jj]) > threshold32))
    return reference, candidate, reference + candidate


def activate_current_truth_partition() -> None:
    """Install the canonical predicate into the audited exact implementation.

    The current stable entrypoint calls this before importing/running the staged trainer.
    Versioned archival runners do not receive the patch implicitly.
    """
    _exact._informative_pair_totals = informative_pair_totals_float32


def train_objective_stage_streaming_v128(*args, **kwargs):
    activate_current_truth_partition()
    return _exact.train_objective_stage_streaming_v128(*args, **kwargs)


__all__ = [
    "V128_OBJECTIVE_TRAINING_CONTRACT",
    "V128_TRUTH_PARTITION_CONTRACT",
    "activate_current_truth_partition",
    "canonical_float32_tfv_and_delta",
    "informative_pair_totals_float32",
    "train_objective_stage_streaming_v128",
]
