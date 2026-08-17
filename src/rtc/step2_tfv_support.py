"""Training-support diagnostics for the Project7 Direct-TFV controller.

The controller is allowed to learn all 109 writable facilities, but online optimisation must not
silently leave the action distribution represented by authoritative SWMM counterfactuals. This
module derives exact single-facility coverage plus empirical first-move, sequence-magnitude and
joint-density support from TrainFit data only.

The original V1 contract is retained for backward-compatible checkpoint loading. The additive
``DIRECT_TFV_JOINT_DENSITY_SUPPORT_V2`` fields expose q95/q99/max joint changed-facility counts so a
new MPC can expand beyond q90 only when the *training data itself* supports denser coordinated
moves. No label, objective or safety threshold is changed.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


DIRECT_TFV_ACTION_SUPPORT_CONTRACT = "PROJECT7_DIRECT_TFV_109ACT_ACTION_SUPPORT_V1"
DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT = "PROJECT7_DIRECT_TFV_JOINT_DENSITY_SUPPORT_V2"


def _q(values: Sequence[float], quantile: float, *, default: float = 0.0) -> float:
    data = np.asarray(tuple(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.quantile(data, quantile)) if data.size else float(default)


def derive_direct_tfv_action_support(
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    *,
    actuator_ids: Sequence[str],
    control_block_steps: int = 2,
    epsilon: float = 1.0e-7,
) -> dict[str, Any]:
    """Derive TrainFit-only action support and facility-identifiability evidence.

    ``settings`` are expected on the frozen H72 five-minute grid. Two model steps form one ten-
    minute control block. All returned per-facility arrays follow ``actuator_ids`` order.
    """

    ids = tuple(str(value) for value in actuator_ids)
    if len(ids) != 109 or len(set(ids)) != 109:
        raise ValueError("Direct-TFV action support requires exactly 109 unique actuators")
    if int(control_block_steps) <= 0:
        raise ValueError("control_block_steps must be positive")

    single_counts = np.zeros(len(ids), dtype=np.int64)
    single_rainfall: list[set[str]] = [set() for _ in ids]
    first_move_values: list[list[float]] = [[] for _ in ids]
    sequence_values: list[list[float]] = [[] for _ in ids]
    changed_counts: list[int] = []
    joint_changed_counts: list[int] = []
    absolute_delta_tfv: list[float] = []
    group_count = branch_count = single_branch_count = joint_branch_count = 0

    for source, names in source_groups.items():
        cache = source_caches[source]
        for name in names:
            entry = cache.entry(name)
            arrays = entry.arrays
            ref = int(entry.reference_index)
            reference = np.asarray(arrays["settings"][ref], dtype=np.float64)
            if reference.ndim != 2 or reference.shape[1] != len(ids):
                raise ValueError(f"{name}: settings do not match Hx109")
            if reference.shape[0] % int(control_block_steps):
                raise ValueError(f"{name}: settings horizon is not divisible by control block")
            ref_tfv = float(
                np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum()
            )
            candidates = [int(index) for index in entry.indices if int(index) != ref]
            if not candidates:
                continue
            group_count += 1
            for index in candidates:
                candidate = np.asarray(arrays["settings"][index], dtype=np.float64)
                delta = np.abs(candidate - reference)
                changed = np.any(delta > float(epsilon), axis=0)
                changed_index = np.flatnonzero(changed)
                if changed_index.size <= 0:
                    continue
                branch_count += 1
                changed_counts.append(int(changed_index.size))
                blocks = delta.reshape(
                    delta.shape[0] // int(control_block_steps), int(control_block_steps), len(ids)
                ).mean(axis=1)
                for actuator_index in changed_index.tolist():
                    first = float(blocks[0, actuator_index])
                    maximum = float(np.max(blocks[:, actuator_index]))
                    if first > float(epsilon):
                        first_move_values[actuator_index].append(first)
                    if maximum > float(epsilon):
                        sequence_values[actuator_index].append(maximum)
                if changed_index.size == 1:
                    actuator_index = int(changed_index[0])
                    single_counts[actuator_index] += 1
                    single_rainfall[actuator_index].add(str(entry.rainfall_group))
                    single_branch_count += 1
                else:
                    joint_branch_count += 1
                    joint_changed_counts.append(int(changed_index.size))
                candidate_tfv = float(
                    np.asarray(arrays["exact_node_flood_volume_m3"][index], dtype=np.float64).sum()
                )
                effect = abs(candidate_tfv - ref_tfv)
                if math.isfinite(effect):
                    absolute_delta_tfv.append(effect)

    if branch_count <= 0 or not changed_counts:
        raise ValueError("Direct-TFV action support found no changed training branches")

    first_q95: list[float] = []
    sequence_q95: list[float] = []
    first_counts: list[int] = []
    sequence_counts: list[int] = []
    for first, sequence in zip(first_move_values, sequence_values, strict=True):
        first_counts.append(len(first))
        sequence_counts.append(len(sequence))
        sequence_radius = _q(sequence, 0.95, default=0.0)
        first_radius = _q(first, 0.95, default=sequence_radius)
        first_q95.append(float(max(first_radius, 0.0)))
        sequence_q95.append(float(max(sequence_radius, 0.0)))

    covered = single_counts > 0
    changed_array = np.asarray(changed_counts, dtype=np.float64)
    joint_changed_array = np.asarray(joint_changed_counts, dtype=np.float64)
    abs_tfv = np.asarray(absolute_delta_tfv, dtype=np.float64)
    joint_default = float(np.quantile(changed_array, 0.50))

    def changed_quantile(q: float) -> float:
        return float(np.quantile(changed_array, q))

    def joint_quantile(q: float) -> float:
        if joint_changed_array.size:
            return float(np.quantile(joint_changed_array, q))
        return joint_default

    return {
        "contract": DIRECT_TFV_ACTION_SUPPORT_CONTRACT,
        "joint_density_extension_contract": DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT,
        "actuator_ids": list(ids),
        "training_group_count": int(group_count),
        "training_branch_count": int(branch_count),
        "single_branch_count": int(single_branch_count),
        "joint_branch_count": int(joint_branch_count),
        "single_facility_coverage_count": int(np.sum(covered)),
        "single_facility_coverage_fraction": float(np.mean(covered)),
        "uncovered_facilities": [ids[index] for index in np.flatnonzero(~covered).tolist()],
        "single_branch_count_per_facility": single_counts.astype(int).tolist(),
        "single_rainfall_group_count_per_facility": [len(value) for value in single_rainfall],
        "first_move_nonzero_count_per_facility": first_counts,
        "sequence_nonzero_count_per_facility": sequence_counts,
        "first_move_abs_q95_per_facility": first_q95,
        "sequence_abs_q95_per_facility": sequence_q95,
        "changed_facility_count_q50": changed_quantile(0.50),
        "changed_facility_count_q75": changed_quantile(0.75),
        "changed_facility_count_q90": changed_quantile(0.90),
        "changed_facility_count_q95": changed_quantile(0.95),
        "changed_facility_count_q99": changed_quantile(0.99),
        "changed_facility_count_max": int(np.max(changed_array)),
        "joint_changed_facility_count_q50": joint_quantile(0.50),
        "joint_changed_facility_count_q75": joint_quantile(0.75),
        "joint_changed_facility_count_q90": joint_quantile(0.90),
        "joint_changed_facility_count_q95": joint_quantile(0.95),
        "joint_changed_facility_count_q99": joint_quantile(0.99),
        "joint_changed_facility_count_max": (
            int(np.max(joint_changed_array)) if joint_changed_array.size else int(round(joint_default))
        ),
        "absolute_delta_tfv_q95_m3": float(np.quantile(abs_tfv, 0.95)) if abs_tfv.size else 0.0,
        "absolute_delta_tfv_max_m3": float(np.max(abs_tfv)) if abs_tfv.size else 0.0,
        "source_groups": {key: int(len(value)) for key, value in source_groups.items()},
        "scientific_role": (
            "TrainFit identifiability evidence and Step3 trust-region geometry only; q95/q99/max "
            "are support diagnostics, not performance-tuned thresholds"
        ),
    }


__all__ = [
    "DIRECT_TFV_ACTION_SUPPORT_CONTRACT",
    "DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT",
    "derive_direct_tfv_action_support",
]
