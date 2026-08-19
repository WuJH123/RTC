"""HOLD-relative joint sequence support for optimizer-consistent Direct-TFV MPC.

The pretrained Step2 keeps 109 hydraulic action channels. Current support derivation can additionally
receive a native supervisory mask and then computes changed-facility and sequence geometry only on
the enabled control subspace. This is label-independent and reuses existing D3 TrainFit actions, so
changing 109 model channels to 82 online control freedoms does not require new SWMM simulation.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


DIRECT_TFV_SEQUENCE_SUPPORT_CONTRACT = "PROJECT7_DIRECT_TFV_D3_HOLD_JOINT_SEQUENCE_SUPPORT_V1"
SEQUENCE_SUPPORT_METRICS = (
    "first_block_l1",
    "h120_l1",
    "h120_total_variation_l1",
)
SUPPORTED_SEQUENCE_QUANTILES = ("q90", "q95", "q99")


def _control_blocks(settings: np.ndarray, control_block_steps: int) -> np.ndarray:
    value = np.asarray(settings, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("Direct-TFV sequence support expects settings shaped [H, actuator]")
    if value.shape[0] % int(control_block_steps):
        raise ValueError("settings horizon is not divisible by the control block")
    return value.reshape(
        value.shape[0] // int(control_block_steps), int(control_block_steps), value.shape[1]
    ).mean(axis=1)


def _mask_array(supervisory_mask: Sequence[bool] | np.ndarray | None, actuator_count: int) -> np.ndarray:
    if supervisory_mask is None:
        return np.ones(int(actuator_count), dtype=bool)
    mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
    if mask.shape != (int(actuator_count),) or int(mask.sum()) <= 0:
        raise ValueError("sequence support supervisory mask has the wrong shape or no enabled channel")
    return mask


def direct_tfv_sequence_geometry(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    control_block_steps: int = 2,
    free_control_blocks: int = 12,
    supervisory_mask: Sequence[bool] | np.ndarray | None = None,
) -> dict[str, float]:
    candidate_blocks = _control_blocks(candidate, control_block_steps)
    reference_blocks = _control_blocks(reference, control_block_steps)
    if candidate_blocks.shape != reference_blocks.shape:
        raise ValueError("candidate/reference sequence shapes differ")
    if candidate_blocks.shape[0] < int(free_control_blocks):
        raise ValueError("sequence is shorter than the H120 free-control horizon")
    mask = _mask_array(supervisory_mask, candidate_blocks.shape[1])
    delta = (
        candidate_blocks[: int(free_control_blocks)]
        - reference_blocks[: int(free_control_blocks)]
    )[:, mask]
    first = float(np.abs(delta[0]).sum())
    h120 = float(np.abs(delta).sum())
    previous = np.concatenate((np.zeros_like(delta[:1]), delta[:-1]), axis=0)
    temporal_tv = float(np.abs(delta - previous).sum())
    return {
        "first_block_l1": first,
        "h120_l1": h120,
        "h120_total_variation_l1": temporal_tv,
    }


def _quantile_fields(values: Sequence[float], prefix: str) -> dict[str, float]:
    data = np.asarray(tuple(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size <= 0:
        raise ValueError(f"Direct-TFV sequence support has no finite values for {prefix}")
    return {
        f"{prefix}_q90": float(np.quantile(data, 0.90)),
        f"{prefix}_q95": float(np.quantile(data, 0.95)),
        f"{prefix}_q99": float(np.quantile(data, 0.99)),
        f"{prefix}_max": float(np.max(data)),
    }


def derive_direct_tfv_sequence_support(
    cache: Any,
    names: Sequence[str],
    *,
    actuator_ids: Sequence[str],
    control_block_steps: int = 2,
    free_control_blocks: int = 12,
    epsilon: float = 1.0e-7,
    supervisory_mask: Sequence[bool] | np.ndarray | None = None,
    supervisory_control_contract: str | None = None,
    supervisory_mask_sha256: str | None = None,
) -> dict[str, Any]:
    ids = tuple(str(value) for value in actuator_ids)
    if len(ids) != 109 or len(set(ids)) != 109:
        raise ValueError("Direct-TFV sequence support requires 109 unique model channels")
    if int(control_block_steps) <= 0 or int(free_control_blocks) <= 0:
        raise ValueError("control-block settings must be positive")
    if not names:
        raise ValueError("Direct-TFV sequence support requires D3 TrainFit groups")
    mask = _mask_array(supervisory_mask, len(ids))

    metrics: dict[str, list[float]] = {key: [] for key in SEQUENCE_SUPPORT_METRICS}
    changed_counts: list[float] = []
    rainfall_groups: set[str] = set()
    group_count = 0
    joint_branch_count = 0
    maximum_reference_drift = 0.0

    for name in names:
        if not str(name).startswith("D3::"):
            raise ValueError(f"sequence support received non-D3 group: {name}")
        entry = cache.entry(name)
        arrays = entry.arrays
        reference_index = int(entry.reference_index)
        reference = np.asarray(arrays["settings"][reference_index], dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != len(ids):
            raise ValueError(f"{name}: reference settings do not match Hx109")
        reference_blocks = _control_blocks(reference, control_block_steps)
        drift = float(np.max(np.abs(reference_blocks - reference_blocks[:1])))
        maximum_reference_drift = max(maximum_reference_drift, drift)
        if drift > float(epsilon):
            raise ValueError(f"{name}: D3 reference is not HOLD-constant across H360")
        group_count += 1
        rainfall_groups.add(str(entry.rainfall_group))
        for index in entry.indices:
            index = int(index)
            if index == reference_index:
                continue
            candidate = np.asarray(arrays["settings"][index], dtype=np.float64)
            candidate_blocks = _control_blocks(candidate, control_block_steps)
            free_delta = (
                candidate_blocks[: int(free_control_blocks)]
                - reference_blocks[: int(free_control_blocks)]
            )[:, mask]
            changed = np.any(np.abs(free_delta) > float(epsilon), axis=0)
            changed_count = int(np.sum(changed))
            if changed_count <= 1:
                continue
            geometry = direct_tfv_sequence_geometry(
                candidate,
                reference,
                control_block_steps=control_block_steps,
                free_control_blocks=free_control_blocks,
                supervisory_mask=mask,
            )
            joint_branch_count += 1
            changed_counts.append(float(changed_count))
            for key in SEQUENCE_SUPPORT_METRICS:
                metrics[key].append(float(geometry[key]))

    if group_count <= 0 or joint_branch_count <= 0:
        raise ValueError("D3 TrainFit contains no usable multi-facility branches on the control subspace")

    mask_bytes = np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    mask_sha = hashlib.sha256(mask_bytes).hexdigest()
    if supervisory_mask_sha256 is not None and str(supervisory_mask_sha256).lower() != mask_sha:
        raise ValueError("provided supervisory-mask SHA does not match the support mask")

    payload: dict[str, Any] = {
        "contract": DIRECT_TFV_SEQUENCE_SUPPORT_CONTRACT,
        "development_only": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "source_semantics": "D3_TRAINFIT_ONLY",
        "label_independent": True,
        "actuator_ids": list(ids),
        "model_action_channel_count": len(ids),
        "supervisory_control_dimension": int(mask.sum()),
        "supervisory_mask": mask.astype(int).tolist(),
        "supervisory_mask_sha256": mask_sha,
        "supervisory_control_contract": str(supervisory_control_contract or "LEGACY_ALL_CHANNELS"),
        "control_block_steps": int(control_block_steps),
        "free_control_blocks": int(free_control_blocks),
        "group_count": int(group_count),
        "rainfall_group_count": int(len(rainfall_groups)),
        "joint_branch_count": int(joint_branch_count),
        "maximum_reference_drift": float(maximum_reference_drift),
    }
    payload.update(_quantile_fields(changed_counts, "joint_changed_facility_count"))
    for key in SEQUENCE_SUPPORT_METRICS:
        payload.update(_quantile_fields(metrics[key], key))
    payload["scientific_role"] = (
        "D3 TrainFit HOLD-reference joint temporal action geometry recomputed on the allowed online "
        "control subspace. It is label-independent and does not require new SWMM truth."
    )
    return payload


def validate_direct_tfv_sequence_support(
    payload: Mapping[str, Any],
    *,
    actuator_ids: Sequence[str],
    step2_checkpoint_sha256: str | None = None,
    supervisory_mask: Sequence[bool] | np.ndarray | None = None,
    supervisory_control_contract: str | None = None,
) -> None:
    if str(payload.get("contract")) != DIRECT_TFV_SEQUENCE_SUPPORT_CONTRACT:
        raise ValueError("wrong Direct-TFV joint-sequence support contract")
    if payload.get("development_only") is not True or payload.get("label_independent") is not True:
        raise ValueError("Direct-TFV sequence support must be label-independent Development evidence")
    if str(payload.get("reference_semantics", "")) != "HOLD_ACTIVE_TARGET_H360":
        raise ValueError("Direct-TFV sequence support has the wrong reference semantics")
    ids = tuple(str(value) for value in payload.get("actuator_ids", ()))
    expected = tuple(str(value) for value in actuator_ids)
    if ids != expected or len(ids) != 109:
        raise ValueError("Direct-TFV sequence-support actuator order differs from the graph")
    if int(payload.get("control_block_steps", -1)) != 2 or int(payload.get("free_control_blocks", -1)) != 12:
        raise ValueError("Direct-TFV sequence support does not match the frozen 10-min/H120 contract")
    if int(payload.get("joint_branch_count", 0)) <= 0:
        raise ValueError("Direct-TFV sequence support contains no joint branches")

    if supervisory_mask is not None:
        mask = _mask_array(supervisory_mask, len(ids))
        stored = np.asarray(payload.get("supervisory_mask", ()), dtype=np.int64).reshape(-1)
        if stored.shape != (109,) or not np.array_equal(stored.astype(bool), mask):
            raise ValueError("Direct-TFV sequence support was not built on the current supervisory mask")
        expected_sha = hashlib.sha256(
            np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
        ).hexdigest()
        if str(payload.get("supervisory_mask_sha256", "")).lower() != expected_sha:
            raise ValueError("Direct-TFV sequence support supervisory-mask SHA mismatch")
        if int(payload.get("supervisory_control_dimension", -1)) != int(mask.sum()):
            raise ValueError("Direct-TFV sequence support control dimension mismatch")
    if supervisory_control_contract is not None:
        if str(payload.get("supervisory_control_contract", "")) != str(supervisory_control_contract):
            raise ValueError("Direct-TFV sequence support uses another supervisory-control contract")

    for metric in ("joint_changed_facility_count", *SEQUENCE_SUPPORT_METRICS):
        maximum = float(payload.get(f"{metric}_max", float("nan")))
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(f"Direct-TFV sequence support has invalid {metric}_max")
        for quantile in SUPPORTED_SEQUENCE_QUANTILES:
            value = float(payload.get(f"{metric}_{quantile}", float("nan")))
            if not np.isfinite(value) or value <= 0.0 or value > maximum + 1.0e-9:
                raise ValueError(f"Direct-TFV sequence support has invalid {metric}_{quantile}")
    if step2_checkpoint_sha256 is not None:
        lineage = payload.get("lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("Direct-TFV sequence support lacks checkpoint lineage")
        trained = str(lineage.get("step2_checkpoint_sha256", "")).lower()
        if trained != str(step2_checkpoint_sha256).lower():
            raise ValueError("Direct-TFV sequence support was built for a different Step2 checkpoint")


def sequence_support_limit(payload: Mapping[str, Any], metric: str, quantile: str) -> float:
    if metric not in SEQUENCE_SUPPORT_METRICS:
        raise ValueError(f"unknown Direct-TFV sequence-support metric: {metric}")
    if quantile not in SUPPORTED_SEQUENCE_QUANTILES:
        raise ValueError(f"unsupported Direct-TFV sequence-support quantile: {quantile}")
    value = float(payload[f"{metric}_{quantile}"])
    maximum = float(payload[f"{metric}_max"])
    return float(min(value, maximum))


def changed_facility_support_limit(payload: Mapping[str, Any], quantile: str = "q95") -> int:
    if quantile not in SUPPORTED_SEQUENCE_QUANTILES:
        raise ValueError(f"unsupported changed-facility support quantile: {quantile}")
    value = float(payload[f"joint_changed_facility_count_{quantile}"])
    maximum = float(payload["joint_changed_facility_count_max"])
    dimension = int(payload.get("supervisory_control_dimension", 109))
    return max(1, min(dimension, int(np.ceil(min(value, maximum)))))


__all__ = [
    "DIRECT_TFV_SEQUENCE_SUPPORT_CONTRACT",
    "SEQUENCE_SUPPORT_METRICS",
    "SUPPORTED_SEQUENCE_QUANTILES",
    "changed_facility_support_limit",
    "derive_direct_tfv_sequence_support",
    "direct_tfv_sequence_geometry",
    "sequence_support_limit",
    "validate_direct_tfv_sequence_support",
]
