"""Train-only, rebuildable numerical-stability contracts for Step2.

This module contains no SWMM execution and no scientific-data selection logic.  The
scale derivation deliberately consumes only the V6 shard manifest supplied by the
caller and fails closed unless every shard is explicitly marked development/train.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

STABILITY_AMENDMENT = "PROJECT7_V069_STEP2_STABILITY_AMENDMENT_V2"
STABILITY_MODEL_CONTRACT = "STEP2_BOUNDED_COUNTERFACTUAL_DYNAMICS_V2"
STABILITY_CACHE_CONTRACT = "STEP2_REBUILDABLE_TRAINING_CACHE_V1"
CURRICULUM_CONTRACT_V2 = "FLOW_H1_TO_H72_EXPLICIT_STABILITY_V2"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_payload(manifest: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        return json.loads(path.read_text(encoding="utf-8")), _sha256_file(path)
    payload = dict(manifest)
    manifest_sha = str(payload.get("manifest_sha256", ""))
    return payload, manifest_sha


def _rms_from_sums(sum_sq: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("cannot derive transition scale from an empty shard manifest")
    result = np.sqrt(np.asarray(sum_sq, dtype=np.float64) / float(count))
    if not np.isfinite(result).all():
        raise ValueError("non-finite transition RMS encountered")
    return result.astype(np.float32)


def derive_train_only_delta_scales(
    manifest: str | Path | dict[str, Any],
    *,
    state_std: np.ndarray,
    flow_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive bounded residual scales from Train-only V6 trajectory differences.

    State deltas are measured as ``target[0] - initial`` followed by consecutive
    target-state differences.  Flow deltas use the analogous previous-flow prefix.
    Only the six state channels and the 109 actuator channels are reduced; no
    Validation or Final outcome is consulted.
    """

    payload, manifest_sha = _manifest_payload(manifest)
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("train-only scale derivation requires non-empty shards")
    state_std = np.asarray(state_std, dtype=np.float64).reshape(-1)
    flow_std = np.asarray(flow_std, dtype=np.float64).reshape(-1)
    if state_std.size != 6:
        raise ValueError(f"expected six state standard deviations, got {state_std.size}")
    if flow_std.size not in (1, 109):
        raise ValueError(f"expected scalar or 109 actuator flow std, got {flow_std.size}")
    if not np.isfinite(state_std).all() or not np.isfinite(flow_std).all():
        raise ValueError("normalization statistics must be finite")

    state_sum_sq = np.zeros(6, dtype=np.float64)
    flow_sum_sq: np.ndarray | None = None
    state_count = 0
    flow_count = 0
    for item in shards:
        path = Path(str(item["path"]))
        if not path.is_file():
            raise ValueError(f"missing Train-only V6 shard: {path}")
        with np.load(path, allow_pickle=False) as raw:
            if "scientific_split" not in raw.files or "development_fold" not in raw.files:
                raise ValueError(f"shard lacks split provenance: {path}")
            split = {str(x).strip().lower() for x in raw["scientific_split"].tolist()}
            fold = {str(x).strip().lower() for x in raw["development_fold"].tolist()}
            if split != {"development"} or fold != {"train"}:
                raise ValueError(
                    "bounded Step2 scales may only use development/train shards; "
                    f"found split={sorted(split)}, fold={sorted(fold)} in {path}"
                )
            initial = np.asarray(raw["initial_state"], dtype=np.float64)
            states = np.asarray(raw["target_states"], dtype=np.float64)
            previous = np.asarray(raw["previous_actuator_flow"], dtype=np.float64)
            flows = np.asarray(raw["target_actuator_flows"], dtype=np.float64)
            if states.ndim != 4 or states.shape[-1] != 6:
                raise ValueError(f"invalid target_states shape in {path}: {states.shape}")
            if flows.ndim != 3:
                raise ValueError(f"invalid target_actuator_flows shape in {path}: {flows.shape}")
            if states.shape[0] != initial.shape[0] or flows.shape[0] != previous.shape[0]:
                raise ValueError(f"trajectory batch dimensions disagree in {path}")
            state_first = states[:, :1] - initial[:, None]
            state_rest = states[:, 1:] - states[:, :-1]
            state_delta = np.concatenate((state_first, state_rest), axis=1)
            flow_first = flows[:, :1] - previous[:, None]
            flow_rest = flows[:, 1:] - flows[:, :-1]
            flow_delta = np.concatenate((flow_first, flow_rest), axis=1)
            if not np.isfinite(state_delta).all() or not np.isfinite(flow_delta).all():
                raise ValueError(f"non-finite trajectory delta in {path}")
            state_sum_sq += np.square(state_delta).sum(axis=(0, 1, 2))
            state_count += int(np.prod(state_delta.shape[:3]))
            if flow_sum_sq is None:
                flow_sum_sq = np.zeros(flow_delta.shape[-1], dtype=np.float64)
            if flow_delta.shape[-1] != flow_sum_sq.size:
                raise ValueError("actuator count differs across V6 shards")
            flow_sum_sq += np.square(flow_delta).sum(axis=(0, 1))
            flow_count += int(np.prod(flow_delta.shape[:2]))

    assert flow_sum_sq is not None
    state_rms = _rms_from_sums(state_sum_sq, state_count)
    flow_rms = _rms_from_sums(flow_sum_sq, flow_count)
    state_floor = np.maximum(np.abs(state_std), 1e-6) * 0.01
    flow_floor_base = np.maximum(np.abs(flow_std), 1e-6) * 0.01
    flow_floor = (
        np.full_like(flow_rms, float(flow_floor_base[0]))
        if flow_floor_base.size == 1
        else flow_floor_base.astype(np.float32)
    )
    state_scale = np.maximum(6.0 * state_rms, state_floor).astype(np.float32)
    flow_scale = np.maximum(6.0 * flow_rms, flow_floor).astype(np.float32)
    details = {
        "contract": STABILITY_MODEL_CONTRACT,
        "amendment": STABILITY_AMENDMENT,
        "source_manifest_sha256": manifest_sha,
        "scientific_split": "development",
        "development_fold": "train",
        "shard_count": len(shards),
        "state_rms_delta": state_rms.tolist(),
        "flow_rms_delta": flow_rms.tolist(),
        "state_floor": state_floor.astype(np.float32).tolist(),
        "flow_floor": flow_floor.tolist(),
        "state_delta_scale": state_scale.tolist(),
        "flow_delta_scale": flow_scale.tolist(),
        "state_delta_definition": "target_states[0]-initial_state; target_states[t]-target_states[t-1]",
        "flow_delta_definition": "target_flows[0]-previous_flow; target_flows[t]-target_flows[t-1]",
    }
    return state_scale, flow_scale, details
