"""Shared construction/lineage helpers for the current action-identifiable V128 Development model.

Training and every smoke/dev audit must reconstruct the same model class and must include the
same edge-physics artifact/source fingerprints in stage-checkpoint lineage.  Keeping this logic
in one module prevents the historical failure where training advanced but audit entrypoints
silently rebuilt an older V128 architecture.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .edge_physics_current_v128 import load_edge_physics_artifact_v128
from .step2_action_identifiable_v128 import (
    ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
    ACTION_IDENTIFIABLE_MODEL_CONTRACT,
    ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
    build_action_identifiable_v128_model_from_graph,
)

CURRENT_DEV_CONTEXT_CONTRACT = "PROJECT7_V128_ACTION_IDENTIFIABLE_DEV_CONTEXT_V1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_identifiable_source_sha256() -> str:
    """Match the training wrapper's enhanced-source fingerprint exactly."""
    digest = hashlib.sha256()
    for module_name in (
        "rtc.step2_action_identifiable_v128",
        "rtc.step2_differentiable_v128_edge",
        "rtc.edge_physics_current_v128",
    ):
        module = importlib.import_module(module_name)
        raw = getattr(module, "__file__", None)
        if raw is None:
            raise RuntimeError(f"cannot fingerprint current Development module: {module_name}")
        path = Path(raw)
        digest.update(module_name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def extend_action_identifiable_stage_lineage(
    lineage: Mapping[str, Any], *, edge_physics_path: str | Path
) -> dict[str, Any]:
    result = dict(lineage)
    result.update(
        {
            "edge_physics_sha256": sha256_file(edge_physics_path),
            "action_identifiable_source_sha256": action_identifiable_source_sha256(),
            "action_identifiable_model_contract": ACTION_IDENTIFIABLE_MODEL_CONTRACT,
            "action_identifiable_training_contract": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
            "flow_scale_contract": ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
        }
    )
    return result


def build_current_action_identifiable_model(
    graph: Any,
    *,
    edge_physics_path: str | Path,
    state_dim: int,
    rainfall_dim: int,
    delta_state_scale: np.ndarray,
    delta_flow_scale: np.ndarray,
    design: Any = None,
):
    artifact = load_edge_physics_artifact_v128(str(edge_physics_path), graph)
    kwargs = {
        "edge_artifact": artifact,
        "state_dim": int(state_dim),
        "rainfall_dim": int(rainfall_dim),
        "delta_state_scale": np.asarray(delta_state_scale, dtype=np.float32),
        "delta_flow_scale": np.asarray(delta_flow_scale, dtype=np.float32),
    }
    if design is not None:
        kwargs["design"] = design
    return build_action_identifiable_v128_model_from_graph(graph, **kwargs)


__all__ = [
    "CURRENT_DEV_CONTEXT_CONTRACT",
    "action_identifiable_source_sha256",
    "build_current_action_identifiable_model",
    "extend_action_identifiable_stage_lineage",
    "sha256_file",
]
