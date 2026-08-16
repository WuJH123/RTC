"""Shared construction/lineage helpers for the current counterfactual-first V128 Development model.

Training and every smoke/dev audit must rebuild the same model class and bind the same frozen
edge artifact plus source contracts. Audit construction may use unit placeholder numerical
scales because source-strict stage loading immediately restores the trained temporal/direct
scale buffers from the checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .edge_physics_current_v128 import load_edge_physics_artifact_v128
from .step2_counterfactual_first_v128 import (
    COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
    COUNTERFACTUAL_FIRST_TRAINING_CONTRACT,
    DIRECT_ACTION_FLOW_SCALE_CONTRACT,
    build_counterfactual_first_v128_model_from_graph,
)
from .step2_counterfactual_training_v3 import (
    COUNTERFACTUAL_STAGE_A_V3_CONTRACT,
    DIRECT_FLOW_A0_V3_CONTRACT,
    JOINT_DIRECT_A2_V3_CONTRACT,
    ORACLE_HYDRAULIC_A1_V3_CONTRACT,
)

CURRENT_DEV_CONTEXT_CONTRACT = "PROJECT7_V128_COUNTERFACTUAL_FIRST_DEV_CONTEXT_V4_STAGE_A_V3"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_identifiable_source_sha256() -> str:
    """Match the current training wrapper's enhanced-source fingerprint exactly."""
    digest = hashlib.sha256()
    for module_name in (
        "rtc.step2_counterfactual_first_v128",
        "rtc.step2_counterfactual_training_v3",
        "rtc.step2_action_identifiable_v128",
        "rtc.step2_action_flow_warmup_v128",
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
            "counterfactual_first_model_contract": COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
            "counterfactual_first_training_contract": COUNTERFACTUAL_FIRST_TRAINING_CONTRACT,
            "direct_action_flow_scale_contract": DIRECT_ACTION_FLOW_SCALE_CONTRACT,
            "counterfactual_stage_a_contract": COUNTERFACTUAL_STAGE_A_V3_CONTRACT,
            "direct_flow_a0_contract": DIRECT_FLOW_A0_V3_CONTRACT,
            "oracle_hydraulic_a1_contract": ORACLE_HYDRAULIC_A1_V3_CONTRACT,
            "joint_direct_a2_contract": JOINT_DIRECT_A2_V3_CONTRACT,
            "gradient_is_training_target": False,
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
    direct_action_flow_scale: np.ndarray | None = None,
    design: Any = None,
):
    artifact = load_edge_physics_artifact_v128(str(edge_physics_path), graph)
    direct = (
        np.ones(len(graph.actuator_ids), dtype=np.float32)
        if direct_action_flow_scale is None
        else np.asarray(direct_action_flow_scale, dtype=np.float32)
    )
    kwargs = {
        "edge_artifact": artifact,
        "state_dim": int(state_dim),
        "rainfall_dim": int(rainfall_dim),
        "delta_state_scale": np.asarray(delta_state_scale, dtype=np.float32),
        "delta_flow_scale": np.asarray(delta_flow_scale, dtype=np.float32),
        "direct_action_flow_scale": direct,
    }
    if design is not None:
        kwargs["design"] = design
    return build_counterfactual_first_v128_model_from_graph(graph, **kwargs)


__all__ = [
    "CURRENT_DEV_CONTEXT_CONTRACT",
    "action_identifiable_source_sha256",
    "build_current_action_identifiable_model",
    "extend_action_identifiable_stage_lineage",
    "sha256_file",
]
