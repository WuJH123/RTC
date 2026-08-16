"""Shared construction/lineage helpers for current counterfactual-first V128 Development.

Training and smoke/dev audits rebuild the same model class and bind the same frozen edge artifact
plus source contracts. Audit construction may use unit placeholder numerical scales because the
source-strict stage checkpoint restores trained temporal/direct scale buffers immediately.
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
from .step2_counterfactual_training_v5 import (
    COUNTERFACTUAL_B0_V5_CONTRACT,
    POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT,
)
from .step2_oracle_isolation_v128 import ORACLE_FLOW_ISOLATION_CONTRACT
from .step2_stagea_balanced_v128 import (
    BALANCED_A0_CONTRACT,
    BALANCED_A1_CONTRACT,
    BALANCED_A2_CONTRACT,
    BALANCED_STAGE_A_CONTRACT,
    DIRECT_PAIR_SAMPLING_CONTRACT,
)

CURRENT_DEV_CONTEXT_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_DEV_CONTEXT_V7_BALANCED_STAGE_A"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_identifiable_source_sha256() -> str:
    digest = hashlib.sha256()
    for module_name in (
        "rtc.step2_counterfactual_first_v128",
        "rtc.step2_counterfactual_training_v5",
        "rtc.step2_stagea_balanced_v128",
        "rtc.step2_oracle_isolation_v128",
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
            "counterfactual_first_model_contract": COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
            "counterfactual_first_training_contract": COUNTERFACTUAL_FIRST_TRAINING_CONTRACT,
            "direct_action_flow_scale_contract": DIRECT_ACTION_FLOW_SCALE_CONTRACT,
            "oracle_flow_isolation_contract": ORACLE_FLOW_ISOLATION_CONTRACT,
            "counterfactual_stage_a_contract": BALANCED_STAGE_A_CONTRACT,
            "direct_flow_a0_contract": BALANCED_A0_CONTRACT,
            "oracle_hydraulic_a1_contract": BALANCED_A1_CONTRACT,
            "joint_direct_a2_contract": BALANCED_A2_CONTRACT,
            "direct_pair_sampling_contract": DIRECT_PAIR_SAMPLING_CONTRACT,
            "counterfactual_b0_contract": COUNTERFACTUAL_B0_V5_CONTRACT,
            "post_objective_trajectory_anchor_contract": POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT,
            "explicit_lazy_stage_a_b0_anchor": True,
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
