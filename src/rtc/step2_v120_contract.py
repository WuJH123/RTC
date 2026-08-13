"""Final simplified Project7 Step2 contract and lineage helpers.

V12.0 is deliberately control-oriented.  Step2 predicts the signed authoritative
change in whole-system cumulative TFV for a *joint* actuator sequence.  A nodewise
counterfactual Hydraulic surrogate is neither an online dependency nor an
acceptance gate.

Two hashes are intentionally separated:

* :func:`rtc_implementation_contract_sha256` in ``code_contract`` is the legacy
  project-wide scientific contract and remains stable for old Step1 checkpoints;
* :func:`v120_runtime_contract_sha256` below binds the exact files that determine
  V120 training, candidate generation, online scoring and first-move execution.

This prevents a V120 bundle from being silently reused after the value policy or
execution semantics change, without needlessly invalidating the frozen Step1
checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

V120_CONTRACT = "PROJECT7_STEP2_V120_TFV_VALUE_ONLY_V2_EXECUTION_BOUND"
V120_BUNDLE_CONTRACT = "PROJECT7_STEP2_V120_TFV_VALUE_ONLY_BUNDLE_V2"
V120_CANDIDATE_POLICY_CONTRACT = "D3_V2_DISTRIBUTION_MATCHED_FINITE_CANDIDATE_POLICY_V2_EXECUTABLE"
V120_DATA_ELIGIBILITY_CONTRACT = "PROJECT7_V120_FROZEN_18TRAIN_ONLY_V1"
V120_GRAPH_CONTRACT = "PROJECT7_V120_GRAPH_SEMANTIC_IDENTITY_V1"

# These are source-census facts, not an instruction to pool all rows as IID
# training samples.  D2 source membership is still filtered by the frozen 18/6/6
# event split before V120 sees any label.
SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS = 4800
TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS = 3600

_V120_BOUND_FILES = (
    "controller.py",
    "runtime.py",
    "runtime_controller_guard.py",
    "step2_control_basis_v60.py",
    "step2_control_response_v70.py",
    "step2_d3_design_v60.py",
    "step2_optimization_v70.py",
    "step2_policy_v120.py",
    "step2_runtime_v120.py",
    "step2_train_response_v60.py",
    "step2_train_response_v70.py",
    "step2_v70_contract.py",
    "production_v120.py",
    "production_v120_router.py",
    "production_cli_router.py",
    "production_guard.py",
)


def v120_runtime_contract_sha256() -> str:
    """Hash the exact V120 training/search/execution implementation.

    The hash is content-addressed rather than a manually bumped version string.
    Any edit to a bound file invalidates an old V120 bundle fail-closed.
    """

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(V120_CONTRACT.encode("utf-8"))
    digest.update(V120_BUNDLE_CONTRACT.encode("utf-8"))
    for name in _V120_BOUND_FILES:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"V120 bound implementation file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _hash_string_sequence(digest: "hashlib._Hash", name: str, values: Any) -> None:
    digest.update(name.encode("utf-8"))
    canonical = json.dumps([str(x) for x in values], ensure_ascii=False, separators=(",", ":"))
    digest.update(canonical.encode("utf-8"))


def _hash_array(digest: "hashlib._Hash", name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))


def v120_graph_semantic_sha256(graph: Any) -> str:
    """Content identity for every graph field consumed by Step1/V120 control.

    A raw NPZ file hash is still recorded for provenance.  This semantic identity
    additionally protects runtime callers that already hold a ``GraphSchema`` and
    therefore no longer know the original file bytes.
    """

    digest = hashlib.sha256()
    digest.update(V120_GRAPH_CONTRACT.encode("utf-8"))
    _hash_string_sequence(digest, "node_ids", graph.node_ids)
    _hash_array(digest, "edge_index", graph.edge_index)
    _hash_array(digest, "static_node_features", graph.static_node_features)
    _hash_string_sequence(digest, "static_node_feature_names", graph.static_node_feature_names)
    _hash_string_sequence(digest, "actuator_ids", graph.actuator_ids)
    _hash_array(digest, "actuator_upstream", graph.actuator_upstream)
    _hash_array(digest, "actuator_downstream", graph.actuator_downstream)
    _hash_array(digest, "actuator_physics", graph.actuator_physics)
    _hash_string_sequence(digest, "actuator_physics_feature_names", graph.actuator_physics_feature_names)
    digest.update(str(graph.system_units).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Step2V120Contract:
    actuator_count: int = 109
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    value_horizon_minutes: int = 360
    control_blocks: int = 36
    candidates_per_bank: int = 24
    candidate_banks: int = 4
    seed: int = 42
    primary_objective: str = "whole_system_cumulative_TFV_m3"
    step2_target: str = "direct_signed_authoritative_delta_TFV_m3"
    hydraulic_model_required_online: bool = False
    nodewise_action_effect_required: bool = False
    sum_d2_effects_for_joint_action: bool = False
    execute_first_move_only: bool = True
    training_event_count: int = 18
    internal_holdout_event_count: int = 4

    def validate(self) -> None:
        if self.actuator_count != 109:
            raise ValueError("V120 requires the frozen 109-actuator testbed")
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("V120 frozen clock is 300-s model / 600-s control")
        if self.value_horizon_minutes != 360 or self.control_blocks != 36:
            raise ValueError("V120 direct value horizon must remain 360 min / 36 blocks")
        if self.candidates_per_bank != 24 or self.candidate_banks < 1:
            raise ValueError("V120 candidate policy must retain D3-v2 24-candidate banks")
        if self.seed != 42:
            raise ValueError("V120 scientific seed drift")
        if self.primary_objective != "whole_system_cumulative_TFV_m3":
            raise ValueError("V120 primary objective drift")
        if self.step2_target != "direct_signed_authoritative_delta_TFV_m3":
            raise ValueError("V120 Step2 target drift")
        if self.hydraulic_model_required_online or self.nodewise_action_effect_required:
            raise ValueError("V120 must not re-introduce nodewise Hydraulic as a production gate")
        if self.sum_d2_effects_for_joint_action:
            raise ValueError("V120 forbids SUM-D2 joint-action reconstruction")
        if not self.execute_first_move_only:
            raise ValueError("V120 must remain receding-horizon first-move control")
        if (self.training_event_count, self.internal_holdout_event_count) != (18, 4):
            raise ValueError("V120 requires frozen Train18 with four-event internal holdout")


__all__ = [
    "SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS",
    "TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS",
    "Step2V120Contract",
    "V120_BUNDLE_CONTRACT",
    "V120_CANDIDATE_POLICY_CONTRACT",
    "V120_CONTRACT",
    "V120_DATA_ELIGIBILITY_CONTRACT",
    "V120_GRAPH_CONTRACT",
    "v120_graph_semantic_sha256",
    "v120_runtime_contract_sha256",
]
