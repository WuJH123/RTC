"""Fail-closed reconstruction of the frozen V7 Value + V11 Hydraulic runtime stack."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .graph import GraphSchema
from .step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from .step2_control_response_v60 import prepare_static_v60
from .step2_control_response_v70 import ControlValueSurrogateV70, HydraulicResponseSurrogateV70
from .step2_control_response_v110 import (
    ActuatorSetHydraulicResponseV110,
    build_actuator_node_relations_v110,
)
from .step2_mpc_v110 import RuntimeNormalizationV110, V7V11RollingMPC
from .step2_v60_contract import MultiResolutionHorizonV60
from .step2_v70_contract import V70_CONTRACT
from .step2_v110_contract import (
    ActuatorSetHydraulicContractV110,
    HydraulicHorizonV110,
    V110_CONTRACT,
)

V110_RUNTIME_BUNDLE_CONTRACT = "PROJECT7_STEP2_V110_RUNTIME_BUNDLE_V1"


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} must contain a dictionary")
    return payload


def _require_v7(payload: dict[str, Any], kind: str) -> None:
    if payload.get("contract") != V70_CONTRACT or payload.get("kind") != kind:
        raise ValueError(f"runtime requires frozen V7 {kind} checkpoint")
    if "state_dict" not in payload:
        raise ValueError(f"V7 {kind} checkpoint lacks state_dict")


def _require_v11(payload: dict[str, Any], *, allow_development: bool) -> None:
    if payload.get("contract") != V110_CONTRACT:
        raise ValueError("runtime requires current V11 hydraulic checkpoint")
    if payload.get("stage") != "d3":
        raise ValueError("runtime V11 hydraulic checkpoint must be D3 interaction stage")
    if "state_dict" not in payload or "input_normalization" not in payload:
        raise ValueError("V11 checkpoint predates self-describing runtime metadata; rerun V11")
    if not bool(payload.get("runtime_compatible", False)):
        raise ValueError("V11 checkpoint did not pass the runtime-compatibility development gate")
    if not allow_development and not bool(payload.get("production_compatible", False)):
        raise ValueError(
            "V11 checkpoint is development-only; production requires an explicitly promoted bundle"
        )


def _basis_matches_payload(basis, payload: dict[str, Any]) -> None:
    recorded = payload.get("basis_manifest")
    if not isinstance(recorded, dict):
        raise ValueError("V7 value checkpoint lacks basis_manifest")
    current = basis_manifest_v60(basis)
    keys = (
        "actuator_count", "control_group_count", "temporal_basis_count",
        "coefficient_dimension", "control_blocks", "model_horizon_steps",
        "group_id_by_actuator", "zone_id_by_actuator",
    )
    for key in keys:
        if recorded.get(key) != current.get(key):
            raise ValueError(f"V7 runtime control basis mismatch: {key}")
    if not np.allclose(
        np.asarray(recorded.get("temporal_basis"), dtype=np.float64),
        np.asarray(current["temporal_basis"], dtype=np.float64),
        atol=1.0e-7,
    ):
        raise ValueError("V7 runtime temporal basis differs from checkpoint")


def _build_v7_value(
    graph: GraphSchema,
    payload: dict[str, Any],
    normalization: RuntimeNormalizationV110,
    device: torch.device,
):
    _require_v7(payload, "control_value")
    basis = build_control_basis_v60(graph)
    _basis_matches_payload(basis, payload)
    prepared = prepare_static_v60(graph, device)
    if normalization.state_mean.shape[0] != 6:
        raise ValueError("V7/V11 runtime requires six-state input normalization")
    model = ControlValueSurrogateV70(
        state_dim=6,
        rainfall_dim=int(normalization.rainfall_mean.shape[0]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(payload["direct_tfv_scale_m3"]),
        hidden_dim=96,
        actuator_embedding_dim=16,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, basis, prepared


def _build_v7_reference(
    graph: GraphSchema,
    payload: dict[str, Any],
    normalization: RuntimeNormalizationV110,
    device: torch.device,
):
    _require_v7(payload, "hydraulic_response")
    prepared = prepare_static_v60(graph, device)
    state_dict = payload["state_dict"]
    if "state_scale" not in state_dict or "flow_scale" not in state_dict:
        raise ValueError("V7 hydraulic reference lacks physical scale buffers")
    model = HydraulicResponseSurrogateV70(
        state_scale=state_dict["state_scale"],
        flow_scale=state_dict["flow_scale"],
        horizon_contract=MultiResolutionHorizonV60(),
        state_dim=6,
        rainfall_dim=int(normalization.rainfall_mean.shape[0]),
        node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=64,
        latent_dim=32,
        temporal_dim=12,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_v7_v11_rolling_mpc(
    *,
    graph: GraphSchema,
    v70_value_checkpoint: str | Path,
    v70_hydraulic_checkpoint: str | Path,
    v110_hydraulic_checkpoint: str | Path,
    device: torch.device,
    priority_indices: torch.Tensor | None = None,
    allow_development: bool = False,
    tfv_cvar_alpha: float = 0.90,
    tfv_near_opt_relative: float = 0.01,
    tfv_near_opt_absolute_m3: float = 1.0,
    near_opt_penalty: float = 1.0e4,
    movement_tiebreak: float = 1.0e-6,
    min_predicted_tfv_improvement_m3: float = 0.0,
    hydraulic_secondary_iterations: int | None = None,
) -> V7V11RollingMPC:
    """Load a lineage-consistent V7/V11 stack and return the controller-facing MPC."""
    value_payload = _load_payload(v70_value_checkpoint)
    reference_payload = _load_payload(v70_hydraulic_checkpoint)
    v11_payload = _load_payload(v110_hydraulic_checkpoint)
    _require_v11(v11_payload, allow_development=allow_development)
    _require_v7(value_payload, "control_value")
    _require_v7(reference_payload, "hydraulic_response")
    if value_payload.get("split_manifest_sha256") != reference_payload.get("split_manifest_sha256"):
        raise ValueError("V7 value/reference checkpoints use different scientific splits")
    lineage = v11_payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V11 checkpoint lacks lineage")
    expected_value_sha = str(lineage.get("v70_value_checkpoint_sha256", ""))
    expected_reference_sha = str(lineage.get("v70_hydraulic_checkpoint_sha256", ""))
    from .inp_runtime import sha256_file
    if expected_value_sha and sha256_file(v70_value_checkpoint) != expected_value_sha:
        raise ValueError("V11 runtime value checkpoint SHA differs from training lineage")
    if expected_reference_sha and sha256_file(v70_hydraulic_checkpoint) != expected_reference_sha:
        raise ValueError("V11 runtime hydraulic-reference SHA differs from training lineage")

    normalization = RuntimeNormalizationV110.from_payload(v11_payload["input_normalization"])
    value_model, basis, prepared = _build_v7_value(
        graph, value_payload, normalization, device
    )
    reference_model = _build_v7_reference(
        graph, reference_payload, normalization, device
    )
    relations = build_actuator_node_relations_v110(graph)
    state_dict = v11_payload["state_dict"]
    if "state_magnitude_scale" not in state_dict or "flow_magnitude_scale" not in state_dict:
        raise ValueError("V11 checkpoint lacks local physical magnitude scales")
    v11 = ActuatorSetHydraulicResponseV110(
        reference_model=reference_model,
        state_magnitude_scale=state_dict["state_magnitude_scale"],
        flow_magnitude_scale=state_dict["flow_magnitude_scale"],
        node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        rainfall_dim=int(normalization.rainfall_mean.shape[0]),
        actuator_count=len(graph.actuator_ids),
        node_count=len(graph.node_ids),
        relations=relations,
        horizon=HydraulicHorizonV110(),
        contract=ActuatorSetHydraulicContractV110(),
    )
    v11.load_state_dict(state_dict, strict=True)
    v11.to(device).eval()
    for parameter in v11.parameters():
        parameter.requires_grad_(False)
    return V7V11RollingMPC(
        value_model=value_model,
        hydraulic_model=v11,
        basis=basis,
        prepared=prepared,
        normalization=normalization,
        priority_indices=priority_indices,
        tfv_cvar_alpha=tfv_cvar_alpha,
        tfv_near_opt_relative=tfv_near_opt_relative,
        tfv_near_opt_absolute_m3=tfv_near_opt_absolute_m3,
        near_opt_penalty=near_opt_penalty,
        movement_tiebreak=movement_tiebreak,
        min_predicted_tfv_improvement_m3=min_predicted_tfv_improvement_m3,
        hydraulic_secondary_iterations=hydraulic_secondary_iterations,
    )


__all__ = ["V110_RUNTIME_BUNDLE_CONTRACT", "load_v7_v11_rolling_mpc"]
