"""Fail-closed loader for the Project7 V12.0 TFV-value-only runtime bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .code_contract import rtc_implementation_contract_sha256
from .graph import GraphSchema
from .step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from .step2_control_response_v60 import prepare_static_v60
from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_policy_v120 import RuntimeNormalizationV120, ValueOnlyCandidatePolicyV120
from .step2_v120_contract import Step2V120Contract, V120_BUNDLE_CONTRACT, V120_CONTRACT
from .step2_v70_contract import V70_CONTRACT


def _load(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("V120 bundle must contain a dictionary")
    return payload


def _basis_matches(recorded: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "actuator_count",
        "control_group_count",
        "temporal_basis_count",
        "coefficient_dimension",
        "control_blocks",
        "model_horizon_steps",
        "group_id_by_actuator",
        "zone_id_by_actuator",
    )
    for key in keys:
        if recorded.get(key) != current.get(key):
            raise ValueError(f"V120 runtime control basis mismatch: {key}")
    if not np.allclose(
        np.asarray(recorded.get("temporal_basis"), dtype=np.float64),
        np.asarray(current.get("temporal_basis"), dtype=np.float64),
        atol=1.0e-7,
    ):
        raise ValueError("V120 temporal basis differs from training bundle")


def load_value_only_policy_v120(
    *,
    graph: GraphSchema,
    bundle_path: str | Path,
    device: torch.device,
    allow_development: bool = False,
    cvar_alpha: float = 0.90,
    min_predicted_improvement_m3: float = 0.0,
    movement_tiebreak: float = 1.0e-6,
) -> ValueOnlyCandidatePolicyV120:
    payload = _load(bundle_path)
    if payload.get("bundle_contract") != V120_BUNDLE_CONTRACT:
        raise ValueError("runtime requires a V120 TFV-value-only bundle")
    if payload.get("step2_contract") != V120_CONTRACT:
        raise ValueError("V120 bundle scientific contract mismatch")
    if payload.get("base_value_contract") != V70_CONTRACT:
        raise ValueError("V120 bundle must use the frozen V7 direct Value architecture")
    if not bool(payload.get("runtime_compatible", False)):
        raise ValueError("V120 bundle did not pass the runtime-compatible value gate")
    if not allow_development and not bool(payload.get("production_compatible", False)):
        raise ValueError("V120 bundle is development-only; promotion is required for production")
    recorded_code = str(payload.get("rtc_implementation_contract_sha256", ""))
    current_code = rtc_implementation_contract_sha256()
    if not recorded_code or recorded_code != current_code:
        raise ValueError("V120 bundle was trained under a different RTC implementation contract")

    contract = Step2V120Contract()
    contract.validate()
    basis = build_control_basis_v60(graph)
    recorded_basis = payload.get("basis_manifest")
    if not isinstance(recorded_basis, dict):
        raise ValueError("V120 bundle lacks basis_manifest")
    _basis_matches(recorded_basis, basis_manifest_v60(basis))
    normalization = RuntimeNormalizationV120.from_payload(payload["input_normalization"])
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("V120 bundle lacks model_config")
    if int(model_config.get("actuator_count", -1)) != len(graph.actuator_ids):
        raise ValueError("V120 model/graph actuator count mismatch")
    prepared = prepare_static_v60(graph, device)
    if int(model_config.get("physics_dim", -1)) != int(prepared.actuator_physics.shape[1]):
        raise ValueError("V120 model/graph actuator physics mismatch")
    model = ControlValueSurrogateV70(
        state_dim=int(model_config["state_dim"]),
        rainfall_dim=int(model_config["rainfall_dim"]),
        physics_dim=int(model_config["physics_dim"]),
        actuator_count=int(model_config["actuator_count"]),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(model_config["direct_tfv_scale_m3"]),
        hidden_dim=int(model_config.get("hidden_dim", 96)),
        actuator_embedding_dim=int(model_config.get("actuator_embedding_dim", 16)),
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("V120 bundle lacks state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return ValueOnlyCandidatePolicyV120(
        model=model,
        basis=basis,
        prepared=prepared,
        normalization=normalization,
        cvar_alpha=cvar_alpha,
        min_predicted_improvement_m3=min_predicted_improvement_m3,
        movement_tiebreak=movement_tiebreak,
        contract=contract,
    )


def v120_bundle_metadata(path: str | Path) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("bundle_contract") != V120_BUNDLE_CONTRACT:
        raise ValueError("not a V120 bundle")
    model_config = payload.get("model_config", {})
    return {
        "bundle_contract": payload.get("bundle_contract"),
        "step2_contract": payload.get("step2_contract"),
        "runtime_compatible": bool(payload.get("runtime_compatible", False)),
        "production_compatible": bool(payload.get("production_compatible", False)),
        "swmm_engine_version": str(model_config.get("swmm_engine_version", "")),
        "value_horizon_minutes": int(payload.get("value_horizon_minutes", -1)),
        "control_update_seconds": int(payload.get("control_update_seconds", -1)),
    }


__all__ = ["load_value_only_policy_v120", "v120_bundle_metadata"]
