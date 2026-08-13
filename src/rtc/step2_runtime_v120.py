"""Fail-closed loader for the causal execution-bound Project7 V120 bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .code_contract import rtc_implementation_contract_sha256
from .graph import GraphSchema
from .step2_causal_forecast_v120 import V120_CAUSAL_RAINFALL_CONTRACT
from .step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from .step2_control_response_v60 import prepare_static_v60
from .step2_control_response_v70 import ControlValueSurrogateV70
from .step2_policy_v120 import RuntimeNormalizationV120, ValueOnlyCandidatePolicyV120
from .step2_v120_contract import (
    Step2V120Contract,
    V120_BUNDLE_CONTRACT,
    V120_CANDIDATE_POLICY_CONTRACT,
    V120_CONTRACT,
    v120_graph_semantic_sha256,
    v120_runtime_contract_sha256,
)
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


def _verify_v120_lineage(payload: dict[str, Any], graph: GraphSchema) -> None:
    recorded_code = str(payload.get("rtc_implementation_contract_sha256", ""))
    current_code = rtc_implementation_contract_sha256()
    if not recorded_code or recorded_code != current_code:
        raise ValueError("V120 bundle was trained under a different global RTC contract")
    if str(payload.get("v120_runtime_contract_sha256", "")) != v120_runtime_contract_sha256():
        raise ValueError("V120 bundle was trained under a different V120 implementation")

    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V120 bundle lacks lineage")
    recorded_graph = str(lineage.get("graph_semantic_sha256", ""))
    if not recorded_graph or recorded_graph != v120_graph_semantic_sha256(graph):
        raise ValueError("V120 runtime graph differs from the training graph")

    rainfall = payload.get("rainfall_input")
    if not isinstance(rainfall, dict) or rainfall.get("contract") != V120_CAUSAL_RAINFALL_CONTRACT:
        raise ValueError("V120 bundle was not trained with the causal rainfall input contract")
    if rainfall.get("future_realized_rainfall_used_as_model_input") is not False:
        raise ValueError("V120 bundle leaks future realised rainfall")
    if int(rainfall.get("required_runtime_history_steps_for_level", -1)) != 1:
        raise ValueError("V120 rainfall information budget is not runtime-aligned")
    if abs(float(rainfall.get("decay_per_step", -1.0)) - 0.92) > 1.0e-12:
        raise ValueError("V120 rainfall decay contract drift")

    candidate = payload.get("candidate_policy")
    if not isinstance(candidate, dict) or candidate.get("contract") != V120_CANDIDATE_POLICY_CONTRACT:
        raise ValueError("V120 candidate-policy contract is stale")
    if candidate.get("continuous_gradient_search") is not False:
        raise ValueError("V120 production forbids unrestricted continuous gradient search")
    if candidate.get("score_only_executable_sequences") is not True:
        raise ValueError("V120 bundle did not bind scoring to executable candidates")
    if candidate.get("first_move_bound_to_current_and_target_readback") is not True:
        raise ValueError("V120 first-move scoring is not bound to SWMM target readback")

    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("V120 bundle lacks split evidence")
    if len(split.get("fit_event_ids", [])) != 14 or len(split.get("holdout_event_ids", [])) != 4:
        raise ValueError("V120 bundle does not preserve frozen 14/4 Train-only split")
    if int(split.get("event_overlap", -1)) != 0 or int(split.get("rainfall_overlap", -1)) != 0:
        raise ValueError("V120 internal TrainFit/holdout split overlaps")
    if int(split.get("validation_events_loaded", -1)) != 0 or int(split.get("final_events_loaded", -1)) != 0:
        raise ValueError("V120 bundle accessed Validation or Final labels")

    census = payload.get("data_census")
    if not isinstance(census, dict):
        raise ValueError("V120 bundle lacks data census")
    if int(census.get("source_d2_authoritative_branch_census", -1)) != 4800:
        raise ValueError("V120 bundle does not record D2 source census=4800")
    if int(census.get("targeted_d3_branches", -1)) != 3600:
        raise ValueError("V120 bundle lacks complete targeted D3=3600 Train view")
    if int(census.get("eligible_event_count", -1)) != 18:
        raise ValueError("V120 bundle is not frozen-Train18 only")
    if list(census.get("non_train_events_loaded", [])):
        raise ValueError("V120 bundle contains non-Train labels")


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
        raise ValueError("runtime requires a causal execution-bound V120 bundle")
    if payload.get("step2_contract") != V120_CONTRACT:
        raise ValueError("V120 bundle scientific contract mismatch")
    if payload.get("base_value_contract") != V70_CONTRACT:
        raise ValueError("V120 bundle must use the frozen V7 direct Value architecture")
    if not bool(payload.get("runtime_compatible", False)):
        raise ValueError("V120 bundle did not pass the internal control-value gate")
    if not allow_development and not bool(payload.get("production_compatible", False)):
        raise ValueError("V120 bundle is development-only; promotion is required for production")
    _verify_v120_lineage(payload, graph)

    contract = Step2V120Contract()
    contract.validate()
    basis = build_control_basis_v60(graph)
    recorded_basis = payload.get("basis_manifest")
    if not isinstance(recorded_basis, dict):
        raise ValueError("V120 bundle lacks basis_manifest")
    _basis_matches(recorded_basis, basis_manifest_v60(basis))
    normalization_raw = payload.get("input_normalization")
    if not isinstance(normalization_raw, dict):
        raise ValueError("V120 bundle lacks input_normalization")
    normalization = RuntimeNormalizationV120.from_payload(normalization_raw)
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("V120 bundle lacks model_config")
    if int(model_config.get("state_dim", -1)) != 6:
        raise ValueError("V120 Value state dimension must match six-channel Step1")
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
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("V120 bundle lacks model_config")
    rainfall = payload.get("rainfall_input")
    return {
        "bundle_contract": payload.get("bundle_contract"),
        "step2_contract": payload.get("step2_contract"),
        "runtime_compatible": bool(payload.get("runtime_compatible", False)),
        "production_compatible": bool(payload.get("production_compatible", False)),
        "swmm_engine_version": str(model_config.get("swmm_engine_version", "")),
        "state_dim": int(model_config.get("state_dim", -1)),
        "value_horizon_minutes": int(payload.get("value_horizon_minutes", -1)),
        "control_update_seconds": int(payload.get("control_update_seconds", -1)),
        "v120_runtime_contract_sha256": str(payload.get("v120_runtime_contract_sha256", "")),
        "rainfall_input_contract": (
            str(rainfall.get("contract", "")) if isinstance(rainfall, dict) else ""
        ),
    }


__all__ = ["load_value_only_policy_v120", "v120_bundle_metadata"]
