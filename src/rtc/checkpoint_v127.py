"""Code-bound V127 differentiable surrogate checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .code_contract import rtc_implementation_contract_sha256
from .step2_differentiable_v127 import (
    ControlOrientedDifferentiableSurrogateV127,
    V127_STEP2_CONTRACT,
)
from .step2_train_response_v60 import InputNormalizationV60

V127_CHECKPOINT_CONTRACT = "PROJECT7_V127_STEP2_CHECKPOINT_V2_CAUSAL_INPUT_BOUND"


def _normalization_payload(value: InputNormalizationV60) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.as_tensor(value.state_mean, dtype=torch.float32).cpu(),
        "state_std": torch.as_tensor(value.state_std, dtype=torch.float32).cpu(),
        "rainfall_mean": torch.as_tensor(value.rainfall_mean, dtype=torch.float32).cpu(),
        "rainfall_std": torch.as_tensor(value.rainfall_std, dtype=torch.float32).cpu(),
        "flow_mean": torch.as_tensor(value.flow_mean, dtype=torch.float32).cpu(),
        "flow_std": torch.as_tensor(value.flow_std, dtype=torch.float32).cpu(),
    }


def input_normalization_from_v127_checkpoint(payload: dict[str, Any]) -> InputNormalizationV60:
    raw = payload.get("input_normalization")
    if not isinstance(raw, dict):
        raise ValueError("V127 Step2 checkpoint lacks causal input_normalization")
    required = {"state_mean", "state_std", "rainfall_mean", "rainfall_std", "flow_mean", "flow_std"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"V127 Step2 checkpoint normalization missing {missing}")
    arrays = {name: torch.as_tensor(raw[name]).detach().cpu().numpy().astype(np.float32) for name in required}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("V127 checkpoint normalization contains non-finite values")
    if np.any(arrays["state_std"] <= 0) or np.any(arrays["rainfall_std"] <= 0) or np.any(arrays["flow_std"] <= 0):
        raise ValueError("V127 checkpoint normalization contains non-positive standard deviation")
    return InputNormalizationV60(
        arrays["state_mean"], arrays["state_std"],
        arrays["rainfall_mean"], arrays["rainfall_std"],
        arrays["flow_mean"], arrays["flow_std"],
    )


def save_step2_v127(
    path: str | Path,
    *,
    model: ControlOrientedDifferentiableSurrogateV127,
    graph: Any,
    input_normalization: InputNormalizationV60,
    training_report: dict[str, Any],
    lineage: dict[str, Any],
) -> Path:
    payload = {
        "checkpoint_contract": V127_CHECKPOINT_CONTRACT,
        "step2_contract": V127_STEP2_CONTRACT,
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "scientific_split": "development",
        "model_config": {
            "state_dim": int(model.transition.state_mean.numel()),
            "rainfall_dim": int(model.transition.rain_mean.numel()),
            "node_static_dim": int(np.asarray(graph.static_node_features).shape[1]),
            "actuator_physics_dim": int(np.asarray(graph.actuator_physics).shape[1]),
            "actuator_count": len(graph.actuator_ids),
            "hidden_dim": int(model.transition.input.out_features),
            "actuator_embedding_dim": int(model.actuator.actuator_embedding_dim),
            "delta_state_scale": model.transition.delta_state_scale.detach().cpu(),
            "delta_flow_scale": model.actuator.delta_flow_scale.detach().cpu(),
            "model_step_seconds": 300,
            "horizon_steps": 72,
            "control_update_seconds": 600,
            "free_control_horizon_steps": 24,
            "time_contract": "PROJECT7_V127_300S_MODEL_600S_RECEDING_CONTROL_V1",
        },
        "input_normalization": _normalization_payload(input_normalization),
        "state_dict": model.state_dict(),
        "training_report": training_report,
        "lineage": lineage,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def load_step2_v127(
    path: str | Path,
    *,
    graph: Any,
    device: torch.device | str,
    require_current_code: bool = True,
) -> tuple[ControlOrientedDifferentiableSurrogateV127, dict[str, Any]]:
    target = torch.device(device)
    payload = torch.load(path, map_location=target)
    if not isinstance(payload, dict):
        raise ValueError("V127 Step2 checkpoint must contain a dictionary")
    if payload.get("checkpoint_contract") != V127_CHECKPOINT_CONTRACT:
        raise ValueError("not a V127 differentiable surrogate checkpoint")
    if payload.get("step2_contract") != V127_STEP2_CONTRACT:
        raise ValueError("V127 Step2 checkpoint scientific contract mismatch")
    if payload.get("scientific_split") != "development":
        raise ValueError("V127 Step2 checkpoint is not development-lineage")
    if require_current_code and payload.get("rtc_implementation_contract_sha256") != rtc_implementation_contract_sha256():
        raise ValueError("V127 Step2 checkpoint was trained under another implementation contract")
    _ = input_normalization_from_v127_checkpoint(payload)
    raw = payload.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("V127 Step2 checkpoint lacks model_config")
    cfg = dict(raw)
    if int(cfg.get("actuator_count", -1)) != len(graph.actuator_ids):
        raise ValueError("V127 checkpoint actuator count differs from graph")
    if int(cfg.get("node_static_dim", -1)) != int(np.asarray(graph.static_node_features).shape[1]):
        raise ValueError("V127 checkpoint node-static schema differs from graph")
    if int(cfg.get("actuator_physics_dim", -1)) != int(np.asarray(graph.actuator_physics).shape[1]):
        raise ValueError("V127 checkpoint actuator-physics schema differs from graph")
    for key, expected in (
        ("model_step_seconds", 300),
        ("horizon_steps", 72),
        ("control_update_seconds", 600),
        ("free_control_horizon_steps", 24),
    ):
        if int(cfg.get(key, -1)) != expected:
            raise ValueError(f"V127 checkpoint {key} differs from frozen contract")
    model = ControlOrientedDifferentiableSurrogateV127(
        state_dim=int(cfg["state_dim"]),
        rainfall_dim=int(cfg["rainfall_dim"]),
        node_static_dim=int(cfg["node_static_dim"]),
        actuator_physics_dim=int(cfg["actuator_physics_dim"]),
        actuator_count=int(cfg["actuator_count"]),
        hidden_dim=int(cfg["hidden_dim"]),
        actuator_embedding_dim=int(cfg["actuator_embedding_dim"]),
        delta_state_scale=torch.as_tensor(cfg["delta_state_scale"], dtype=torch.float32),
        delta_flow_scale=torch.as_tensor(cfg["delta_flow_scale"], dtype=torch.float32),
        model_step_seconds=300,
        horizon_steps=72,
        control_update_seconds=600,
        free_control_horizon_steps=24,
        time_contract="PROJECT7_V127_300S_MODEL_600S_RECEDING_CONTROL_V1",
        v127_step2_contract=V127_STEP2_CONTRACT,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("V127 Step2 checkpoint lacks state_dict")
    model.load_state_dict(state)
    model.runtime_metadata = {
        "model_step_seconds": 300,
        "horizon_steps": 72,
        "control_update_seconds": 600,
        "free_control_horizon_steps": 24,
        "time_contract": "PROJECT7_V127_300S_MODEL_600S_RECEDING_CONTROL_V1",
        "rtc_implementation_contract_sha256": payload["rtc_implementation_contract_sha256"],
    }
    return model.to(target).eval(), payload


__all__ = [
    "V127_CHECKPOINT_CONTRACT",
    "input_normalization_from_v127_checkpoint",
    "load_step2_v127",
    "save_step2_v127",
]
