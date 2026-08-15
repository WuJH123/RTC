"""Contract-strict checkpoints for Project7 V128 typed actuator-message Step2."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint_v127 import (
    _normalization_payload,
    graph_semantic_sha256_v127,
    input_normalization_from_v127_checkpoint,
)
from .code_contract import rtc_implementation_contract_sha256
from .development_profile_v128 import V128_EXECUTION_PROFILE_CONTRACT
from .step2_differentiable_v128 import (
    TypedActuatorMessageSurrogateV128,
    V128_STEP2_CONTRACT,
)
from .step2_train_response_v60 import InputNormalizationV60

V128_CHECKPOINT_CONTRACT = (
    "PROJECT7_V128_STEP2_CHECKPOINT_V6_CURRENT_PROFILE_TRAINING_SOURCE_STRICT"
)
_V128_MODEL_SOURCE_FILES = (
    "models.py",
    "flood_volume.py",
    "step2_differentiable_v127.py",
    "step2_differentiable_v128.py",
    "checkpoint_v128.py",
)
_V128_TRAINING_SOURCE_FILES = (
    "step2_train_v127_streaming.py",
    "step2_train_v127_control.py",
    "step2_train_v128_hydraulic.py",
    "step2_train_v128_exact.py",
    "step2_lazy_stream_v128.py",
    "development_profile_v128.py",
    "v128_control_profile.py",
    "step2_gradient_v127.py",
    "step2_gradient_v127_streaming.py",
    "step2_gradient_v127_fast.py",
    "step3_mpc_v127.py",
)


def _source_files_sha256(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"V128 source file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def v128_step2_source_sha256() -> str:
    return _source_files_sha256(Path(__file__).resolve().parent, _V128_MODEL_SOURCE_FILES)


def v128_training_source_sha256() -> str:
    return _source_files_sha256(Path(__file__).resolve().parent, _V128_TRAINING_SOURCE_FILES)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"V128 checkpoint lacks canonical {label} SHA256")
    return text


def save_step2_v128(
    path: str | Path,
    *,
    model: TypedActuatorMessageSurrogateV128,
    graph: Any,
    input_normalization: InputNormalizationV60,
    training_report: dict[str, Any],
    lineage: dict[str, Any],
) -> Path:
    if not isinstance(model, TypedActuatorMessageSurrogateV128):
        raise TypeError("V128 checkpoint saver requires TypedActuatorMessageSurrogateV128")
    if str(training_report.get("profile", "")).lower() != "full":
        raise ValueError("strict V128 checkpoint saver accepts only explicit --profile full")
    if training_report.get("execution_profile_contract") != V128_EXECUTION_PROFILE_CONTRACT:
        raise ValueError("V128 strict checkpoint requires the current smoke/dev/full runner contract")
    if training_report.get("final_checkpoint_allowed") is not True:
        raise ValueError("training report does not authorize a strict final base checkpoint")
    payload = {
        "checkpoint_contract": V128_CHECKPOINT_CONTRACT,
        "step2_contract": V128_STEP2_CONTRACT,
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "v128_step2_source_sha256": v128_step2_source_sha256(),
        "v128_training_source_sha256": v128_training_source_sha256(),
        "graph_semantic_sha256": graph_semantic_sha256_v127(graph),
        "scientific_split": "development",
        "execution_profile": "full",
        "model_config": {
            "state_dim": int(model.transition.state_mean.numel()),
            "rainfall_dim": int(model.transition.rain_mean.numel()),
            "node_static_dim": int(np.asarray(graph.static_node_features).shape[1]),
            "actuator_physics_dim": int(np.asarray(graph.actuator_physics).shape[1]),
            "actuator_count": len(graph.actuator_ids),
            "hidden_dim": int(model.transition.input.out_features),
            "actuator_embedding_dim": int(model.actuator.actuator_embedding_dim),
            "action_message_dim": int(model.action_message_dim),
            "delta_state_scale": model.transition.delta_state_scale.detach().cpu(),
            "delta_flow_scale": model.actuator.delta_flow_scale.detach().cpu(),
            "model_step_seconds": 300,
            "horizon_steps": 72,
            "control_update_seconds": 600,
            "free_control_horizon_steps": 24,
            "time_contract": "PROJECT7_V128_300S_MODEL_600S_RECEDING_CONTROL_V1",
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


def load_step2_v128(
    path: str | Path,
    *,
    graph: Any,
    device: torch.device | str,
) -> tuple[TypedActuatorMessageSurrogateV128, dict[str, Any]]:
    target = torch.device(device)
    payload = torch.load(path, map_location=target)
    if not isinstance(payload, dict):
        raise ValueError("V128 Step2 checkpoint must contain a dictionary")
    if payload.get("checkpoint_contract") != V128_CHECKPOINT_CONTRACT:
        raise ValueError("not a current V128 typed-actuator checkpoint")
    if payload.get("step2_contract") != V128_STEP2_CONTRACT:
        raise ValueError("V128 Step2 scientific contract mismatch")
    if payload.get("scientific_split") != "development":
        raise ValueError("V128 Step2 checkpoint is not development-lineage")
    if str(payload.get("execution_profile", "")).lower() != "full":
        raise ValueError("V128 runtime rejects smoke/dev nonfinal artifacts")

    stored_model_source = _require_sha256(
        payload.get("v128_step2_source_sha256"), label="Step2 model source"
    )
    if stored_model_source != v128_step2_source_sha256():
        raise ValueError("V128 Step2 model-source semantics changed after checkpoint creation")
    stored_training_source = _require_sha256(
        payload.get("v128_training_source_sha256"), label="training source"
    )
    if stored_training_source != v128_training_source_sha256():
        raise ValueError("V128 Step2 training-source semantics changed after checkpoint creation")
    if payload.get("graph_semantic_sha256") != graph_semantic_sha256_v127(graph):
        raise ValueError("V128 Step2 graph topology/features differ from runtime graph")
    _ = input_normalization_from_v127_checkpoint(payload)
    raw = payload.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("V128 Step2 checkpoint lacks model_config")
    cfg = dict(raw)
    if int(cfg.get("actuator_count", -1)) != len(graph.actuator_ids):
        raise ValueError("V128 checkpoint actuator count differs from graph")
    if int(cfg.get("node_static_dim", -1)) != int(np.asarray(graph.static_node_features).shape[1]):
        raise ValueError("V128 checkpoint node-static schema differs from graph")
    if int(cfg.get("actuator_physics_dim", -1)) != int(np.asarray(graph.actuator_physics).shape[1]):
        raise ValueError("V128 checkpoint actuator-physics schema differs from graph")
    if int(cfg.get("action_message_dim", 0)) <= 0:
        raise ValueError("V128 checkpoint lacks typed action-message dimension")
    for key, expected in (
        ("model_step_seconds", 300),
        ("horizon_steps", 72),
        ("control_update_seconds", 600),
        ("free_control_horizon_steps", 24),
    ):
        if int(cfg.get(key, -1)) != expected:
            raise ValueError(f"V128 checkpoint {key} differs from frozen time contract")

    model = TypedActuatorMessageSurrogateV128(
        state_dim=int(cfg["state_dim"]),
        rainfall_dim=int(cfg["rainfall_dim"]),
        node_static_dim=int(cfg["node_static_dim"]),
        actuator_physics_dim=int(cfg["actuator_physics_dim"]),
        actuator_count=int(cfg["actuator_count"]),
        hidden_dim=int(cfg["hidden_dim"]),
        actuator_embedding_dim=int(cfg["actuator_embedding_dim"]),
        action_message_dim=int(cfg["action_message_dim"]),
        delta_state_scale=torch.as_tensor(cfg["delta_state_scale"], dtype=torch.float32),
        delta_flow_scale=torch.as_tensor(cfg["delta_flow_scale"], dtype=torch.float32),
        model_step_seconds=300,
        horizon_steps=72,
        control_update_seconds=600,
        free_control_horizon_steps=24,
        time_contract="PROJECT7_V128_300S_MODEL_600S_RECEDING_CONTROL_V1",
        v128_step2_contract=V128_STEP2_CONTRACT,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("V128 Step2 checkpoint lacks state_dict")
    model.load_state_dict(state, strict=True)
    model.runtime_metadata.update(
        {
            "model_step_seconds": 300,
            "horizon_steps": 72,
            "control_update_seconds": 600,
            "free_control_horizon_steps": 24,
            "time_contract": "PROJECT7_V128_300S_MODEL_600S_RECEDING_CONTROL_V1",
            "graph_semantic_sha256": payload["graph_semantic_sha256"],
            "v128_step2_source_sha256": stored_model_source,
            "v128_training_source_sha256": stored_training_source,
            "execution_profile": "full",
        }
    )
    return model.to(target).eval(), payload


__all__ = [
    "V128_CHECKPOINT_CONTRACT",
    "load_step2_v128",
    "save_step2_v128",
    "v128_step2_source_sha256",
    "v128_training_source_sha256",
]
