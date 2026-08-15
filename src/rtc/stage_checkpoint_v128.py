"""Nonfinal source-strict stage checkpoints for fast/restartable V128 development.

These checkpoints are intentionally incompatible with the strict final V128 loader. They
exist only to resume Development training at stage boundaries without recomputing Stage A or
B0. All metadata is restricted to weights-only-safe primitive/tensor objects so PyTorch 2.6+
can load them with its safe default. A stage checkpoint must never be accepted by Policy Lock,
Final, or runtime controllers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from .checkpoint_v128 import v128_step2_source_sha256, v128_training_source_sha256

V128_STAGE_CHECKPOINT_CONTRACT = (
    "PROJECT7_V128_NONFINAL_DEVELOPMENT_STAGE_CHECKPOINT_V2_SOURCE_STRICT_WEIGHTS_ONLY"
)
_ALLOWED_STAGES = ("initialized", "stage_a", "stage_b0", "objective")


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _numpy_rng_payload() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "name": str(name),
        "keys": torch.as_tensor(keys.astype(np.uint32, copy=False).astype(np.int64)),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _restore_numpy_rng(payload: Mapping[str, Any]) -> None:
    keys = torch.as_tensor(payload["keys"], dtype=torch.int64).cpu().numpy().astype(np.uint32)
    np.random.set_state(
        (
            str(payload["name"]),
            keys,
            int(payload["pos"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
    )


def _model_class_source_sha256(model: torch.nn.Module) -> str:
    import importlib

    module = importlib.import_module(model.__class__.__module__)
    raw = getattr(module, "__file__", None)
    if raw is None:
        raise RuntimeError(f"cannot fingerprint model class module: {model.__class__.__module__}")
    return _sha256(raw)


def save_stage_checkpoint_v128(
    path: str | Path,
    *,
    model: torch.nn.Module,
    completed_stage: str,
    profile: str,
    graph_path: str | Path,
    lineage: Mapping[str, Any],
    training_design: Mapping[str, Any],
    history: Mapping[str, Any],
) -> Path:
    stage = str(completed_stage)
    if stage not in _ALLOWED_STAGES:
        raise ValueError(f"unsupported V128 stage checkpoint: {stage}")
    payload = {
        "checkpoint_contract": V128_STAGE_CHECKPOINT_CONTRACT,
        "scientific_claim_allowed": False,
        "policy_lock_allowed": False,
        "final_allowed": False,
        "runtime_allowed": False,
        "completed_stage": stage,
        "profile": str(profile),
        "graph_file_sha256": _sha256(graph_path),
        "v128_step2_source_sha256": v128_step2_source_sha256(),
        "v128_training_source_sha256": v128_training_source_sha256(),
        "model_class_source_sha256": _model_class_source_sha256(model),
        "lineage": dict(lineage),
        "training_design": dict(training_design),
        "history": dict(history),
        "model_state_dict": model.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": _numpy_rng_payload(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def load_stage_checkpoint_v128(
    path: str | Path,
    *,
    model: torch.nn.Module,
    expected_profile: str,
    graph_path: str | Path,
    expected_lineage: Mapping[str, Any],
    expected_training_design: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V128 stage checkpoint payload must be a dictionary")
    if payload.get("checkpoint_contract") != V128_STAGE_CHECKPOINT_CONTRACT:
        raise ValueError("not a current V128 nonfinal development stage checkpoint")
    if any(
        bool(payload.get(key))
        for key in ("scientific_claim_allowed", "policy_lock_allowed", "final_allowed", "runtime_allowed")
    ):
        raise ValueError("V128 stage checkpoint incorrectly claims production/final validity")
    stage = str(payload.get("completed_stage", ""))
    if stage not in _ALLOWED_STAGES:
        raise ValueError("V128 stage checkpoint has invalid completed_stage")
    if str(payload.get("profile")) != str(expected_profile):
        raise ValueError("V128 stage checkpoint execution profile differs from requested profile")
    if str(payload.get("graph_file_sha256")) != _sha256(graph_path):
        raise ValueError("V128 stage checkpoint graph file differs from current graph")
    if str(payload.get("v128_step2_source_sha256")) != v128_step2_source_sha256():
        raise ValueError("V128 stage checkpoint model-source semantics changed; restart this stage chain")
    if str(payload.get("v128_training_source_sha256")) != v128_training_source_sha256():
        raise ValueError("V128 stage checkpoint training-source semantics changed; restart this stage chain")
    if str(payload.get("model_class_source_sha256")) != _model_class_source_sha256(model):
        raise ValueError("V128 stage checkpoint model-class source changed; restart this architecture run")
    if dict(payload.get("lineage") or {}) != dict(expected_lineage):
        raise ValueError("V128 stage checkpoint data lineage differs from current run")
    if dict(payload.get("training_design") or {}) != dict(expected_training_design):
        raise ValueError("V128 stage checkpoint training design differs from current run")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("V128 stage checkpoint lacks model_state_dict")
    model.load_state_dict(state, strict=True)
    rng = payload.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        _restore_numpy_rng(rng["numpy"])
    if "torch_cpu" in rng:
        torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda"):
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return payload


def stage_checkpoint_summary(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("invalid V128 stage checkpoint")
    return {
        "checkpoint_contract": payload.get("checkpoint_contract"),
        "completed_stage": payload.get("completed_stage"),
        "profile": payload.get("profile"),
        "scientific_claim_allowed": payload.get("scientific_claim_allowed"),
        "graph_file_sha256": payload.get("graph_file_sha256"),
        "v128_step2_source_sha256": payload.get("v128_step2_source_sha256"),
        "v128_training_source_sha256": payload.get("v128_training_source_sha256"),
        "model_class_source_sha256": payload.get("model_class_source_sha256"),
    }


__all__ = [
    "V128_STAGE_CHECKPOINT_CONTRACT",
    "load_stage_checkpoint_v128",
    "save_stage_checkpoint_v128",
    "stage_checkpoint_summary",
]
