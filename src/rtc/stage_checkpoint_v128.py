"""Nonfinal stage checkpoints for fast/restartable V128 development training.

These checkpoints are intentionally incompatible with the strict final V128 loader.  They
exist only to resume Development training at stage boundaries without recomputing Stage A or
B0.  A stage checkpoint must never be accepted by Policy Lock, Final, or runtime controllers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

V128_STAGE_CHECKPOINT_CONTRACT = "PROJECT7_V128_NONFINAL_DEVELOPMENT_STAGE_CHECKPOINT_V1"
_ALLOWED_STAGES = ("initialized", "stage_a", "stage_b0", "objective")


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "lineage": dict(lineage),
        "training_design": dict(training_design),
        "history": dict(history),
        "model_state_dict": model.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
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
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("V128 stage checkpoint payload must be a dictionary")
    if payload.get("checkpoint_contract") != V128_STAGE_CHECKPOINT_CONTRACT:
        raise ValueError("not a V128 nonfinal development stage checkpoint")
    if any(bool(payload.get(key)) for key in ("scientific_claim_allowed", "policy_lock_allowed", "final_allowed", "runtime_allowed")):
        raise ValueError("V128 stage checkpoint incorrectly claims production/final validity")
    stage = str(payload.get("completed_stage", ""))
    if stage not in _ALLOWED_STAGES:
        raise ValueError("V128 stage checkpoint has invalid completed_stage")
    if str(payload.get("profile")) != str(expected_profile):
        raise ValueError("V128 stage checkpoint execution profile differs from requested profile")
    if str(payload.get("graph_file_sha256")) != _sha256(graph_path):
        raise ValueError("V128 stage checkpoint graph file differs from current graph")
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
        np.random.set_state(rng["numpy"])
    if "torch_cpu" in rng:
        torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda"):
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return payload


def stage_checkpoint_summary(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("invalid V128 stage checkpoint")
    return {
        "checkpoint_contract": payload.get("checkpoint_contract"),
        "completed_stage": payload.get("completed_stage"),
        "profile": payload.get("profile"),
        "scientific_claim_allowed": payload.get("scientific_claim_allowed"),
        "graph_file_sha256": payload.get("graph_file_sha256"),
    }


__all__ = [
    "V128_STAGE_CHECKPOINT_CONTRACT",
    "load_stage_checkpoint_v128",
    "save_stage_checkpoint_v128",
    "stage_checkpoint_summary",
]
