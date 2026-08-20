"""V127 loader for the already-frozen Step1 state estimator.

Historical production entrypoints intentionally remain code-tree bound.  V127, however,
uses an accepted frozen Step1 while Step2/D5/runtime code continues to evolve.  Requiring
that old Step1 checkpoint to carry the *current* whole-project source contract would force
irrelevant Step1 retraining after a D5 or baseline edit.  This loader retains the actual
Step1 checkpoint/model/time requirements and strict state_dict loading, then lets V127 bind
train/deploy identity to the loaded parameter tensors.

Runtime loading is deliberately host-memory conservative.  Checkpoint storages are memory-mapped
on CPU, copied once into the module, and only then is the frozen module moved to the requested
device.  This avoids transiently materialising a second full checkpoint copy on CUDA for every
short-lived authoritative policy-return branch.  The model parameters and numerical inference
contract are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .models import SparseStateEstimator

V127_STEP1_RUNTIME_LOADER_CONTRACT = "PROJECT7_V127_FROZEN_STEP1_SEMANTIC_LOADER_V2_MMAP_CPU_STAGE"


def load_frozen_step1_v127(
    path: str | Path, device: torch.device | str
) -> SparseStateEstimator:
    target = torch.device(device)
    # ``mmap=True`` keeps serialized tensor storages file-backed until touched.  Load on CPU first so
    # a CUDA run does not hold both checkpoint storages and the final module parameters on the GPU.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("V127 Step1 checkpoint payload must be a dictionary")
    if payload.get("checkpoint_contract") != "RTC_TORCH_CHECKPOINT_V2_CODE_BOUND":
        raise ValueError("V127 Step1 checkpoint is not the accepted V2 model checkpoint")
    if payload.get("scientific_split") != "development":
        raise ValueError("V127 Step1 checkpoint is not development-lineage")
    raw = payload.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("V127 Step1 checkpoint lacks model_config")
    cfg = dict(raw)
    cfg.pop("state_weights", None)
    training_contract = cfg.pop("training_contract_sha256", None)
    history_steps = int(cfg.get("history_steps", -1))
    model_step_seconds = int(cfg.get("model_step_seconds", -1))
    engine = str(cfg.get("swmm_engine_version", "")).strip()
    if history_steps <= 0 or model_step_seconds != 300:
        raise ValueError("V127 frozen Step1 time/history configuration is incompatible")
    if not engine:
        raise ValueError("V127 frozen Step1 lacks SWMM engine lineage")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("V127 frozen Step1 checkpoint lacks state_dict")
    model = SparseStateEstimator(**cfg)
    model.load_state_dict(state)
    model.runtime_metadata = {  # type: ignore[attr-defined]
        "history_steps": history_steps,
        "model_step_seconds": model_step_seconds,
        "swmm_engine_version": engine,
        "context_contract": str(cfg.get("context_contract", "")),
        "training_contract_sha256": training_contract,
        "original_rtc_source_tree_sha256": str(payload.get("rtc_source_tree_sha256", "")),
        "v127_loader_contract": V127_STEP1_RUNTIME_LOADER_CONTRACT,
        "runtime_checkpoint_mmap": True,
        "runtime_checkpoint_staged_on_cpu": True,
    }
    # Drop references to file-backed checkpoint tensors before moving the final module to CUDA.
    del state, payload
    return model.to(target).eval()


__all__ = ["V127_STEP1_RUNTIME_LOADER_CONTRACT", "load_frozen_step1_v127"]
