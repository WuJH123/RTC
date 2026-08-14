from __future__ import annotations

from pathlib import Path

import torch

from .production_cli import _require_current_checkpoint
from .step1_model_v122 import STEP1_V122_CONTRACT, SparseStateEstimatorV122


def load_step1_v122(path: str | Path, device: torch.device) -> SparseStateEstimatorV122:
    payload = _require_current_checkpoint(
        torch.load(path, map_location=device), model_name="Step1-V122"
    )
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("V122 Step1 checkpoint lacks model_config")
    cfg = dict(raw_config)
    if cfg.pop("step1_contract", None) != STEP1_V122_CONTRACT:
        raise ValueError("checkpoint is not a V122 Step1 model")
    # Training-only metadata is retained in the checkpoint/report but is not part of the
    # network constructor.
    cfg.pop("sensor_dropout_probability", None)
    cfg.pop("loss_contract", None)
    training_contract = cfg.pop("training_contract_sha256", None)
    runtime_metadata = {
        "step1_contract": STEP1_V122_CONTRACT,
        "history_steps": int(cfg.get("history_steps", -1)),
        "model_step_seconds": int(cfg.get("model_step_seconds", -1)),
        "swmm_engine_version": str(cfg.get("swmm_engine_version", "")).strip(),
        "context_contract": str(cfg.get("context_contract", "")),
        "training_contract_sha256": training_contract,
        "rtc_source_tree_sha256": str(payload["rtc_source_tree_sha256"]),
    }
    if runtime_metadata["history_steps"] != 13:
        raise ValueError("V122 Step1 checkpoint does not use 13 causal frames")
    if runtime_metadata["model_step_seconds"] != 300:
        raise ValueError("V122 Step1 checkpoint does not use the frozen 300-s model step")
    if not runtime_metadata["swmm_engine_version"]:
        raise ValueError("V122 Step1 checkpoint lacks SWMM engine lineage")
    model = SparseStateEstimatorV122(**cfg)
    model.runtime_metadata = runtime_metadata
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


__all__ = ["load_step1_v122"]
