from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .generation_contract import generation_key


TRAIN_STATE_CONTRACT = "RTC_TRAIN_STATE_V1_CODE_BOUND"


def training_contract_sha(kind: str, payload: Mapping[str, object]) -> tuple[str, str]:
    return generation_key(
        f"training:{kind}",
        {"train_state_contract": TRAIN_STATE_CONTRACT, **dict(payload)},
    )


def save_training_state(
    path: str | Path,
    *,
    contract_sha256: str,
    rtc_source_tree_sha256: str,
    completed_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object | None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "contract": TRAIN_STATE_CONTRACT,
        "training_contract_sha256": str(contract_sha256),
        "rtc_source_tree_sha256": str(rtc_source_tree_sha256),
        "completed_epochs": int(completed_epochs),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra_state": dict(extra_state or {}),
    }
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state_dict"] = scaler.state_dict()
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    return out


def restore_training_state(
    path: str | Path,
    *,
    expected_contract_sha256: str,
    expected_code_sha256: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object | None,
    map_location: torch.device | str,
) -> tuple[int, dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return 0, {}
    payload = torch.load(p, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("contract") != TRAIN_STATE_CONTRACT:
        raise ValueError(f"not a {TRAIN_STATE_CONTRACT}: {p}")
    if payload.get("training_contract_sha256") != expected_contract_sha256:
        raise ValueError(
            "training resume state belongs to different data/configuration; delete it or use --no-resume-training"
        )
    if payload.get("rtc_source_tree_sha256") != expected_code_sha256:
        raise ValueError("training resume state was produced by a different RTC source tree")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(payload["scaler_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_state = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    extra = payload.get("extra_state", {})
    if not isinstance(extra, dict):
        raise ValueError("training resume extra_state must be a mapping")
    return int(payload.get("completed_epochs", 0)), dict(extra)
