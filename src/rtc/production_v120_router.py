from __future__ import annotations

import torch

from .step2_v120_contract import V120_BUNDLE_CONTRACT


def is_v120_bundle(path: str | None) -> bool:
    if not path:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("bundle_contract") == V120_BUNDLE_CONTRACT


__all__ = ["is_v120_bundle"]
