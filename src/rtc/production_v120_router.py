from __future__ import annotations

import torch

from .step2_v120_contract import V120_BUNDLE_CONTRACT
from .step2_v120_data_contract import STATE_DOMAIN_CONTRACT


def is_v120_bundle(path: str | None) -> bool:
    if not path:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    if not isinstance(payload, dict) or payload.get("bundle_contract") != V120_BUNDLE_CONTRACT:
        return False
    lineage = payload.get("lineage")
    state_input = payload.get("state_input")
    gate = payload.get("value_gate")
    if not isinstance(lineage, dict) or not str(lineage.get("d2_source_audit_sha256", "")).strip():
        raise ValueError("V120 bundle lacks strict D2 source-audit lineage")
    if not isinstance(state_input, dict) or state_input.get("contract") != STATE_DOMAIN_CONTRACT:
        raise ValueError("V120 bundle lacks explicit train/runtime state-domain evidence")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("V120 strict combined Value gate did not pass")
    return True


__all__ = ["is_v120_bundle"]
