from __future__ import annotations

import torch

from .step2_v120_contract import V120_BUNDLE_CONTRACT
from .step2_v120_data_contract import STATE_DOMAIN_CONTRACT


def is_v120_bundle(path: str | None) -> bool:
    """Identify the V120 bundle type without imposing production evidence semantics."""
    if not path:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("bundle_contract") == V120_BUNDLE_CONTRACT


def require_strict_v120_evidence(path: str) -> None:
    """Fail closed unless a V120 bundle was produced by the strict audited path."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("bundle_contract") != V120_BUNDLE_CONTRACT:
        raise ValueError("strict V120 evidence guard received a non-V120 bundle")
    lineage = payload.get("lineage")
    state_input = payload.get("state_input")
    gate = payload.get("value_gate")
    if not isinstance(lineage, dict) or not str(lineage.get("d2_source_audit_sha256", "")).strip():
        raise ValueError("V120 bundle lacks strict D2 source-audit lineage")
    if not isinstance(state_input, dict) or state_input.get("contract") != STATE_DOMAIN_CONTRACT:
        raise ValueError("V120 bundle lacks explicit train/runtime state-domain evidence")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("V120 strict combined Value gate did not pass")
    d2_gate = gate.get("auxiliary_d2_integrity")
    if not isinstance(d2_gate, dict) or d2_gate.get("passed") is not True:
        raise ValueError("V120 D2 auxiliary integrity did not pass")


__all__ = ["is_v120_bundle", "require_strict_v120_evidence"]
