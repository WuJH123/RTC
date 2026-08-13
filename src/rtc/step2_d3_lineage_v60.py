"""V6 targeted-D3 lineage helpers."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import pandas as pd

from .step2_control_basis_v60 import ControlBasisV60, basis_manifest_v60
from .step2_d3_design_v60 import D3V60DesignContract
from .step2_v60_contract import V60_D3_DATA_CONTRACT


def canonical_contract_sha256(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stamp_d3_v60_lineage(frame: pd.DataFrame, *, basis: ControlBasisV60, design_contract: D3V60DesignContract) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty:
        raise ValueError("cannot stamp an empty V6 D3 manifest")
    basis_payload = basis_manifest_v60(basis)
    design_payload = asdict(design_contract)
    basis_sha = canonical_contract_sha256(basis_payload)
    design_sha = canonical_contract_sha256(design_payload)
    result = frame.copy()
    result["v60_data_contract"] = V60_D3_DATA_CONTRACT
    result["v60_control_basis_contract"] = str(basis_payload["contract"])
    result["v60_control_basis_sha256"] = basis_sha
    result["v60_design_contract_sha256"] = design_sha
    return result, {
        "v60_data_contract": V60_D3_DATA_CONTRACT,
        "v60_control_basis_contract": str(basis_payload["contract"]),
        "v60_control_basis_sha256": basis_sha,
        "v60_design_contract_sha256": design_sha,
        "design_contract": design_payload,
        "basis_manifest": basis_payload,
    }


__all__ = ["canonical_contract_sha256", "stamp_d3_v60_lineage"]
