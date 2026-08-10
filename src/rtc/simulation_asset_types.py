from __future__ import annotations

import json
from pathlib import Path

from .inp_lineage import physical_contract_sha256
from .simulation_assets import (
    SIMULATION_IDENTITY_CONTRACT,
    VALID_REUSABLE,
    SimulationAssetRegistry,
    checkpoint_state_sha256,
    event_prefix_family_sha256,
    sha256_json,
)


D3_EXECUTION_SEMANTICS = "D3_CONTROLS_DISABLED_COMPACT_V3_PREFIX_VERIFIED"


def d3_identity(
    *,
    inp_path: str | Path,
    reference_metadata_path: str | Path,
    checkpoint_seconds: int,
    sequence_sha256: str,
    swmm_engine_version: str,
    stride_seconds: int,
    control_block_seconds: int,
    horizon_seconds: int,
) -> tuple[str, str, dict[str, object]]:
    if min(checkpoint_seconds, stride_seconds, control_block_seconds, horizon_seconds) <= 0:
        raise ValueError("D3 identity timing values must be positive")
    if horizon_seconds % control_block_seconds:
        raise ValueError("D3 identity horizon must contain complete control blocks")
    family = {
        "identity_contract": SIMULATION_IDENTITY_CONTRACT,
        "kind": "D3_SEQUENCE",
        "execution_semantics": D3_EXECUTION_SEMANTICS,
        "physical_network_sha256": physical_contract_sha256(inp_path),
        "event_prefix_family_sha256": event_prefix_family_sha256(inp_path),
        "checkpoint_elapsed_seconds": int(checkpoint_seconds),
        "checkpoint_state_sha256": checkpoint_state_sha256(
            reference_metadata_path, elapsed_seconds=int(checkpoint_seconds)
        ),
        "sequence_sha256": str(sequence_sha256),
        "native_controls_enabled": False,
        "swmm_engine_version": str(swmm_engine_version),
        "record_stride_seconds": int(stride_seconds),
        "control_block_seconds": int(control_block_seconds),
    }
    family_key = sha256_json(family)
    payload = {**family, "horizon_seconds": int(horizon_seconds)}
    return sha256_json(payload), family_key, payload


def register_d3_metadata(
    registry: SimulationAssetRegistry,
    metadata_path: str | Path,
    *,
    qualification: str = VALID_REUSABLE,
    qualification_reason: str = "exact-prefix D3 sequence verified",
) -> tuple[str, str]:
    path = Path(metadata_path).resolve()
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("data_contract") != D3_EXECUTION_SEMANTICS:
        raise ValueError(f"not a supported D3 metadata contract: {path}")
    verification = meta.get("same_prefix_verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise ValueError(f"D3 metadata lacks passed exact-prefix verification: {path}")
    reference = str(verification.get("reference_metadata_path", ""))
    if not reference:
        raise ValueError(f"D3 metadata lacks reference metadata path: {path}")
    horizon_seconds = int(meta.get("model_horizon_seconds", 0))
    simulation_key, family_key, identity = d3_identity(
        inp_path=str(meta["inp_path"]),
        reference_metadata_path=reference,
        checkpoint_seconds=int(meta["checkpoint_minutes"]) * 60,
        sequence_sha256=str(meta["sequence_sha256"]),
        swmm_engine_version=str(meta["swmm_engine_version"]),
        stride_seconds=int(meta["model_step_seconds"]),
        control_block_seconds=int(meta["control_block_seconds"]),
        horizon_seconds=horizon_seconds,
    )
    registry.register(
        simulation_key=simulation_key,
        family_key=family_key,
        kind="D3_SEQUENCE",
        horizon_seconds=horizon_seconds,
        metadata_path=path,
        identity=identity,
        qualification=qualification,
        qualification_reason=qualification_reason,
    )
    return simulation_key, family_key
