"""Load a frozen core Direct-TFV checkpoint for Development runtime."""
from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .step2_tfv_support import DIRECT_TFV_ACTION_SUPPORT_CONTRACT
from .step2_tfv_value import DIRECT_TFV_VALUE_CONTRACT, DirectFacilityTFVValueModel, DirectTFVValueDesign
from .step2_tfv_value_training_v4 import DIRECT_TFV_TRAINING_CONTRACT
from .step2_train_response_v60 import InputNormalizationV60


DIRECT_TFV_RUNTIME_CHECKPOINT_CONTRACT = "PROJECT7_DIRECT_TFV_RUNTIME_CHECKPOINT_V3"
LEGACY_DIRECT_TFV_TRAINING_CONTRACT = "PROJECT7_DIRECT_TFV_CORE_TRAINING_V4"

# Byte-level fingerprint for the source files that define the V11 first-move query, its support,
# target-latch valuation and authoritative runtime adapter.  This is intentionally separate from the
# historical rtc_source_tree_sha256 compatibility alias, which is only a semantic contract hash.
_DIRECT_TFV_FIRST_MOVE_SOURCE_FILES = (
    "step2_tfv_value.py",
    "step2_tfv_support.py",
    "direct_tfv_admission.py",
    "direct_tfv_sequence_support.py",
    "direct_tfv_policy_admission.py",
    "step3_tfv_value_mpc_v3.py",
    "step3_tfv_value_mpc_v4.py",
    "step3_tfv_value_mpc_v5.py",
    "step3_tfv_value_mpc_v6.py",
    "step3_tfv_value_mpc_v7.py",
    "direct_tfv_first_move.py",
    "direct_tfv_first_move_admission.py",
    "step3_tfv_value_mpc_v9.py",
    "controller_v122.py",
    "controller_direct_tfv.py",
    "closed_loop.py",
)


def direct_tfv_first_move_source_sha256() -> str:
    """Hash the exact source implementation used by V11 first-move calibration/execution."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _DIRECT_TFV_FIRST_MOVE_SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Direct-TFV first-move source file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _design_from_payload(payload: dict[str, Any]) -> DirectTFVValueDesign:
    allowed = {field.name for field in fields(DirectTFVValueDesign)}
    values = {key: value for key, value in dict(payload).items() if key in allowed}
    design = DirectTFVValueDesign(**values)
    design.validate()
    return design


def _normalization(payload: dict[str, Any]) -> InputNormalizationV60:
    required = ("state_mean", "state_std", "rainfall_mean", "rainfall_std", "flow_mean", "flow_std")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Direct-TFV checkpoint normalization lacks {missing}")
    arrays = {name: np.asarray(payload[name], dtype=np.float32) for name in required}
    for name, value in arrays.items():
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"Direct-TFV checkpoint normalization {name} is invalid")
    return InputNormalizationV60(**arrays)


def load_direct_tfv_runtime_checkpoint(
    path: str | Path,
    *,
    graph: Any,
    device: torch.device,
) -> tuple[DirectFacilityTFVValueModel, InputNormalizationV60, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Direct-TFV checkpoint payload must be a mapping")
    if str(payload.get("contract")) != DIRECT_TFV_VALUE_CONTRACT:
        raise ValueError("Direct-TFV runtime checkpoint has the wrong model contract")
    training_contract = str(payload.get("training_contract"))
    if training_contract not in {DIRECT_TFV_TRAINING_CONTRACT, LEGACY_DIRECT_TFV_TRAINING_CONTRACT}:
        raise ValueError("Direct-TFV runtime requires the V4 legacy or current V5 training contract")
    if str(payload.get("profile")) != "dev" or payload.get("development_only") is not True:
        raise ValueError("Direct-TFV runtime requires a full Development checkpoint")
    support = payload.get("action_support")
    if not isinstance(support, dict) or str(support.get("contract")) != DIRECT_TFV_ACTION_SUPPORT_CONTRACT:
        raise ValueError("Direct-TFV runtime checkpoint lacks current action-support evidence")
    if int(support.get("single_facility_coverage_count", -1)) != 109:
        raise ValueError("Direct-TFV runtime requires 109/109 exact single-facility TrainFit coverage")
    if tuple(str(value) for value in support.get("actuator_ids", ())) != tuple(str(value) for value in graph.actuator_ids):
        raise ValueError("Direct-TFV runtime action-support actuator order differs from graph")
    design = _design_from_payload(dict(payload["model_design"]))
    model = DirectFacilityTFVValueModel(
        state_dim=int(payload["state_dim"]),
        rainfall_dim=int(payload["rainfall_dim"]),
        actuator_physics_dim=int(payload["actuator_physics_dim"]),
        target_scale_m3=float(payload["target_scale_m3"]),
        design=design,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    normalization = _normalization(dict(payload["normalization"]))
    payload["runtime_loader_contract"] = DIRECT_TFV_RUNTIME_CHECKPOINT_CONTRACT
    payload["runtime_training_contract_is_legacy"] = training_contract == LEGACY_DIRECT_TFV_TRAINING_CONTRACT
    return model, normalization, payload


__all__ = [
    "DIRECT_TFV_RUNTIME_CHECKPOINT_CONTRACT",
    "LEGACY_DIRECT_TFV_TRAINING_CONTRACT",
    "direct_tfv_first_move_source_sha256",
    "load_direct_tfv_runtime_checkpoint",
]
