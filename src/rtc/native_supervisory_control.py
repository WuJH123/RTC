"""Native supervisory-control contract for Project7.

Project7 keeps the frozen 109-channel hydraulic action representation used by the pretrained Step2,
but only facilities with explicit actions in the source SWMM ``[CONTROLS]`` section are allowed to
change online. The remaining setting channels stay in the model/hydraulic representation while
candidate and reference targets are forced equal on those channels.

This is intentionally a *control-space mask*, not a new hydraulic model. It therefore does not change
the network topology, the five added storage nodes, Step1, or the 109-channel Step2 tensor contract.
The mask is matched case-insensitively to INP identifiers and serialized in the frozen graph order so
its SHA is stable even when native control-rule ordering differs from link ordering.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .inp import discover_actuators


NATIVE_SUPERVISORY_CONTROL_CONTRACT = (
    "PROJECT7_NATIVE_SWMM_CONTROLS_82_OF_109_SUPERVISORY_MASK_V1"
)
PROJECT7_MODEL_ACTION_CHANNEL_COUNT = 109
PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION = 82
_ACTION_OBJECTS = {"PUMP", "ORIFICE", "WEIR", "OUTLET"}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iter_control_actions(path: str | Path):
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        if section != "CONTROLS":
            continue
        tokens = line.split()
        if len(tokens) < 3:
            continue
        lead = tokens[0].upper()
        obj = tokens[1].upper()
        if lead not in {"THEN", "ELSE", "AND"} or obj not in _ACTION_OBJECTS:
            continue
        yield obj, tokens[2]


def derive_native_supervisory_control(
    inp_path: str | Path,
    *,
    actuator_ids: Sequence[str],
    expected_control_dimension: int | None = PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION,
) -> dict[str, Any]:
    ids = tuple(str(value) for value in actuator_ids)
    if len(ids) != PROJECT7_MODEL_ACTION_CHANNEL_COUNT or len(set(ids)) != len(ids):
        raise ValueError("Project7 native supervisory control requires 109 unique model channels")

    catalog = discover_actuators(inp_path)
    catalog_by_fold = {row.actuator_id.casefold(): row for row in catalog.actuators}
    graph_by_fold = {value.casefold(): value for value in ids}
    if len(graph_by_fold) != len(ids):
        raise ValueError("Project7 graph actuator IDs are not unique under case-insensitive matching")
    missing = sorted(value for value in ids if value.casefold() not in catalog_by_fold)
    if missing:
        raise ValueError(f"source INP does not contain graph action channels: {missing[:10]}")

    controlled_fold: set[str] = set()
    action_kind_by_fold: dict[str, str] = {}
    non_model_actions: list[dict[str, str]] = []
    for obj, raw_actuator_id in _iter_control_actions(inp_path):
        key = raw_actuator_id.casefold()
        graph_id = graph_by_fold.get(key)
        if graph_id is None:
            non_model_actions.append({"object": obj, "id": raw_actuator_id})
            continue
        previous_kind = action_kind_by_fold.get(key)
        if previous_kind is not None and previous_kind != obj:
            raise ValueError(
                f"native control target {raw_actuator_id!r} appears with multiple object kinds"
            )
        controlled_fold.add(key)
        action_kind_by_fold[key] = obj

    if not controlled_fold:
        raise ValueError("source INP [CONTROLS] contains no actions on Project7 model channels")

    # Canonicalize all current artifacts to the pretrained/frozen graph order. This guarantees that
    # candidate masks, support statistics and lineage hashes do not depend on rule-file ordering.
    mask = np.asarray([value.casefold() in controlled_fold for value in ids], dtype=bool)
    controlled = tuple(value for value, enabled in zip(ids, mask.tolist(), strict=True) if enabled)
    passive = tuple(value for value, enabled in zip(ids, mask.tolist(), strict=True) if not enabled)
    dimension = int(mask.sum())
    if expected_control_dimension is not None and dimension != int(expected_control_dimension):
        raise ValueError(
            "native supervisory-control dimension differs from the frozen Project7 contract: "
            f"observed={dimension} expected={int(expected_control_dimension)}"
        )

    kind_counts: dict[str, int] = {}
    for actuator_id in controlled:
        kind = str(catalog_by_fold[actuator_id.casefold()].kind)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    mask_bytes = np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    return {
        "contract": NATIVE_SUPERVISORY_CONTROL_CONTRACT,
        "development_only": True,
        "source_semantics": "FACILITY_HAS_EXPLICIT_ACTION_IN_SOURCE_INP_CONTROLS",
        "model_action_channel_count": PROJECT7_MODEL_ACTION_CHANNEL_COUNT,
        "supervisory_control_dimension": dimension,
        "passive_setting_channel_count": int((~mask).sum()),
        "actuator_ids": list(ids),
        "supervisory_mask": mask.astype(int).tolist(),
        "supervisory_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "controlled_actuator_ids": list(controlled),
        "passive_setting_channel_ids": list(passive),
        "controlled_kind_counts": dict(sorted(kind_counts.items())),
        "non_model_control_actions": non_model_actions,
        "source_inp_path": str(Path(inp_path).resolve()),
        "source_inp_sha256": sha256_file(inp_path),
        "scientific_role": (
            "Keep the pretrained 109-channel hydraulic action representation, but allow online "
            "candidate/reference differences only on facilities with native SWMM control evidence."
        ),
        "step1_retraining_required": False,
        "base_step2_retraining_required": False,
    }


def validate_native_supervisory_control(
    payload: Mapping[str, Any],
    *,
    actuator_ids: Sequence[str],
    expected_control_dimension: int | None = PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION,
) -> np.ndarray:
    if str(payload.get("contract", "")) != NATIVE_SUPERVISORY_CONTROL_CONTRACT:
        raise ValueError("wrong native supervisory-control contract")
    ids = tuple(str(value) for value in payload.get("actuator_ids", ()))
    expected_ids = tuple(str(value) for value in actuator_ids)
    if ids != expected_ids or len(ids) != PROJECT7_MODEL_ACTION_CHANNEL_COUNT:
        raise ValueError("native supervisory-control actuator order differs from the 109-channel graph")
    raw = np.asarray(payload.get("supervisory_mask", ()), dtype=np.int64).reshape(-1)
    if raw.shape != (PROJECT7_MODEL_ACTION_CHANNEL_COUNT,) or not np.all(np.isin(raw, (0, 1))):
        raise ValueError("native supervisory-control mask must contain 109 binary entries")
    mask = raw.astype(bool)
    dimension = int(mask.sum())
    if int(payload.get("supervisory_control_dimension", -1)) != dimension:
        raise ValueError("native supervisory-control dimension does not match its mask")
    if int(payload.get("passive_setting_channel_count", -1)) != int((~mask).sum()):
        raise ValueError("native passive-setting count does not match its mask")
    if expected_control_dimension is not None and dimension != int(expected_control_dimension):
        raise ValueError(
            f"current Project7 requires {int(expected_control_dimension)} supervisory facilities, got {dimension}"
        )
    expected_sha = hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()
    if str(payload.get("supervisory_mask_sha256", "")).lower() != expected_sha:
        raise ValueError("native supervisory-control mask SHA is inconsistent")
    controlled = tuple(str(value) for value in payload.get("controlled_actuator_ids", ()))
    passive = tuple(str(value) for value in payload.get("passive_setting_channel_ids", ()))
    expected_controlled = tuple(value for value, enabled in zip(ids, mask.tolist(), strict=True) if enabled)
    expected_passive = tuple(value for value, enabled in zip(ids, mask.tolist(), strict=True) if not enabled)
    if controlled != expected_controlled or passive != expected_passive:
        raise ValueError("native supervisory-control ID lists differ from the frozen mask/order")
    return mask


def load_native_supervisory_control(
    path: str | Path,
    *,
    actuator_ids: Sequence[str],
    expected_control_dimension: int | None = PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION,
) -> tuple[dict[str, Any], np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("native supervisory-control artifact must contain one JSON object")
    mask = validate_native_supervisory_control(
        payload,
        actuator_ids=actuator_ids,
        expected_control_dimension=expected_control_dimension,
    )
    return payload, mask


__all__ = [
    "NATIVE_SUPERVISORY_CONTROL_CONTRACT",
    "PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION",
    "PROJECT7_MODEL_ACTION_CHANNEL_COUNT",
    "derive_native_supervisory_control",
    "load_native_supervisory_control",
    "sha256_file",
    "validate_native_supervisory_control",
]
