"""Project7 native-control supervisory scope.

The frozen hydraulic representation keeps all 109 actuator channels used by the pretrained Step2.
Only links that appear as action targets in the source SWMM ``[CONTROLS]`` section are allowed to
change under the paper-facing supervisory RTC.  All other channels remain part of the hydraulic
context but are forced to ``candidate == reference``.

This is intentionally a scope restriction, not a model redesign: no Step1/Step2 retraining and no
additional control objective are implied by this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUPERVISORY_CONTROL_SCOPE_CONTRACT = (
    "PROJECT7_NATIVE_CONTROLS_82DOF_IN_109CHANNEL_SCOPE_V1"
)
PROJECT7_EXPECTED_SUPERVISORY_COUNTS = {
    "PUMP": 57,
    "ORIFICE": 16,
    "WEIR": 9,
    "OUTLET": 0,
}
PROJECT7_EXPECTED_SUPERVISORY_TOTAL = 82

_ACTION_RE = re.compile(
    r"\b(?:THEN|ELSE|AND)\s+(PUMP|ORIFICE|WEIR|OUTLET)\s+(\S+)\s+"
    r"(?:STATUS|SETTING)\s*=",
    flags=re.IGNORECASE,
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _controls_lines(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    result: list[str] = []
    in_controls = False
    found = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            in_controls = section == "CONTROLS"
            found = found or in_controls
            continue
        if in_controls:
            result.append(raw)
    if not found:
        raise ValueError("native-controls template has no [CONTROLS] section")
    return result


def parse_native_control_action_targets(path: str | Path) -> dict[str, str]:
    """Return ``actuator_id -> SWMM object kind`` for native control action targets."""
    targets: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for raw in _controls_lines(path):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        for match in _ACTION_RE.finditer(line):
            kind = match.group(1).upper()
            actuator_id = match.group(2).strip()
            key = actuator_id.casefold()
            previous = canonical.get(key)
            if previous is not None and targets[previous] != kind:
                raise ValueError(
                    f"native control target {actuator_id!r} appears with multiple object kinds"
                )
            canonical[key] = actuator_id
            targets[actuator_id] = kind
    if not targets:
        raise ValueError("native [CONTROLS] contains no pump/orifice/weir/outlet action targets")
    return targets


def build_supervisory_control_scope(
    *,
    actuator_ids: Sequence[str],
    native_controls_inp: str | Path,
    expected_total: int = PROJECT7_EXPECTED_SUPERVISORY_TOTAL,
    expected_kind_counts: Mapping[str, int] = PROJECT7_EXPECTED_SUPERVISORY_COUNTS,
) -> dict[str, Any]:
    ids = tuple(str(value) for value in actuator_ids)
    if len(ids) != 109 or len(set(ids)) != 109:
        raise ValueError("Project7 supervisory scope requires the frozen 109-channel actuator order")
    targets = parse_native_control_action_targets(native_controls_inp)
    graph_by_fold = {value.casefold(): value for value in ids}
    unknown = sorted(value for value in targets if value.casefold() not in graph_by_fold)
    if unknown:
        raise ValueError(f"native [CONTROLS] targets are absent from the frozen graph: {unknown}")

    kind_by_graph_id: dict[str, str] = {}
    for raw_id, kind in targets.items():
        kind_by_graph_id[graph_by_fold[raw_id.casefold()]] = str(kind).upper()
    mask = np.asarray([value in kind_by_graph_id for value in ids], dtype=bool)
    controlled = tuple(value for value, active in zip(ids, mask, strict=True) if bool(active))
    passive = tuple(value for value, active in zip(ids, mask, strict=True) if not bool(active))
    counts = {
        kind: int(sum(kind_by_graph_id.get(value) == kind for value in controlled))
        for kind in ("PUMP", "ORIFICE", "WEIR", "OUTLET")
    }
    expected_counts = {str(key).upper(): int(value) for key, value in expected_kind_counts.items()}
    if int(mask.sum()) != int(expected_total):
        raise ValueError(
            f"native supervisory scope count drift: {int(mask.sum())} != expected {int(expected_total)}"
        )
    if counts != expected_counts:
        raise ValueError(f"native supervisory kind counts drift: {counts} != {expected_counts}")

    return {
        "contract": SUPERVISORY_CONTROL_SCOPE_CONTRACT,
        "development_only": True,
        "model_actuator_channel_count": 109,
        "supervisory_facility_count": int(mask.sum()),
        "passive_hydraulic_channel_count": int((~mask).sum()),
        "supervisory_kind_counts": counts,
        "actuator_ids": list(ids),
        "supervisory_mask": mask.astype(np.int8).tolist(),
        "supervisory_actuator_ids": list(controlled),
        "passive_hydraulic_actuator_ids": list(passive),
        "scope_semantics": "82_SUPERVISORY_DOF_EMBEDDED_IN_FROZEN_109_CHANNEL_REPRESENTATION",
        "passive_channel_action_semantics": "CANDIDATE_EQUALS_REFERENCE",
        "source_semantics": "ACTION_TARGETS_IN_NATIVE_SWMM_CONTROLS_ONLY",
        "native_controls_template_path": str(Path(native_controls_inp).resolve()),
        "native_controls_template_sha256": _sha(native_controls_inp),
        "step1_retraining_required": False,
        "base_step2_retraining_required": False,
    }


def validate_supervisory_control_scope(
    payload: Mapping[str, Any], *, actuator_ids: Sequence[str]
) -> None:
    if str(payload.get("contract", "")) != SUPERVISORY_CONTROL_SCOPE_CONTRACT:
        raise ValueError("wrong Project7 supervisory-control-scope contract")
    ids = tuple(str(value) for value in payload.get("actuator_ids", ()))
    expected = tuple(str(value) for value in actuator_ids)
    if ids != expected or len(ids) != 109:
        raise ValueError("supervisory scope actuator order differs from frozen graph")
    mask = np.asarray(payload.get("supervisory_mask", ()), dtype=np.int8).reshape(-1)
    if mask.shape != (109,) or not np.isin(mask, (0, 1)).all():
        raise ValueError("supervisory scope mask must contain 109 binary entries")
    if int(mask.sum()) != PROJECT7_EXPECTED_SUPERVISORY_TOTAL:
        raise ValueError("supervisory scope does not contain the frozen 82 online facilities")
    if int(payload.get("model_actuator_channel_count", -1)) != 109:
        raise ValueError("supervisory scope changed the frozen Step2 channel dimension")
    if int(payload.get("passive_hydraulic_channel_count", -1)) != 27:
        raise ValueError("supervisory scope does not retain exactly 27 passive hydraulic channels")
    counts = {
        str(key).upper(): int(value)
        for key, value in dict(payload.get("supervisory_kind_counts", {})).items()
    }
    if counts != PROJECT7_EXPECTED_SUPERVISORY_COUNTS:
        raise ValueError("supervisory scope object-kind census differs from the frozen INP evidence")
    source_path = Path(str(payload.get("native_controls_template_path", "")))
    expected_sha = str(payload.get("native_controls_template_sha256", "")).lower()
    if not source_path.is_absolute() or not source_path.is_file():
        raise FileNotFoundError("supervisory scope native-controls template path is unavailable")
    if _sha(source_path) != expected_sha:
        raise ValueError("supervisory scope native-controls template SHA drift")


def load_supervisory_control_scope(
    path: str | Path, *, actuator_ids: Sequence[str]
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("supervisory scope JSON must be an object")
    validate_supervisory_control_scope(payload, actuator_ids=actuator_ids)
    return payload


def supervisory_mask_array(payload: Mapping[str, Any]) -> np.ndarray:
    mask = np.asarray(payload.get("supervisory_mask", ()), dtype=np.int8).reshape(-1)
    if mask.shape != (109,) or not np.isin(mask, (0, 1)).all():
        raise ValueError("invalid supervisory mask")
    return mask.astype(bool)


__all__ = [
    "PROJECT7_EXPECTED_SUPERVISORY_COUNTS",
    "PROJECT7_EXPECTED_SUPERVISORY_TOTAL",
    "SUPERVISORY_CONTROL_SCOPE_CONTRACT",
    "build_supervisory_control_scope",
    "load_supervisory_control_scope",
    "parse_native_control_action_targets",
    "supervisory_mask_array",
    "validate_supervisory_control_scope",
]
