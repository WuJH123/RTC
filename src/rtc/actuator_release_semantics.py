"""Type-aware mapping between hydraulic release intent and SWMM actuator SETTING.

SWMM does not give SETTING the same physical meaning for every controllable link.  Pump,
orifice and outlet settings increase their discharge multiplier/opening when SETTING increases,
while a weir SETTING is the fraction of the original freeboard: reducing SETTING lowers the
controlled crest and therefore represents the release-increasing direction.  Keeping this mapping
explicit prevents hydraulic heuristics from treating every 0..1 control as an interchangeable
"opening" variable.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


RELEASE_SETTING_SEMANTICS_CONTRACT = "PROJECT7_SWMM_TYPE_AWARE_RELEASE_SETTING_V1"
_SUPPORTED_KINDS = {"pump", "orifice", "weir", "outlet"}


def release_setting_sign(kind: str) -> float:
    """Return d(SETTING)/d(release-intent) for one SWMM actuator type."""

    value = str(kind).strip().lower()
    if value not in _SUPPORTED_KINDS:
        raise ValueError(f"unsupported SWMM actuator kind for release semantics: {kind!r}")
    return -1.0 if value == "weir" else 1.0


def release_fraction_to_setting(kind: str, release_fraction: float) -> float:
    """Map a normalized desired release fraction to the actuator's SWMM SETTING coordinate."""

    release = float(np.clip(float(release_fraction), 0.0, 1.0))
    return float(1.0 - release) if str(kind).strip().lower() == "weir" else release


def release_intent_delta_to_setting_delta(kind: str, release_delta: float) -> float:
    """Map a signed local release-intent increment to the correct SETTING direction."""

    return float(release_setting_sign(kind) * float(release_delta))


def graph_release_setting_signs(graph) -> np.ndarray:
    """Return one release-to-setting sign per graph actuator from frozen type indicators."""

    names = tuple(str(value) for value in getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(getattr(graph, "actuator_physics", ()), dtype=np.float64)
    actuator_ids = tuple(str(value) for value in getattr(graph, "actuator_ids", ()))
    if physics.ndim != 2 or physics.shape[0] != len(actuator_ids):
        raise ValueError("graph actuator physics is not aligned with actuator IDs")
    type_columns = {
        kind: names.index(f"is_{kind}")
        for kind in ("pump", "orifice", "weir", "outlet")
        if f"is_{kind}" in names
    }
    if len(type_columns) != 4:
        raise ValueError("graph lacks complete pump/orifice/weir/outlet type indicators")
    result = np.empty(len(actuator_ids), dtype=np.float64)
    for row in range(len(actuator_ids)):
        kinds = [kind for kind, column in type_columns.items() if physics[row, column] > 0.5]
        if len(kinds) != 1:
            raise ValueError(f"actuator {actuator_ids[row]!r} has ambiguous SWMM type indicators")
        result[row] = release_setting_sign(kinds[0])
    return result


def catalog_release_setting_signs(kinds: Sequence[str]) -> np.ndarray:
    """Return release-to-setting signs for an INP actuator-kind sequence."""

    return np.asarray([release_setting_sign(value) for value in kinds], dtype=np.float64)


__all__ = [
    "RELEASE_SETTING_SEMANTICS_CONTRACT",
    "catalog_release_setting_signs",
    "graph_release_setting_signs",
    "release_fraction_to_setting",
    "release_intent_delta_to_setting_delta",
    "release_setting_sign",
]
