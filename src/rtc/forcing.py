from __future__ import annotations

from typing import Mapping

import numpy as np

from .units import rainfall_rate_to_mmhr


def resolve_subcatchment_outlets(
    connection: Mapping[str, tuple[int, str] | str],
    node_ids: tuple[str, ...],
) -> dict[str, str | None]:
    valid_nodes = set(node_ids)
    result: dict[str, str | None] = {}
    for sid in connection:
        seen: set[str] = set()
        current = sid
        outlet: str | None = None
        while current not in seen:
            seen.add(current)
            raw_connection = connection[current]
            if isinstance(raw_connection, str):
                target = raw_connection
                if target in valid_nodes:
                    kind = 2
                elif target in connection:
                    kind = 1
                else:
                    kind = 0
            else:
                kind, target = raw_connection
            if int(kind) == 2:
                outlet = target if target in valid_nodes else None
                break
            if int(kind) != 1 or target not in connection:
                break
            current = target
        if current in seen and current != sid and outlet is None:
            # A normal chain terminates before revisiting; this catches malformed cycles.
            pass
        result[sid] = outlet
    return result


def current_node_rainfall_mmhr(
    subcatchment_objects: Mapping[str, object],
    resolved_outlets: Mapping[str, str | None],
    node_ids: tuple[str, ...],
    system_units: str,
) -> np.ndarray:
    """Aggregate current realised gage rainfall to receiving nodes in mm/h.

    This is an online-causal forcing feature. Realised SWMM runoff is deliberately not
    returned because it is an authoritative diagnostic outcome, not an operational input.
    """

    node_index = {node: i for i, node in enumerate(node_ids)}
    values: dict[str, list[float]] = {}
    for sid, obj in subcatchment_objects.items():
        outlet = resolved_outlets.get(sid)
        if outlet is not None:
            values.setdefault(outlet, []).append(float(obj.rainfall))
    result = np.zeros(len(node_ids), dtype=np.float32)
    for node, rainfall in values.items():
        result[node_index[node]] = float(
            rainfall_rate_to_mmhr(float(np.mean(rainfall)), system_units)
        )
    return result
