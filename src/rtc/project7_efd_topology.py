"""Topology-aware storage-to-control mapping for Project7 EFD comparators.

The publication comparator must not assume that a controllable outlet is the link whose
``FromNode`` is the storage itself.  In large SWMM models a storage may drain through one or more
passive conduits/regulators before the first supervisory actuator.  This module therefore maps each
storage to the *first supervisory control boundary on every downstream passive branch*.

Only static INP topology and the frozen supervisory mask are used; no online/full-state hydraulic
information is consumed.  Traversal stops when it reaches another storage or a supervisory
actuator.  Non-supervisory regulator links are treated as passive hydraulic links.  If one
supervisory actuator is reachable from more than one storage, that actuator is excluded rather than
arbitrarily assigned.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .inp import CONTROL_SECTIONS, discover_actuators


EFD_TOPOLOGY_CONTRACT = "PROJECT7_EFD_FIRST_SUPERVISORY_OUTLET_TOPOLOGY_V1"


@dataclass(frozen=True)
class EFDTopologyMap:
    contract: str
    storage_to_actuators: dict[str, tuple[str, ...]]
    actuator_to_storage: dict[str, str]
    all_storage_ids: tuple[str, ...]
    mapped_storage_ids: tuple[str, ...]
    mapped_actuator_ids: tuple[str, ...]
    unmapped_storage_ids: tuple[str, ...]
    ambiguous_actuator_ids: tuple[str, ...]
    supervisory_actuator_count: int
    maximum_passive_hops: int

    def validate_for_equalization(
        self, *, minimum_storages: int = 2, minimum_actuators: int = 2
    ) -> None:
        if self.contract != EFD_TOPOLOGY_CONTRACT:
            raise ValueError("wrong EFD topology contract")
        if len(self.mapped_storage_ids) < int(minimum_storages):
            raise ValueError(
                "topology-aware EFD requires at least "
                f"{minimum_storages} independently controllable storages; "
                f"mapped={len(self.mapped_storage_ids)}"
            )
        if len(self.mapped_actuator_ids) < int(minimum_actuators):
            raise ValueError(
                "topology-aware EFD requires at least "
                f"{minimum_actuators} supervisory outlet actuators; "
                f"mapped={len(self.mapped_actuator_ids)}"
            )
        if len(self.actuator_to_storage) != len(self.mapped_actuator_ids):
            raise ValueError("EFD topology actuator ownership is not one-to-one")
        flattened = [
            actuator_id
            for storage_id in self.mapped_storage_ids
            for actuator_id in self.storage_to_actuators[storage_id]
        ]
        if len(flattened) != len(set(flattened)):
            raise ValueError("EFD topology contains a multiply-owned actuator")


def _iter_rows(path: str | Path) -> Iterable[tuple[str, tuple[str, ...]]]:
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        tokens = tuple(line.split())
        if tokens:
            yield section, tokens


def _storage_ids(path: str | Path) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            tokens[0]
            for section, tokens in _iter_rows(path)
            if section == "STORAGE" and tokens
        )
    )


def _conduit_edges(path: str | Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (tokens[1], tokens[2])
        for section, tokens in _iter_rows(path)
        if section == "CONDUITS" and len(tokens) >= 3
    )


def build_efd_topology_map(
    inp_path: str | Path,
    *,
    actuator_ids: tuple[str, ...] | list[str],
    supervisory_mask: np.ndarray | list[bool],
    maximum_passive_hops: int = 64,
) -> EFDTopologyMap:
    """Map storage units to their first downstream supervisory actuator(s).

    ``actuator_ids`` must use the exact runtime/graph ordering.  Supervisory-disabled SWMM
    regulators are traversed as passive links; supervisory-enabled regulators terminate a branch
    and become an EFD outlet candidate.
    """

    ordered_ids = tuple(str(value) for value in actuator_ids)
    mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
    if len(ordered_ids) == 0 or mask.shape != (len(ordered_ids),):
        raise ValueError("EFD topology requires an actuator-aligned supervisory mask")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("EFD topology actuator IDs are not unique")
    if int(maximum_passive_hops) <= 0:
        raise ValueError("maximum_passive_hops must be positive")

    catalog = discover_actuators(inp_path)
    by_id = {actuator.actuator_id: actuator for actuator in catalog.actuators}
    missing = [actuator_id for actuator_id in ordered_ids if actuator_id not in by_id]
    if missing:
        raise ValueError(
            "runtime actuator order contains IDs absent from the source INP: "
            + ",".join(missing[:8])
        )

    storage_ids = _storage_ids(inp_path)
    if not storage_ids:
        raise ValueError("topology-aware EFD found no [STORAGE] units")
    storage_set = set(storage_ids)

    passive_downstream: dict[str, set[str]] = defaultdict(set)
    for upstream, downstream in _conduit_edges(inp_path):
        passive_downstream[upstream].add(downstream)

    enabled_ids = {
        actuator_id for actuator_id, enabled in zip(ordered_ids, mask, strict=True) if enabled
    }
    # Frozen/non-supervisory regulators remain hydraulic links, so EFD may reach the first
    # *supervisory* boundary downstream of them.
    for actuator in catalog.actuators:
        if actuator.actuator_id not in enabled_ids:
            passive_downstream[actuator.upstream_node].add(actuator.downstream_node)

    enabled_by_upstream: dict[str, list[str]] = defaultdict(list)
    for actuator_id in ordered_ids:
        if actuator_id in enabled_ids:
            enabled_by_upstream[by_id[actuator_id].upstream_node].append(actuator_id)

    preliminary: dict[str, tuple[str, ...]] = {}
    for storage_id in storage_ids:
        found: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(storage_id, 0)])
        visited: set[str] = set()
        while queue:
            node, hops = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            controls_here = enabled_by_upstream.get(node, ())
            if controls_here:
                found.update(controls_here)
                # This branch stops at the first supervisory control boundary.
                continue
            if hops >= int(maximum_passive_hops):
                continue
            for downstream in sorted(passive_downstream.get(node, ())):
                # Once flow reaches another storage, its downstream controls belong to that
                # storage rather than the upstream storage that fed it.
                if downstream in storage_set and downstream != storage_id:
                    continue
                queue.append((downstream, hops + 1))
        preliminary[storage_id] = tuple(sorted(found))

    owners: dict[str, list[str]] = defaultdict(list)
    for storage_id, candidates in preliminary.items():
        for actuator_id in candidates:
            owners[actuator_id].append(storage_id)
    ambiguous = tuple(sorted(aid for aid, values in owners.items() if len(set(values)) > 1))
    ambiguous_set = set(ambiguous)

    storage_to_actuators: dict[str, tuple[str, ...]] = {}
    for storage_id in storage_ids:
        clean = tuple(aid for aid in preliminary[storage_id] if aid not in ambiguous_set)
        if clean:
            storage_to_actuators[storage_id] = clean

    actuator_to_storage = {
        actuator_id: storage_id
        for storage_id, actuator_list in storage_to_actuators.items()
        for actuator_id in actuator_list
    }
    mapped_storage_ids = tuple(
        storage_id for storage_id in storage_ids if storage_id in storage_to_actuators
    )
    mapped_actuator_ids = tuple(
        actuator_id for actuator_id in ordered_ids if actuator_id in actuator_to_storage
    )
    unmapped_storage_ids = tuple(
        storage_id for storage_id in storage_ids if storage_id not in storage_to_actuators
    )

    result = EFDTopologyMap(
        contract=EFD_TOPOLOGY_CONTRACT,
        storage_to_actuators=storage_to_actuators,
        actuator_to_storage=actuator_to_storage,
        all_storage_ids=storage_ids,
        mapped_storage_ids=mapped_storage_ids,
        mapped_actuator_ids=mapped_actuator_ids,
        unmapped_storage_ids=unmapped_storage_ids,
        ambiguous_actuator_ids=ambiguous,
        supervisory_actuator_count=int(mask.sum()),
        maximum_passive_hops=int(maximum_passive_hops),
    )
    result.validate_for_equalization()
    return result


__all__ = [
    "EFD_TOPOLOGY_CONTRACT",
    "EFDTopologyMap",
    "build_efd_topology_map",
]
