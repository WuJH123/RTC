"""Low-dimensional, engineering-feasible control manifold for Step2 V6.0."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .step2_v60_contract import (
    ControlBasisContractV60,
    MultiResolutionHorizonV60,
    V60_CONTROL_MANIFOLD_CONTRACT,
    require_feature,
)

_ACTUATOR_TYPE_FEATURES = ("is_pump", "is_orifice", "is_weir", "is_outlet")


@dataclass(frozen=True)
class ControlGroupingV60:
    zone_id_by_actuator: np.ndarray
    type_id_by_actuator: np.ndarray
    group_id_by_actuator: np.ndarray
    group_labels: tuple[str, ...]
    medoid_actuator_indices: tuple[int, ...]

    @property
    def actuator_count(self) -> int:
        return int(self.group_id_by_actuator.shape[0])

    @property
    def group_count(self) -> int:
        return len(self.group_labels)

    def validate(self) -> None:
        n = self.actuator_count
        if self.zone_id_by_actuator.shape != (n,) or self.type_id_by_actuator.shape != (n,):
            raise ValueError("control grouping arrays must align with actuators")
        if n <= 0 or self.group_count <= 0:
            raise ValueError("control grouping cannot be empty")
        unique = np.unique(self.group_id_by_actuator)
        if not np.array_equal(unique, np.arange(self.group_count, dtype=unique.dtype)):
            raise ValueError("group ids must be compact [0,G)")


@dataclass(frozen=True)
class ControlBasisV60:
    grouping: ControlGroupingV60
    temporal_basis: np.ndarray
    min_setting: np.ndarray
    max_setting: np.ndarray
    horizon: MultiResolutionHorizonV60
    contract: ControlBasisContractV60

    @property
    def group_count(self) -> int:
        return self.grouping.group_count

    @property
    def temporal_basis_count(self) -> int:
        return int(self.temporal_basis.shape[1])

    @property
    def coefficient_dimension(self) -> int:
        return self.group_count * self.temporal_basis_count

    def validate(self) -> None:
        self.grouping.validate()
        self.horizon.validate()
        self.contract.validate(
            actuator_count=self.grouping.actuator_count,
            control_blocks=self.horizon.control_blocks,
        )
        expected = (self.horizon.control_blocks, self.contract.temporal_basis_count)
        if self.temporal_basis.shape != expected:
            raise ValueError(f"temporal basis shape {self.temporal_basis.shape} != {expected}")
        if self.min_setting.shape != (self.grouping.actuator_count,) or self.max_setting.shape != self.min_setting.shape:
            raise ValueError("actuator bounds do not align with grouping")
        if np.any(self.max_setting < self.min_setting):
            raise ValueError("invalid actuator bounds")
        if not np.allclose(self.temporal_basis.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("temporal basis must form a partition of unity")

    def decode(self, reference_settings: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Decode low-dimensional coefficients into a feasible H72 setting sequence."""
        self.validate()
        if reference_settings.shape[-2:] != (self.horizon.horizon_steps, self.grouping.actuator_count):
            raise ValueError("reference settings do not match V6 horizon/actuator contract")
        if coefficients.shape[-2:] != (self.temporal_basis_count, self.group_count):
            raise ValueError("coefficient tensor does not match V6 basis")
        if reference_settings.shape[:-2] != coefficients.shape[:-2]:
            raise ValueError("reference/coefficient leading dimensions must match")
        dtype, device = reference_settings.dtype, reference_settings.device
        basis = torch.as_tensor(self.temporal_basis, dtype=dtype, device=device)
        group_ids = torch.as_tensor(self.grouping.group_id_by_actuator, dtype=torch.long, device=device)
        lows = torch.as_tensor(self.min_setting, dtype=dtype, device=device)
        highs = torch.as_tensor(self.max_setting, dtype=dtype, device=device)
        coeff = coefficients.clamp(-float(self.contract.coefficient_limit), float(self.contract.coefficient_limit))
        group_delta = torch.einsum("tk,...kg->...tg", basis, coeff)
        actuator_delta = group_delta.index_select(-1, group_ids)
        block_steps = self.horizon.control_block_steps
        reference_blocks = reference_settings[..., ::block_steps, :]
        if reference_blocks.shape[-2] != self.horizon.control_blocks:
            raise RuntimeError("reference block extraction violated horizon contract")
        max_delta = float(self.contract.max_setting_delta_per_update)
        raw = torch.maximum(torch.minimum(reference_blocks + actuator_delta * max_delta, highs), lows)
        projected: list[torch.Tensor] = []
        previous = reference_blocks[..., 0, :]
        for block in range(self.horizon.control_blocks):
            current = previous + (raw[..., block, :] - previous).clamp(-max_delta, max_delta)
            current = torch.maximum(torch.minimum(current, highs), lows)
            projected.append(current)
            previous = current
        return torch.stack(projected, dim=-2).repeat_interleave(block_steps, dim=-2)

    def design_matrix(self) -> np.ndarray:
        t, k = self.temporal_basis.shape
        g, a = self.group_count, self.grouping.actuator_count
        incidence = np.zeros((g, a), dtype=np.float64)
        incidence[self.grouping.group_id_by_actuator, np.arange(a)] = 1.0
        matrix = np.einsum("tk,ga->takg", self.temporal_basis.astype(np.float64), incidence)
        return matrix.reshape(t * a, k * g)

    def project_actions_to_coefficients(self, delta_settings: np.ndarray) -> np.ndarray:
        array = np.asarray(delta_settings, dtype=np.float64)
        if array.shape[-2:] == (self.horizon.horizon_steps, self.grouping.actuator_count):
            array = array[..., :: self.horizon.control_block_steps, :]
        if array.shape[-2:] != (self.horizon.control_blocks, self.grouping.actuator_count):
            raise ValueError("delta settings do not match model/control horizon")
        matrix = self.design_matrix()
        flat = array.reshape(-1, matrix.shape[0]).T
        coeff, *_ = np.linalg.lstsq(matrix, flat, rcond=None)
        return coeff.T.reshape(*array.shape[:-2], self.temporal_basis_count, self.group_count)


def _adjacency(edge_index: np.ndarray, node_count: int) -> list[list[int]]:
    edge = np.asarray(edge_index, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    out = [[] for _ in range(node_count)]
    for src, dst in edge.T.tolist():
        if not 0 <= src < node_count or not 0 <= dst < node_count:
            raise ValueError("edge endpoint outside node range")
        out[src].append(dst)
        out[dst].append(src)
    return [sorted(set(values)) for values in out]


def _distances_from_endpoints(adjacency: list[list[int]], endpoints: Sequence[int]) -> np.ndarray:
    dist = np.full(len(adjacency), np.iinfo(np.int32).max, dtype=np.int32)
    queue: deque[int] = deque()
    for raw in endpoints:
        node = int(raw)
        if not 0 <= node < len(adjacency):
            raise ValueError("actuator endpoint outside node range")
        if dist[node] != 0:
            dist[node] = 0
            queue.append(node)
    while queue:
        node = queue.popleft()
        nxt = int(dist[node]) + 1
        for neighbour in adjacency[node]:
            if dist[neighbour] > nxt:
                dist[neighbour] = nxt
                queue.append(neighbour)
    return dist


def _actuator_type_ids(physics: np.ndarray, names: Sequence[str]) -> np.ndarray:
    indices = [require_feature(names, name) for name in _ACTUATOR_TYPE_FEATURES]
    values = np.asarray(physics, dtype=np.float64)[:, indices]
    type_id = values.argmax(axis=1).astype(np.int64)
    if np.any(values.max(axis=1) <= 0):
        raise ValueError("every actuator must have a recognised type indicator")
    return type_id


def derive_control_grouping_v60(graph: Any, *, spatial_zone_count: int = 8) -> ControlGroupingV60:
    """Topology-aware deterministic actuator zones, then type-specific control groups."""
    actuator_count, node_count = len(graph.actuator_ids), len(graph.node_ids)
    if not 1 <= spatial_zone_count <= actuator_count:
        raise ValueError("invalid spatial_zone_count")
    up = np.asarray(graph.actuator_upstream, dtype=np.int64)
    down = np.asarray(graph.actuator_downstream, dtype=np.int64)
    if up.shape != (actuator_count,) or down.shape != (actuator_count,):
        raise ValueError("actuator endpoint arrays do not align with actuator ids")
    adjacency = _adjacency(np.asarray(graph.edge_index), node_count)
    distances = np.empty((actuator_count, actuator_count), dtype=np.float64)
    unreachable = float(node_count + 1)
    for i in range(actuator_count):
        node_distance = _distances_from_endpoints(adjacency, (int(up[i]), int(down[i])))
        values = np.minimum(node_distance[up], node_distance[down]).astype(np.float64)
        values[values >= np.iinfo(np.int32).max] = unreachable
        distances[i] = values
    distances = np.minimum(distances, distances.T)
    capacity_idx = require_feature(tuple(graph.static_node_feature_names), "storage_capacity_m3")
    static = np.asarray(graph.static_node_features, dtype=np.float64)
    endpoint_capacity = np.maximum(static[up, capacity_idx], static[down, capacity_idx])
    first = int(np.lexsort((np.arange(actuator_count), -endpoint_capacity))[0])
    medoids = [first]
    while len(medoids) < spatial_zone_count:
        nearest = distances[:, medoids].min(axis=1)
        nearest[np.asarray(medoids, dtype=np.int64)] = -1.0
        candidate = int(np.argmax(nearest))
        if candidate in medoids:
            candidate = next(i for i in range(actuator_count) if i not in medoids)
        medoids.append(candidate)
    zone_id = distances[:, medoids].argmin(axis=1).astype(np.int64)
    type_id = _actuator_type_ids(np.asarray(graph.actuator_physics), tuple(graph.actuator_physics_feature_names))
    type_names = ("pump", "orifice", "weir", "outlet")
    keys = [(int(zone_id[i]), int(type_id[i])) for i in range(actuator_count)]
    unique_keys = sorted(set(keys))
    key_to_group = {key: index for index, key in enumerate(unique_keys)}
    group_id = np.asarray([key_to_group[key] for key in keys], dtype=np.int64)
    result = ControlGroupingV60(
        zone_id_by_actuator=zone_id,
        type_id_by_actuator=type_id,
        group_id_by_actuator=group_id,
        group_labels=tuple(f"zone{zone:02d}:{type_names[kind]}" for zone, kind in unique_keys),
        medoid_actuator_indices=tuple(int(i) for i in medoids),
    )
    result.validate()
    return result


def build_temporal_hat_basis_v60(control_blocks: int = 36, basis_count: int = 6) -> np.ndarray:
    if not 2 <= basis_count <= control_blocks:
        raise ValueError("basis_count must lie in [2, control_blocks]")
    x = np.arange(control_blocks, dtype=np.float64)
    centers = np.linspace(0.0, float(control_blocks - 1), basis_count)
    width = float(centers[1] - centers[0])
    basis = np.maximum(1.0 - np.abs(x[:, None] - centers[None, :]) / max(width, 1e-9), 0.0)
    rowsum = basis.sum(axis=1, keepdims=True)
    if np.any(rowsum <= 0):
        raise RuntimeError("temporal basis left an uncovered control block")
    return (basis / rowsum).astype(np.float32)


def build_control_basis_v60(
    graph: Any,
    *,
    horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(),
    contract: ControlBasisContractV60 = ControlBasisContractV60(),
) -> ControlBasisV60:
    horizon.validate()
    contract.validate(actuator_count=len(graph.actuator_ids), control_blocks=horizon.control_blocks)
    grouping = derive_control_grouping_v60(graph, spatial_zone_count=contract.spatial_zone_count)
    names = tuple(graph.actuator_physics_feature_names)
    physics = np.asarray(graph.actuator_physics, dtype=np.float64)
    result = ControlBasisV60(
        grouping=grouping,
        temporal_basis=build_temporal_hat_basis_v60(horizon.control_blocks, contract.temporal_basis_count),
        min_setting=physics[:, require_feature(names, "min_setting")].astype(np.float32),
        max_setting=physics[:, require_feature(names, "max_setting")].astype(np.float32),
        horizon=horizon,
        contract=contract,
    )
    result.validate()
    return result


def basis_manifest_v60(basis: ControlBasisV60) -> dict[str, object]:
    basis.validate()
    return {
        "contract": V60_CONTROL_MANIFOLD_CONTRACT,
        "actuator_count": basis.grouping.actuator_count,
        "spatial_zone_count": int(np.unique(basis.grouping.zone_id_by_actuator).size),
        "control_group_count": basis.group_count,
        "temporal_basis_count": basis.temporal_basis_count,
        "coefficient_dimension": basis.coefficient_dimension,
        "control_blocks": basis.horizon.control_blocks,
        "model_horizon_steps": basis.horizon.horizon_steps,
        "max_setting_delta_per_update": basis.contract.max_setting_delta_per_update,
        "group_labels": list(basis.grouping.group_labels),
        "group_id_by_actuator": basis.grouping.group_id_by_actuator.tolist(),
        "zone_id_by_actuator": basis.grouping.zone_id_by_actuator.tolist(),
        "medoid_actuator_indices": list(basis.grouping.medoid_actuator_indices),
        "temporal_basis": basis.temporal_basis.tolist(),
    }


__all__ = [
    "ControlBasisV60",
    "ControlGroupingV60",
    "basis_manifest_v60",
    "build_control_basis_v60",
    "build_temporal_hat_basis_v60",
    "derive_control_grouping_v60",
]
