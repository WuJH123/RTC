from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .inp import ActuatorCatalog, discover_actuators
from .units import length_to_m, volume_to_m3

AUTO_RBC_SOURCE = "AUTO_RBC_V2_TARGET_LATCH"
AUTO_RBC_CONTRACT = "AUTO_RBC_TOPOLOGY_NORMALIZED_DEPTH_TARGET_COMMAND_V2"
EFD_SOURCE = "EFD_V3_TARGET_LATCH"
EFD_CONTRACT = "STORAGE_EQUAL_FILLING_DEGREE_VOLUME_TARGET_COMMAND_V3"


@dataclass(frozen=True)
class StorageGeometry:
    shape: str
    max_depth_native: float
    functional_a1: float = 0.0
    functional_a2: float = 0.0
    functional_a0: float = 0.0
    tabular_points: tuple[tuple[float, float], ...] = ()
    system_units: str = "SI"

    def _depth_native(self, depth_m: float) -> float:
        if self.system_units == "SI":
            return float(np.clip(depth_m, 0.0, self.max_depth_native))
        return float(np.clip(depth_m / 0.3048, 0.0, self.max_depth_native))

    def volume_m3(self, depth_m: float) -> float:
        depth = self._depth_native(depth_m)
        if depth <= 0.0:
            return 0.0
        shape = self.shape.upper()
        volume_native = 0.0
        if shape == "FUNCTIONAL":
            if self.functional_a2 > -1.0:
                volume_native = (
                    self.functional_a1 * depth ** (self.functional_a2 + 1.0)
                    / (self.functional_a2 + 1.0)
                    + self.functional_a0 * depth
                )
        elif shape == "TABULAR":
            volume_native = _tabular_volume(self.tabular_points, depth)
        return float(volume_to_m3(max(volume_native, 0.0), self.system_units))

    @property
    def capacity_m3(self) -> float:
        depth_m = float(length_to_m(self.max_depth_native, self.system_units))
        return self.volume_m3(depth_m)


@dataclass(frozen=True)
class RuleNetwork:
    catalog: ActuatorCatalog
    node_max_depth_m: dict[str, float]
    storage_ids: tuple[str, ...]
    storage_geometry: dict[str, StorageGeometry]


def _iter_rows(path: str | Path):
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        tokens = line.split()
        if tokens:
            yield section, tokens


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _curve_catalog(path: str | Path) -> dict[str, tuple[tuple[float, float], ...]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for section, tokens in _iter_rows(path):
        if section != "CURVES" or len(tokens) < 3:
            continue
        offset = 2 if len(tokens) >= 4 and not _is_number(tokens[1]) else 1
        if len(tokens) <= offset + 1:
            continue
        try:
            x, area = float(tokens[offset]), float(tokens[offset + 1])
        except ValueError:
            continue
        curves.setdefault(tokens[0], []).append((x, area))
    return {key: tuple(values) for key, values in curves.items()}


def _tabular_volume(points: tuple[tuple[float, float], ...], depth: float) -> float:
    if not points or depth <= 0.0:
        return 0.0
    pts = sorted((max(0.0, float(x)), max(0.0, float(a))) for x, a in points)
    if pts[0][0] > 0.0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < depth:
        pts.append((depth, pts[-1][1]))
    samples: list[tuple[float, float]] = []
    for j, (x, area) in enumerate(pts):
        if x <= depth:
            samples.append((x, area))
            continue
        x0, a0 = pts[j - 1]
        fraction = 0.0 if x == x0 else (depth - x0) / (x - x0)
        samples.append((depth, a0 + fraction * (area - a0)))
        break
    if samples[-1][0] < depth:
        samples.append((depth, samples[-1][1]))
    return float(
        sum(
            0.5 * (a0 + a1) * max(0.0, x1 - x0)
            for (x0, a0), (x1, a1) in zip(samples[:-1], samples[1:], strict=False)
        )
    )


def _system_units(path: str | Path) -> str:
    flow_units = ""
    for section, tokens in _iter_rows(path):
        if (
            section == "OPTIONS"
            and len(tokens) >= 2
            and tokens[0].upper() == "FLOW_UNITS"
        ):
            flow_units = tokens[1].upper()
            break
    if flow_units in {"CMS", "LPS", "MLD"}:
        return "SI"
    if flow_units in {"CFS", "GPM", "MGD"}:
        return "US"
    raise ValueError(f"cannot infer SWMM system units from FLOW_UNITS in {path}")


def load_rule_network(path: str | Path) -> RuleNetwork:
    units = _system_units(path)
    curves = _curve_catalog(path)
    max_depth: dict[str, float] = {}
    storage: list[str] = []
    geometry: dict[str, StorageGeometry] = {}
    for section, tokens in _iter_rows(path):
        if section not in {"JUNCTIONS", "STORAGE"} or len(tokens) < 3:
            continue
        node_id = str(tokens[0])
        try:
            depth_native = float(tokens[2])
        except ValueError:
            continue
        depth_m = float(length_to_m(depth_native, units))
        if depth_m > 0:
            max_depth[node_id] = depth_m
        if section != "STORAGE":
            continue
        storage.append(node_id)
        shape = tokens[4].upper() if len(tokens) > 4 else ""
        if shape == "FUNCTIONAL" and len(tokens) > 7:
            geometry[node_id] = StorageGeometry(
                shape=shape,
                max_depth_native=max(0.0, depth_native),
                functional_a1=float(tokens[5]),
                functional_a2=float(tokens[6]),
                functional_a0=float(tokens[7]),
                system_units=units,
            )
        elif shape == "TABULAR" and len(tokens) > 5:
            geometry[node_id] = StorageGeometry(
                shape=shape,
                max_depth_native=max(0.0, depth_native),
                tabular_points=curves.get(tokens[5], ()),
                system_units=units,
            )
    return RuleNetwork(
        catalog=discover_actuators(path),
        node_max_depth_m=max_depth,
        storage_ids=tuple(dict.fromkeys(storage)),
        storage_geometry=geometry,
    )


def rule_baseline_sensor_nodes(
    strategy: str, inp_path: str | Path
) -> tuple[str, ...]:
    network = load_rule_network(inp_path)
    if strategy == "auto_rbc":
        nodes: list[str] = []
        for actuator in network.catalog.actuators:
            for node in (actuator.upstream_node, actuator.downstream_node):
                if node in network.node_max_depth_m and node not in nodes:
                    nodes.append(node)
        if not nodes:
            raise ValueError(
                "Auto-RBC found no actuator-adjacent nodes with a valid design max depth"
            )
        return tuple(nodes)
    if strategy == "efd":
        controlled = {
            actuator.upstream_node
            for actuator in network.catalog.actuators
            if actuator.upstream_node in set(network.storage_ids)
        }
        nodes = tuple(node for node in network.storage_ids if node in controlled)
        if not nodes:
            raise ValueError(
                "EFD requires at least one storage node with a writable outgoing actuator"
            )
        return nodes
    return ()


def _limit_move(
    command_anchor: np.ndarray, target: np.ndarray, max_delta: float | None
) -> np.ndarray:
    """Limit consecutive supervisory commands, not physical device tracking lag."""
    anchor = np.asarray(command_anchor, dtype=float).reshape(-1)
    target = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    if target.shape != anchor.shape:
        raise ValueError("rule target/command-anchor shapes differ")
    if max_delta is None:
        return target
    delta = float(max_delta)
    if delta < 0:
        raise ValueError("max setting delta must be non-negative")
    return np.clip(target, anchor - delta, anchor + delta).clip(0.0, 1.0)


class AutoRBCController:
    """Automatically parameterized local causal rule-based control."""

    def __init__(
        self,
        inp_path: str | Path,
        *,
        low_fill: float = 0.25,
        high_fill: float = 0.75,
        downstream_congestion_fill: float = 0.90,
        response: float = 0.60,
        max_delta_per_update: float | None = None,
    ) -> None:
        if not (0 <= low_fill < high_fill <= 1.0):
            raise ValueError("Auto-RBC filling thresholds are invalid")
        if not 0 < downstream_congestion_fill < 1.0:
            raise ValueError(
                "Auto-RBC downstream congestion threshold must lie in (0,1)"
            )
        if not 0 < response <= 1.0:
            raise ValueError("Auto-RBC response must lie in (0,1]")
        self.network = load_rule_network(inp_path)
        self.sensor_nodes = rule_baseline_sensor_nodes("auto_rbc", inp_path)
        self.low_fill = float(low_fill)
        self.high_fill = float(high_fill)
        self.downstream_congestion_fill = float(downstream_congestion_fill)
        self.response = float(response)
        self.max_delta_per_update = max_delta_per_update

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        if tuple(obs.actuator_ids) != self.network.catalog.ids:
            raise ValueError("Auto-RBC actuator ordering differs from the INP catalog")
        sensor_depth = dict(
            zip(obs.sensor_ids, np.asarray(obs.sensor_depth_m, dtype=float), strict=True)
        )
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        active_target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
        if active_target.shape != current.shape:
            raise ValueError("Auto-RBC target/current readback shapes differ")
        # Untouched actuators preserve the supervisory latch; they are not reset to a lagged
        # physical current setting at every 10-minute decision.
        target = active_target.copy()
        upstream_fill: list[float] = []
        downstream_fill: list[float] = []
        for i, actuator in enumerate(self.network.catalog.actuators):
            up_cap = self.network.node_max_depth_m.get(actuator.upstream_node)
            if up_cap is None or actuator.upstream_node not in sensor_depth:
                continue
            up = float(np.clip(sensor_depth[actuator.upstream_node] / up_cap, 0.0, 1.5))
            down_cap = self.network.node_max_depth_m.get(actuator.downstream_node)
            if down_cap is not None and actuator.downstream_node in sensor_depth:
                down = float(np.clip(sensor_depth[actuator.downstream_node] / down_cap, 0.0, 1.5))
            else:
                down = 0.0
            open_drive = float(
                np.clip((up - self.low_fill) / (self.high_fill - self.low_fill), 0.0, 1.0)
            )
            downstream_penalty = float(
                np.clip(
                    (down - self.downstream_congestion_fill)
                    / (1.0 - self.downstream_congestion_fill),
                    0.0,
                    1.0,
                )
            )
            raw = open_drive * (1.0 - downstream_penalty)
            target[i] = active_target[i] + self.response * (raw - active_target[i])
            upstream_fill.append(up)
            downstream_fill.append(down)
        target = _limit_move(active_target, target, self.max_delta_per_update)
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, target, strict=True)),
            source=AUTO_RBC_SOURCE,
            diagnostics={
                "rule_contract": AUTO_RBC_CONTRACT,
                "observed_rule_nodes": len(self.sensor_nodes),
                "mean_upstream_fill": float(np.mean(upstream_fill)) if upstream_fill else 0.0,
                "max_downstream_fill": float(np.max(downstream_fill)) if downstream_fill else 0.0,
                "command_anchor": "actuator_target_setting",
                "current_tracking_lag_max": float(np.abs(current-active_target).max(initial=0.0)),
            },
        )


class EFDController:
    """Storage-volume Equal Filling Degree baseline.

    Filling degree is current static storage volume divided by storage capacity. Current
    volume is reconstructed causally from monitored storage depth and FUNCTIONAL/TABULAR
    SWMM geometry. Only storage units with writable outgoing actuators are controlled;
    other actuators preserve their current supervisory target latch.
    """

    def __init__(
        self,
        inp_path: str | Path,
        *,
        equalization_gain: float = 1.0,
        response: float = 0.60,
        max_delta_per_update: float | None = None,
    ) -> None:
        if equalization_gain < 0:
            raise ValueError("EFD equalization gain must be non-negative")
        if not 0 < response <= 1.0:
            raise ValueError("EFD response must lie in (0,1]")
        self.network = load_rule_network(inp_path)
        self.sensor_nodes = rule_baseline_sensor_nodes("efd", inp_path)
        self.equalization_gain = float(equalization_gain)
        self.response = float(response)
        self.max_delta_per_update = max_delta_per_update
        storage_set = set(self.sensor_nodes)
        self.outgoing = {
            storage: tuple(
                i
                for i, actuator in enumerate(self.network.catalog.actuators)
                if actuator.upstream_node == storage
            )
            for storage in storage_set
        }

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        if tuple(obs.actuator_ids) != self.network.catalog.ids:
            raise ValueError("EFD actuator ordering differs from the INP catalog")
        depth = dict(
            zip(obs.sensor_ids, np.asarray(obs.sensor_depth_m, dtype=float), strict=True)
        )
        fill: dict[str, float] = {}
        depth_fallback_count = 0
        for node in self.sensor_nodes:
            geometry = self.network.storage_geometry.get(node)
            capacity = 0.0 if geometry is None else geometry.capacity_m3
            if geometry is not None and capacity > 1.0e-9:
                filling = geometry.volume_m3(depth[node]) / capacity
            else:
                # Retain a transparent fallback for malformed legacy INP geometry rather
                # than silently crashing a development comparison; evidence reports it.
                capacity_depth = self.network.node_max_depth_m[node]
                filling = depth[node] / capacity_depth
                depth_fallback_count += 1
            fill[node] = float(np.clip(filling, 0.0, 1.5))
        values = np.asarray(list(fill.values()), dtype=float)
        mean_fill = float(values.mean())
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        active_target = np.asarray(obs.actuator_target_setting, dtype=float).reshape(-1)
        if active_target.shape != current.shape:
            raise ValueError("EFD target/current readback shapes differ")
        target = active_target.copy()
        for storage, actuator_indices in self.outgoing.items():
            filling = fill[storage]
            raw = float(
                np.clip(
                    filling + self.equalization_gain * (filling - mean_fill),
                    0.0,
                    1.0,
                )
            )
            for i in actuator_indices:
                target[i] = active_target[i] + self.response * (raw - active_target[i])
        target = _limit_move(active_target, target, self.max_delta_per_update)
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, target, strict=True)),
            source=EFD_SOURCE,
            diagnostics={
                "rule_contract": EFD_CONTRACT,
                "controlled_storages": len(self.sensor_nodes),
                "mean_filling_degree": mean_fill,
                "filling_degree_std": float(values.std()),
                "volume_based_storage_count": len(self.sensor_nodes) - depth_fallback_count,
                "depth_fallback_storage_count": depth_fallback_count,
                "command_anchor": "actuator_target_setting",
                "current_tracking_lag_max": float(np.abs(current-active_target).max(initial=0.0)),
            },
        )
