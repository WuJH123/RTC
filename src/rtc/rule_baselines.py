from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .inp import ActuatorCatalog, discover_actuators
from .units import length_to_m

AUTO_RBC_SOURCE = "AUTO_RBC_V1"
AUTO_RBC_CONTRACT = "AUTO_RBC_TOPOLOGY_NORMALIZED_DEPTH_V1"
EFD_SOURCE = "EFD_V1"
EFD_CONTRACT = "STORAGE_EQUAL_FILLING_DEGREE_V1"


@dataclass(frozen=True)
class RuleNetwork:
    catalog: ActuatorCatalog
    node_max_depth_m: dict[str, float]
    storage_ids: tuple[str, ...]


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


def _system_units(path: str | Path) -> str:
    flow_units = ""
    for section, tokens in _iter_rows(path):
        if section == "OPTIONS" and len(tokens) >= 2 and tokens[0].upper() == "FLOW_UNITS":
            flow_units = tokens[1].upper()
            break
    if flow_units in {"CMS", "LPS", "MLD"}:
        return "SI"
    if flow_units in {"CFS", "GPM", "MGD"}:
        return "US"
    raise ValueError(f"cannot infer SWMM system units from FLOW_UNITS in {path}")


def load_rule_network(path: str | Path) -> RuleNetwork:
    units = _system_units(path)
    max_depth: dict[str, float] = {}
    storage: list[str] = []
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
        if section == "STORAGE":
            storage.append(node_id)
    return RuleNetwork(
        catalog=discover_actuators(path),
        node_max_depth_m=max_depth,
        storage_ids=tuple(dict.fromkeys(storage)),
    )


def rule_baseline_sensor_nodes(strategy: str, inp_path: str | Path) -> tuple[str, ...]:
    network = load_rule_network(inp_path)
    if strategy == "auto_rbc":
        nodes: list[str] = []
        for actuator in network.catalog.actuators:
            for node in (actuator.upstream_node, actuator.downstream_node):
                if node in network.node_max_depth_m and node not in nodes:
                    nodes.append(node)
        if not nodes:
            raise ValueError("Auto-RBC found no actuator-adjacent nodes with a valid design max depth")
        return tuple(nodes)
    if strategy == "efd":
        controlled = {
            actuator.upstream_node
            for actuator in network.catalog.actuators
            if actuator.upstream_node in set(network.storage_ids)
        }
        nodes = tuple(node for node in network.storage_ids if node in controlled)
        if not nodes:
            raise ValueError("EFD requires at least one storage node with a writable outgoing actuator")
        return nodes
    return ()


def _limit_move(current: np.ndarray, target: np.ndarray, max_delta: float | None) -> np.ndarray:
    target = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    if max_delta is None:
        return target
    delta = float(max_delta)
    if delta < 0:
        raise ValueError("max setting delta must be non-negative")
    return np.clip(target, current - delta, current + delta).clip(0.0, 1.0)


class AutoRBCController:
    """Automatically parameterized local rule-based control.

    The rule uses only current actuator-adjacent node depths normalized by the design max
    depths written in the INP. High upstream filling opens the link; severe downstream
    filling suppresses discharge. No future rainfall/state or event-specific tuning is used.
    """

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
            raise ValueError("Auto-RBC downstream congestion threshold must lie in (0,1)")
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
        sensor_depth = dict(zip(obs.sensor_ids, np.asarray(obs.sensor_depth_m, dtype=float), strict=True))
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        target = current.copy()
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
            target[i] = current[i] + self.response * (raw - current[i])
            upstream_fill.append(up)
            downstream_fill.append(down)
        target = _limit_move(current, target, self.max_delta_per_update)
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, target, strict=True)),
            source=AUTO_RBC_SOURCE,
            diagnostics={
                "rule_contract": AUTO_RBC_CONTRACT,
                "observed_rule_nodes": len(self.sensor_nodes),
                "mean_upstream_fill": float(np.mean(upstream_fill)) if upstream_fill else 0.0,
                "max_downstream_fill": float(np.max(downstream_fill)) if downstream_fill else 0.0,
            },
        )


class EFDController:
    """Storage-aware Equal Filling Degree baseline.

    Only storage nodes that have writable outgoing actuators participate in the equalization
    target. More-filled storages are commanded to discharge more strongly than less-filled
    storages, while the common filling level also controls the absolute discharge level.
    Non-storage actuators simply keep their current readback setting.
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
        depth = dict(zip(obs.sensor_ids, np.asarray(obs.sensor_depth_m, dtype=float), strict=True))
        fill = {
            node: float(
                np.clip(depth[node] / self.network.node_max_depth_m[node], 0.0, 1.5)
            )
            for node in self.sensor_nodes
        }
        values = np.asarray(list(fill.values()), dtype=float)
        mean_fill = float(values.mean())
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)
        target = current.copy()
        for storage, actuator_indices in self.outgoing.items():
            filling = fill[storage]
            raw = float(np.clip(filling + self.equalization_gain * (filling - mean_fill), 0.0, 1.0))
            for i in actuator_indices:
                target[i] = current[i] + self.response * (raw - current[i])
        target = _limit_move(current, target, self.max_delta_per_update)
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, target, strict=True)),
            source=EFD_SOURCE,
            diagnostics={
                "rule_contract": EFD_CONTRACT,
                "controlled_storages": len(self.sensor_nodes),
                "mean_filling_degree": mean_fill,
                "filling_degree_std": float(values.std()),
            },
        )
