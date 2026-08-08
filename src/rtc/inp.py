from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CONTROL_SECTIONS = {
    "PUMPS": "pump",
    "ORIFICES": "orifice",
    "WEIRS": "weir",
    "OUTLETS": "outlet",
}
NODE_SECTIONS = {"JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS"}


@dataclass(frozen=True)
class Actuator:
    actuator_id: str
    kind: str
    upstream_node: str
    downstream_node: str
    min_setting: float = 0.0
    max_setting: float = 1.0
    continuous: bool = True
    raw_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActuatorCatalog:
    actuators: tuple[Actuator, ...]

    def __post_init__(self) -> None:
        ids = [a.actuator_id for a in self.actuators]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate actuator IDs")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(a.actuator_id for a in self.actuators)

    def by_id(self, actuator_id: str) -> Actuator:
        for actuator in self.actuators:
            if actuator.actuator_id == actuator_id:
                return actuator
        raise KeyError(actuator_id)


def _iter_inp_rows(path: str | Path) -> Iterable[tuple[str, list[str]]]:
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


def discover_actuators(path: str | Path) -> ActuatorCatalog:
    """Discover every SWMM link whose SETTING can participate in online control.

    No manual Engineering-N list and no binary-pump mask are used. All discovered
    settings are represented continuously on [0, 1] unless future engineering
    metadata explicitly narrows the physical interval.
    """

    actuators: list[Actuator] = []
    for section, tokens in _iter_inp_rows(path):
        if section not in CONTROL_SECTIONS or len(tokens) < 3:
            continue
        actuators.append(
            Actuator(
                actuator_id=tokens[0],
                kind=CONTROL_SECTIONS[section],
                upstream_node=tokens[1],
                downstream_node=tokens[2],
                raw_tokens=tuple(tokens[3:]),
            )
        )
    if not actuators:
        raise ValueError(f"no controllable SWMM links found in {path}")
    return ActuatorCatalog(tuple(actuators))


def discover_nodes(path: str | Path) -> tuple[str, ...]:
    node_ids: list[str] = []
    for section, tokens in _iter_inp_rows(path):
        if section in NODE_SECTIONS:
            node_ids.append(tokens[0])
    if not node_ids:
        raise ValueError(f"no SWMM nodes found in {path}")
    return tuple(dict.fromkeys(node_ids))
