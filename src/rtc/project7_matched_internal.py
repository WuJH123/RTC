"""Reconstructed-state interpreter for the matched Project7 native rules.

The frozen Wuhan ``[CONTROLS]`` section is deliberately interpreted from the causal Step1
state, not from simulator-only node observations.  The parser is intentionally narrow: accepting
only the condition and action grammar that can be reproduced from the current Project7 causal
state contract is safer than silently approximating an unsupported rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from .swmm_data import STATE_CHANNELS
from .units import length_to_m

MATCHED_INTERNAL_RULE_CONTRACT = (
    "PROJECT7_NATIVE_CONTROLS_RECONSTRUCTED_STEP1_HEAD_MATCHED_V1"
)
_SUPPORTED_FLOW_UNITS = {
    "CMS": "SI",
    "LPS": "SI",
    "MLD": "SI",
    "CFS": "US",
    "GPM": "US",
    "MGD": "US",
}
_ACTION_OBJECTS = {"PUMP", "ORIFICE", "WEIR", "OUTLET"}
_COMPARATORS = {"=", "<", "<=", ">", ">="}


@dataclass(frozen=True)
class ReconstructedControlCondition:
    node_id: str
    node_index: int
    comparator: str
    threshold_m: float


@dataclass(frozen=True)
class ReconstructedControlRule:
    name: str
    source_line: int
    conditions: tuple[ReconstructedControlCondition, ...]
    actuator_id: str
    actuator_index: int
    actuator_kind: str
    command_kind: str
    command_value: float
    priority: float | None


def _fold_map(values: Any, *, label: str) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for index, value in enumerate(values):
        text = str(value)
        key = text.casefold()
        if key in result:
            raise ValueError(f"graph {label} IDs are not unique case-insensitively")
        result[key] = (index, text)
    return result


def _graph_actuator_kinds(graph: Any) -> tuple[str, ...]:
    ids = tuple(str(value) for value in getattr(graph, "actuator_ids", ()))
    names = tuple(str(value) for value in getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(getattr(graph, "actuator_physics", ()), dtype=np.float64)
    columns = {
        kind: names.index(f"is_{kind}")
        for kind in ("pump", "orifice", "weir", "outlet")
        if f"is_{kind}" in names
    }
    if physics.ndim != 2 or physics.shape[0] != len(ids) or len(columns) != 4:
        raise ValueError("matched Internal requires complete actuator physics/type indicators")
    result: list[str] = []
    for row in range(len(ids)):
        kinds = [kind for kind, column in columns.items() if physics[row, column] > 0.5]
        if len(kinds) != 1:
            raise ValueError(f"actuator {ids[row]!r} has ambiguous type")
        result.append(kinds[0])
    return tuple(result)


def _system_units(path: str | Path) -> str:
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].upper() == "FLOW_UNITS":
            flow_units = tokens[1].upper()
            try:
                return _SUPPORTED_FLOW_UNITS[flow_units]
            except KeyError as exc:
                raise ValueError(f"unsupported SWMM FLOW_UNITS: {flow_units}") from exc
    raise ValueError("[OPTIONS] FLOW_UNITS is required for native control thresholds")


def _control_lines(path: str | Path):
    section = ""
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        if section == "CONTROLS":
            yield line_number, line


def _parse_action_tokens(
    line: str,
    *,
    actuator_map: dict[str, tuple[int, str]],
    actuator_kinds: tuple[str, ...],
) -> tuple[int, str, str, float]:
    tokens = line.split()
    if len(tokens) != 6 or tokens[0].upper() != "THEN":
        raise ValueError(f"unsupported native control action grammar: {line}")
    object_kind, raw_id, command_kind, equals, raw_value = (
        tokens[1].upper(),
        tokens[2],
        tokens[3].upper(),
        tokens[4],
        tokens[5],
    )
    if object_kind not in _ACTION_OBJECTS or equals != "=":
        raise ValueError(f"unsupported native control action grammar: {line}")
    mapped = actuator_map.get(raw_id.casefold())
    if mapped is None:
        raise ValueError(f"native control action is absent from graph catalog: {raw_id}")
    index, actuator_id = mapped
    if actuator_kinds[index] != object_kind.casefold():
        raise ValueError(
            f"native control action kind mismatch for {raw_id}: "
            f"{object_kind.casefold()} != {actuator_kinds[index]}"
        )
    if command_kind == "STATUS" and object_kind != "PUMP":
        raise ValueError(f"STATUS action is only supported for PUMP: {line}")
    if command_kind not in {"STATUS", "SETTING"}:
        raise ValueError(f"unsupported native control action attribute: {line}")
    if command_kind == "STATUS":
        values = {"ON": 1.0, "OFF": 0.0}
        if raw_value.upper() not in values:
            raise ValueError(f"unsupported pump STATUS value: {line}")
        value = values[raw_value.upper()]
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"non-numeric SETTING action: {line}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"native control setting is outside [0,1]: {line}")
    return index, actuator_id, command_kind, value


def load_reconstructed_native_controls(
    path: str | Path, graph: Any
) -> tuple[ReconstructedControlRule, ...]:
    """Parse the source native rules into the exact Step1-reconstructable subset."""
    node_map = _fold_map(getattr(graph, "node_ids", ()), label="node")
    actuator_ids = tuple(str(value) for value in getattr(graph, "actuator_ids", ()))
    actuator_map = _fold_map(actuator_ids, label="actuator")
    actuator_kinds = _graph_actuator_kinds(graph)
    system_units = _system_units(path)
    rules: list[ReconstructedControlRule] = []
    current_name: str | None = None
    current_line = 0
    conditions: list[ReconstructedControlCondition] = []
    action: tuple[int, str, str, float] | None = None
    priority: float | None = None

    def finish() -> None:
        nonlocal current_name, current_line, conditions, action, priority
        if current_name is None:
            return
        if not conditions or action is None:
            raise ValueError(f"native control rule {current_name!r} is incomplete")
        index, actuator_id, command_kind, value = action
        rules.append(
            ReconstructedControlRule(
                name=current_name,
                source_line=current_line,
                conditions=tuple(conditions),
                actuator_id=actuator_id,
                actuator_index=index,
                actuator_kind=actuator_kinds[index],
                command_kind=command_kind,
                command_value=value,
                priority=priority,
            )
        )
        current_name = None
        current_line = 0
        conditions = []
        action = None
        priority = None

    for line_number, line in _control_lines(path):
        upper = line.upper()
        if upper.startswith("RULE "):
            finish()
            current_name = line.split(None, 1)[1].strip()
            if not current_name:
                raise ValueError(f"empty native control rule at line {line_number}")
            current_line = line_number
            continue
        if current_name is None:
            raise ValueError(f"native control clause precedes RULE at line {line_number}")
        if upper.startswith(("IF ", "AND ")):
            if upper.startswith("AND ") and action is not None:
                raise ValueError("multiple native actions are not supported")
            match = re.fullmatch(
                r"(?:IF|AND)\s+NODE\s+(\S+)\s+HEAD\s*(=|<=|>=|<|>)\s*(\S+)",
                line,
                flags=re.IGNORECASE,
            )
            if match is None:
                raise ValueError(
                    "native Internal rule condition is not reproducible from Step1 NODE HEAD: "
                    f"{line}"
                )
            raw_node, comparator, raw_threshold = match.groups()
            mapped = node_map.get(raw_node.casefold())
            if mapped is None:
                raise ValueError(f"native control condition node is absent from graph: {raw_node}")
            try:
                threshold = float(raw_threshold)
            except ValueError as exc:
                raise ValueError(f"non-numeric NODE HEAD threshold: {line}") from exc
            if comparator not in _COMPARATORS or not math.isfinite(threshold):
                raise ValueError(f"invalid NODE HEAD condition: {line}")
            conditions.append(
                ReconstructedControlCondition(
                    node_id=mapped[1],
                    node_index=mapped[0],
                    comparator=comparator,
                    threshold_m=float(length_to_m(threshold, system_units)),
                )
            )
            continue
        if upper.startswith("THEN "):
            if action is not None:
                raise ValueError("multiple native actions are not supported")
            action = _parse_action_tokens(
                line,
                actuator_map=actuator_map,
                actuator_kinds=actuator_kinds,
            )
            continue
        if upper.startswith("ELSE ") or upper.startswith("OR "):
            raise ValueError(
                "native Internal rules contain ELSE/OR semantics that are outside the exact "
                f"matched interpreter: {line}"
            )
        if upper.startswith("PRIORITY "):
            tokens = line.split()
            if len(tokens) != 2:
                raise ValueError(f"invalid native control priority: {line}")
            try:
                priority = float(tokens[1])
            except ValueError as exc:
                raise ValueError(f"invalid native control priority: {line}") from exc
            if not math.isfinite(priority):
                raise ValueError(f"invalid native control priority: {line}")
            continue
        raise ValueError(f"unsupported native control clause: {line}")
    finish()
    if not rules:
        raise ValueError("native [CONTROLS] contains no reconstructable rules")
    return tuple(rules)


def _rule_priority(rule: ReconstructedControlRule) -> tuple[int, float, int]:
    # SWMM rules with an explicit priority outrank rules without one. For equal priority, the
    # rule appearing first in the input has precedence.
    return (
        int(rule.priority is not None),
        float(rule.priority) if rule.priority is not None else 0.0,
        -int(rule.source_line),
    )


def _condition_holds(value: torch.Tensor, comparator: str, threshold: float) -> bool:
    scalar = float(value.detach().cpu())
    if comparator == "=":
        return scalar == threshold
    if comparator == "<":
        return scalar < threshold
    if comparator == "<=":
        return scalar <= threshold
    if comparator == ">":
        return scalar > threshold
    if comparator == ">=":
        return scalar >= threshold
    raise ValueError(f"unsupported native condition comparator: {comparator}")


def evaluate_reconstructed_native_controls(
    *,
    rules: tuple[ReconstructedControlRule, ...],
    graph: Any,
    current_state: torch.Tensor,
    active_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate native rules using only the causal Step1 state and latched target."""
    state = (
        current_state[0]
        if current_state.ndim == 3 and int(current_state.shape[0]) == 1
        else current_state
    )
    if state.ndim != 2 or tuple(state.shape)[1] != len(STATE_CHANNELS):
        raise ValueError(
            "matched Internal requires the authoritative Step1 state contract "
            f"[node,{len(STATE_CHANNELS)}] with channels {STATE_CHANNELS}"
        )
    node_count = len(tuple(getattr(graph, "node_ids", ())))
    if int(state.shape[0]) != node_count:
        raise ValueError("matched Internal Step1 state/node ordering differs from graph")
    target = active_target.reshape(-1).clone()
    actuator_count = len(tuple(getattr(graph, "actuator_ids", ())))
    if tuple(target.shape) != (actuator_count,) or not torch.isfinite(target).all():
        raise ValueError("matched Internal active target shape or finiteness is invalid")
    if torch.any((target < -1.0e-7) | (target > 1.0 + 1.0e-7)):
        raise ValueError("matched Internal active target is outside [0,1]")

    triggered: list[ReconstructedControlRule] = []
    by_actuator: dict[int, list[ReconstructedControlRule]] = {}
    head = state[:, STATE_CHANNELS.index("head_m")]
    for rule in rules:
        if all(
            _condition_holds(
                head[condition.node_index], condition.comparator, condition.threshold_m
            )
            for condition in rule.conditions
        ):
            triggered.append(rule)
            by_actuator.setdefault(rule.actuator_index, []).append(rule)
    selected: list[ReconstructedControlRule] = []
    for index, candidates in by_actuator.items():
        winner = max(candidates, key=_rule_priority)
        target[index] = float(winner.command_value)
        selected.append(winner)
    selected.sort(key=lambda rule: rule.source_line)
    diagnostics: dict[str, Any] = {
        "contract": MATCHED_INTERNAL_RULE_CONTRACT,
        "state_channel": "head_m",
        "state_channel_index": STATE_CHANNELS.index("head_m"),
        "matched_rule_count": len(triggered),
        "selected_rule_count": len(selected),
        "triggered_rule_ids": [rule.name for rule in triggered],
        "selected_rule_ids": [rule.name for rule in selected],
        "selected_actuator_ids": [rule.actuator_id for rule in selected],
        "priority_resolution": "explicit_priority_then_first_source_rule",
        "uses_simulator_node_truth": False,
    }
    return target, diagnostics


__all__ = [
    "MATCHED_INTERNAL_RULE_CONTRACT",
    "ReconstructedControlCondition",
    "ReconstructedControlRule",
    "evaluate_reconstructed_native_controls",
    "load_reconstructed_native_controls",
]
