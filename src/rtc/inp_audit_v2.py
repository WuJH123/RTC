from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .graph import infer_flow_units, infer_system_units
from .inp import discover_actuators, discover_nodes
from .inp_runtime import section_has_payload, sha256_file


def _sections(path: str | Path) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = defaultdict(list)
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        if line:
            result[section].append(line.split())
    return result


def audit_inp(path: str | Path, priority_nodes: tuple[str, ...] = ()) -> dict[str, object]:
    sections = _sections(path)
    nodes = tuple(discover_nodes(path))
    catalog = discover_actuators(path)
    options = {row[0].upper(): " ".join(row[1:]) for row in sections.get("OPTIONS", []) if len(row) >= 2}
    actuator_kind = {a.actuator_id: a.kind for a in catalog.actuators}
    controlled: set[str] = set()
    for row in sections.get("CONTROLS", []):
        if row and row[0].upper() in {"THEN", "ELSE"}:
            controlled.update(token for token in row if token in actuator_kind)
    curve_types: dict[str, str] = {}
    for row in sections.get("CURVES", []):
        if len(row) >= 4:
            try:
                float(row[1])
            except ValueError:
                curve_types.setdefault(row[0], row[1].upper())
    pump_curve_types = []
    for row in sections.get("PUMPS", []):
        if len(row) >= 4:
            pump_curve_types.append(curve_types.get(row[3], "UNKNOWN"))
    missing_priority = sorted(set(priority_nodes) - set(nodes))
    result: dict[str, object] = {
        "contract": "LARGE_SWMM_INP_PREFLIGHT_V2",
        "inp_path": str(Path(path).resolve()),
        "inp_sha256": sha256_file(path),
        "flow_units": infer_flow_units(path),
        "system_units": infer_system_units(path),
        "flow_routing": options.get("FLOW_ROUTING"),
        "report_step": options.get("REPORT_STEP"),
        "wet_step": options.get("WET_STEP"),
        "routing_step": options.get("ROUTING_STEP"),
        "rule_step": options.get("RULE_STEP"),
        "swmm_threads": options.get("THREADS"),
        "native_controls_present": section_has_payload(path, "CONTROLS"),
        "nodes": len(nodes),
        "junctions": len(sections.get("JUNCTIONS", [])),
        "outfalls": len(sections.get("OUTFALLS", [])),
        "storage_nodes": len(sections.get("STORAGE", [])),
        "conduits": len(sections.get("CONDUITS", [])),
        "subcatchments": len(sections.get("SUBCATCHMENTS", [])),
        "actuators": len(catalog.actuators),
        "actuator_types": dict(Counter(a.kind for a in catalog.actuators)),
        "native_controlled_actuators": len(controlled),
        "native_controlled_by_kind": dict(Counter(actuator_kind[x] for x in controlled)),
        "pump_curve_types": dict(Counter(pump_curve_types)),
        "priority_nodes_supplied": list(priority_nodes),
        "missing_priority_nodes": missing_priority,
        "priority_mapping_valid": bool(priority_nodes) and not missing_priority,
        "pfv_semantics": "cumulative flooding volume over a defined horizon/event; never instantaneous Node.flooding rate",
        "tfv_semantics": "sum of cumulative flooding volume over all nodes for a defined horizon/event",
        "instantaneous_flooding_semantics": "Node.flooding is a rate in FLOW_UNITS",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast audit of a large SWMM INP before data generation")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    priority: tuple[str, ...] = ()
    if args.priority:
        priority = tuple(
            line.strip() for line in Path(args.priority).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    payload = audit_inp(args.inp, priority)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if priority and payload["missing_priority_nodes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
