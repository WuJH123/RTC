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


def _pump_local_depth_controls(sections: dict[str, list[list[str]]]) -> list[dict[str, object]]:
    """Return intrinsic SWMM pump startup/shutoff depth logic.

    These fields live in [PUMPS], not [CONTROLS]. They therefore remain active in the
    scientific No-control runtime, which intentionally means no *supervisory* RTC rather
    than disabling local equipment protection/level logic or changing physical parameters.
    """

    rows: list[dict[str, object]] = []
    for row in sections.get("PUMPS", []):
        # Name Node1 Node2 Curve [Status Startup Shutoff]
        if len(row) < 4:
            continue
        status = row[4].upper() if len(row) >= 5 else "ON"
        try:
            startup = float(row[5]) if len(row) >= 6 else 0.0
            shutoff = float(row[6]) if len(row) >= 7 else 0.0
        except ValueError as exc:
            raise ValueError(f"invalid [PUMPS] startup/shutoff row: {' '.join(row)}") from exc
        if abs(startup) > 1e-12 or abs(shutoff) > 1e-12:
            rows.append(
                {
                    "pump_id": row[0],
                    "inlet_node": row[1],
                    "initial_status": status,
                    "startup_depth_native": startup,
                    "shutoff_depth_native": shutoff,
                }
            )
    return rows


def audit_inp(path: str | Path, priority_nodes: tuple[str, ...] = ()) -> dict[str, object]:
    sections = _sections(path)
    nodes = tuple(discover_nodes(path))
    catalog = discover_actuators(path)
    options = {
        row[0].upper(): " ".join(row[1:])
        for row in sections.get("OPTIONS", [])
        if len(row) >= 2
    }
    actuator_kind = {a.actuator_id: a.kind for a in catalog.actuators}
    controlled: set[str] = set()
    executable_control_lines = 0
    for row in sections.get("CONTROLS", []):
        if row:
            executable_control_lines += 1
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
    local_pump_controls = _pump_local_depth_controls(sections)
    missing_priority = sorted(set(priority_nodes) - set(nodes))
    result: dict[str, object] = {
        "contract": "LARGE_SWMM_INP_PREFLIGHT_V3_CAUSAL_RTC",
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
        "native_control_executable_lines": executable_control_lines,
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
        "pump_intrinsic_startup_shutoff_controls": local_pump_controls,
        "pump_intrinsic_startup_shutoff_count": len(local_pump_controls),
        "no_control_contract": {
            "id": "NO_SUPERVISORY_RTC_V2",
            "remove": "all executable user-defined [CONTROLS] rules",
            "preserve": [
                "all hydraulic/network geometry",
                "rainfall/runoff forcing",
                "pump curves and initial status",
                "intrinsic [PUMPS] startup/shutoff depth logic",
                "orifice/weir physical properties and default state",
            ],
            "python_actuator_writes": False,
            "interpretation": "no supervisory RTC; intrinsic local equipment logic is retained",
        },
        "priority_nodes_supplied": list(priority_nodes),
        "missing_priority_nodes": missing_priority,
        "priority_mapping_valid": bool(priority_nodes) and not missing_priority,
        "pfv_semantics": "cumulative SWMM flooding volume at the verified priority-node set over a defined horizon/event",
        "tfv_semantics": "sum of cumulative SWMM flooding volume over every hydraulic node for a defined horizon/event",
        "instantaneous_flooding_semantics": "Node.flooding is a rate in FLOW_UNITS and is never itself PFV/TFV",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast audit of a large SWMM INP before fresh RTC data generation"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    priority: tuple[str, ...] = ()
    if args.priority:
        priority = tuple(
            line.strip()
            for line in Path(args.priority).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    payload = audit_inp(args.inp, priority)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if priority and payload["missing_priority_nodes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
