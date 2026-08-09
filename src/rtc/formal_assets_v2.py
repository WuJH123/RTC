from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_priority_nodes, require_nodes_exist
from .graph import build_graph_schema, save_graph_schema
from .inp import discover_nodes
from .inp_lineage import physical_contract_sha256, write_physical_contract_manifest
from .pipeline import sha256_file


STATE_SCHEMA = {
    "contract": "RTC_STATE_SCHEMA_V2",
    "state_channels": [
        {"index": 0, "name": "depth", "unit": "m"},
        {"index": 1, "name": "head", "unit": "m"},
        {"index": 2, "name": "flooding_rate", "unit": "m3/s"},
        {"index": 3, "name": "node_volume", "unit": "m3"},
    ],
    "step1_observed_channels": ["depth", "head"],
    "step2_exogenous_channels": ["causal_node_rainfall_mm_per_h"],
    "forbidden_online_features": [
        "future_realized_SWMM_state",
        "future_realized_SWMM_runoff",
        "future_realized_flooding",
        "final_or_locked_truth",
    ],
}


def _lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def compile_assets_v2(
    *, inp_path: str | Path, priority_path: str | Path, sensor_path: str | Path, output_dir: str | Path
) -> dict[str, str]:
    inp = Path(inp_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = discover_nodes(inp)
    priority = load_priority_nodes(priority_path)
    sensors = _lines(sensor_path)
    require_nodes_exist(priority, nodes)
    require_nodes_exist(sensors, nodes)
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError(f"formal contract requires exactly 8 unique priority nodes, got {len(priority)}")
    if not sensors or len(set(sensors)) != len(sensors):
        raise ValueError("sensor layout must contain unique sensor node IDs")

    graph = build_graph_schema(inp)
    graph_path = save_graph_schema(graph, out / "graph_schema.npz")
    physical_path = write_physical_contract_manifest(inp, out / "physical_contract.json")
    state_path = out / "state_schema.json"
    state_path.write_text(json.dumps(STATE_SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    type_names = ("PUMP", "ORIFICE", "WEIR", "OUTLET")
    actuators: list[dict[str, object]] = []
    for i, actuator_id in enumerate(graph.actuator_ids):
        physics = graph.actuator_physics[i]
        one_hot = physics[:4]
        kind = type_names[int(one_hot.argmax())]
        up_idx = int(graph.actuator_upstream[i])
        down_idx = int(graph.actuator_downstream[i])
        actuators.append(
            {
                "actuator_id": actuator_id,
                "kind": kind,
                "upstream_node": graph.node_ids[up_idx],
                "downstream_node": graph.node_ids[down_idx],
                "min_setting": float(physics[4]),
                "max_setting": float(physics[5]),
                "continuous": True,
            }
        )
    actuator_path = out / "actuator_catalog.json"
    actuator_payload = {
        "contract": "ACTUATOR_AGNOSTIC_CONTINUOUS_CATALOG_V2",
        "source_inp": str(inp.resolve()),
        "source_inp_sha256": sha256_file(inp),
        "physical_network_sha256": physical_contract_sha256(inp),
        "fixed_active_subset": False,
        "binary_actuator_mask": False,
        "actuator_count": len(actuators),
        "actuators": actuators,
    }
    actuator_path.write_text(
        json.dumps(actuator_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path = out / "formal_asset_audit.json"
    audit = {
        "contract": "FORMAL_ASSET_AUDIT_V2",
        "passed": True,
        "frozen_inp": str(inp.resolve()),
        "frozen_inp_sha256": sha256_file(inp),
        "physical_network_sha256": physical_contract_sha256(inp),
        "node_count": len(nodes),
        "actuator_count": len(actuators),
        "priority_nodes": list(priority),
        "sensor_nodes": list(sensors),
        "graph_schema": str(graph_path),
        "actuator_catalog": str(actuator_path),
        "state_schema": str(state_path),
        "physical_contract": str(physical_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "graph_schema": str(graph_path),
        "actuator_catalog": str(actuator_path),
        "state_schema": str(state_path),
        "physical_contract": str(physical_path),
        "asset_audit": str(audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile robust frozen Formal RTC assets")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    payload = compile_assets_v2(
        inp_path=args.inp,
        priority_path=args.priority,
        sensor_path=args.sensors,
        output_dir=args.out_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
