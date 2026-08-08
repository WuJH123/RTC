from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .contracts import load_priority_nodes, require_nodes_exist
from .data_design import design_independent_actuator_probes, summarise_probe_design
from .inp import discover_actuators, discover_nodes


def audit_inp_main() -> None:
    parser = argparse.ArgumentParser(description="Audit Wuhan SWMM actuator/state contract")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    catalog = discover_actuators(args.inp)
    nodes = discover_nodes(args.inp)
    priority = load_priority_nodes(args.priority)
    require_nodes_exist(priority, nodes)
    result = {
        "inp": str(Path(args.inp).resolve()),
        "node_count": len(nodes),
        "actuator_count": len(catalog.actuators),
        "actuator_types": dict(Counter(a.kind for a in catalog.actuators)),
        "continuous_actuators": sum(a.continuous for a in catalog.actuators),
        "hard_binary_actuators": 0,
        "priority_nodes": list(priority),
        "priority_nodes_present": True,
        "fixed_active_subset": None,
        "actuator_ids": list(catalog.ids),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)


def design_probes_main() -> None:
    parser = argparse.ArgumentParser(
        description="Design same-checkpoint, single-actuator D2 counterfactual probes"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--checkpoints", required=True, help="CSV with checkpoint_id and setting:<id>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--no-center", action="store_true")
    args = parser.parse_args()

    catalog = discover_actuators(args.inp)
    checkpoints = pd.read_csv(args.checkpoints)
    manifest = design_independent_actuator_probes(
        checkpoints,
        catalog,
        epsilon=args.epsilon,
        include_center=not args.no_center,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = summarise_probe_design(manifest)
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
