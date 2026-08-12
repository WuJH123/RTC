"""Design Train-only targeted D3-v2 sequences for Step2 V6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.graph import GraphSchema
from rtc.inp import discover_actuators
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_d3_design_v60 import D3V60DesignContract, design_targeted_d3_v60, targeted_d3_design_summary_v60
from rtc.step2_d3_lineage_v60 import stamp_d3_v60_lineage


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Design Project7 Step2 V6 targeted D3-v2 candidate-manifold branches"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--basis-out")
    parser.add_argument("--candidates-per-checkpoint", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    catalog = discover_actuators(args.inp)
    if tuple(catalog.ids) != tuple(graph.actuator_ids):
        raise ValueError("frozen INP actuator order/identity differs from graph schema")
    checkpoints = pd.read_csv(args.checkpoints)
    if checkpoints.empty:
        raise ValueError("checkpoint table is empty")
    if "scientific_split" in checkpoints and {
        str(value).strip().lower() for value in checkpoints.scientific_split
    } != {"development"}:
        raise ValueError("V6 D3-v2 design is development-only")
    if "development_fold" in checkpoints and {
        str(value).strip().lower() for value in checkpoints.development_fold
    } != {"train"}:
        raise ValueError("V6 D3-v2 design is Train-only")

    basis = build_control_basis_v60(graph)
    contract = D3V60DesignContract(
        candidates_per_checkpoint=args.candidates_per_checkpoint,
        seed=args.seed,
    )
    manifest = design_targeted_d3_v60(checkpoints, graph, basis, contract=contract)
    manifest, lineage = stamp_d3_v60_lineage(
        manifest,
        basis=basis,
        design_contract=contract,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = targeted_d3_design_summary_v60(manifest)
    summary.update(lineage)
    summary["out"] = str(out)
    basis_payload = basis_manifest_v60(basis)
    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    basis_path = Path(args.basis_out) if args.basis_out else out.with_suffix(".basis.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    basis_path.write_text(
        json.dumps(basis_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
