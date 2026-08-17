"""Design fresh Development-only D3-HOLD branches for Direct-TFV admission calibration.

This wrapper intentionally does not reuse the historical Step2 Train-only CLI contract. It uses the
same frozen actuator basis and targeted D3 candidate generator, but requires checkpoint rows to carry
``scientific_split=development`` and ``development_fold=admission_calibration`` so the resulting
SWMM evidence cannot be mistaken for Step2 model-fitting data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.graph import GraphSchema
from rtc.inp import discover_actuators
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_d3_design_v60 import (
    D3V60DesignContract,
    design_targeted_d3_v60,
    targeted_d3_design_summary_v60,
)
from rtc.step2_d3_lineage_v60 import stamp_d3_v60_lineage


DIRECT_TFV_ADMISSION_D3_DESIGN_CONTRACT = "PROJECT7_DIRECT_TFV_FRESH_ADMISSION_D3_DESIGN_V1"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--basis-out")
    parser.add_argument("--candidates-per-checkpoint", type=int, default=24)
    parser.add_argument("--seed", type=int, default=4217)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    catalog = discover_actuators(args.inp)
    if tuple(catalog.ids) != tuple(graph.actuator_ids):
        raise ValueError("frozen INP actuator order/identity differs from graph schema")
    checkpoints = pd.read_csv(args.checkpoints)
    if checkpoints.empty:
        raise ValueError("fresh admission checkpoint table is empty")
    required = {"scientific_split", "development_fold", "rainfall_group", "event_id"}
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        raise ValueError(f"fresh admission checkpoint table lacks columns: {missing}")
    if {str(value).strip().lower() for value in checkpoints.scientific_split} != {"development"}:
        raise ValueError("fresh Direct-TFV admission D3 design is Development-only")
    if {str(value).strip().lower() for value in checkpoints.development_fold} != {
        "admission_calibration"
    }:
        raise ValueError(
            "fresh Direct-TFV admission checkpoints must use development_fold=admission_calibration"
        )
    if checkpoints.rainfall_group.astype(str).nunique() < 9:
        raise ValueError("fresh admission D3 design requires at least nine rainfall groups")
    forbidden = ("validation", "final", "formal", "policy_lock", "policylock")
    identifiers = (
        checkpoints.rainfall_group.astype(str).tolist() + checkpoints.event_id.astype(str).tolist()
    )
    leaked = sorted(
        value for value in identifiers if any(token in value.lower() for token in forbidden)
    )
    if leaked:
        raise ValueError(f"fresh admission checkpoints contain untouched-evaluation identifiers: {leaked}")

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
    manifest["scientific_split"] = "development"
    manifest["development_fold"] = "admission_calibration"
    manifest["admission_data_contract"] = DIRECT_TFV_ADMISSION_D3_DESIGN_CONTRACT

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = targeted_d3_design_summary_v60(manifest)
    summary.update(lineage)
    summary.update(
        {
            "contract": DIRECT_TFV_ADMISSION_D3_DESIGN_CONTRACT,
            "development_only": True,
            "development_fold": "admission_calibration",
            "rainfall_group_count": int(checkpoints.rainfall_group.astype(str).nunique()),
            "event_count": int(checkpoints.event_id.astype(str).nunique()),
            "out": str(out),
        }
    )
    basis_payload = basis_manifest_v60(basis)
    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    basis_path = Path(args.basis_out) if args.basis_out else out.with_suffix(".basis.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    basis_path.write_text(json.dumps(basis_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
