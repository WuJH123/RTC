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


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()), edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32), static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()), actuator_upstream=raw["actuator_upstream"].astype(np.int64), actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32), actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()), system_units=str(raw["system_units"].item()),
        )


def main() -> None:
    p=argparse.ArgumentParser(description="Design Project7 Step2 V6 targeted D3-v2 candidate-manifold branches")
    p.add_argument("--inp",required=True); p.add_argument("--graph",required=True); p.add_argument("--checkpoints",required=True); p.add_argument("--out",required=True)
    p.add_argument("--summary-out"); p.add_argument("--basis-out"); p.add_argument("--candidates-per-checkpoint",type=int,default=24); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); graph=_load_graph(a.graph); catalog=discover_actuators(a.inp)
    if tuple(catalog.ids) != tuple(graph.actuator_ids): raise ValueError("frozen INP actuator order/identity differs from graph schema")
    checkpoints=pd.read_csv(a.checkpoints)
    if checkpoints.empty: raise ValueError("checkpoint table is empty")
    if "scientific_split" in checkpoints and {str(x).strip().lower() for x in checkpoints.scientific_split} != {"development"}: raise ValueError("V6 D3-v2 design is development-only")
    if "development_fold" in checkpoints and {str(x).strip().lower() for x in checkpoints.development_fold} != {"train"}: raise ValueError("V6 D3-v2 design is Train-only")
    basis=build_control_basis_v60(graph); contract=D3V60DesignContract(candidates_per_checkpoint=a.candidates_per_checkpoint,seed=a.seed); manifest=design_targeted_d3_v60(checkpoints,graph,basis,contract=contract)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); manifest.to_csv(out,index=False); summary=targeted_d3_design_summary_v60(manifest); summary["out"]=str(out); summary["basis"]=basis_manifest_v60(basis)
    summary_path=Path(a.summary_out) if a.summary_out else out.with_suffix(".summary.json"); basis_path=Path(a.basis_out) if a.basis_out else out.with_suffix(".basis.json")
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); basis_path.write_text(json.dumps(basis_manifest_v60(basis),indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
