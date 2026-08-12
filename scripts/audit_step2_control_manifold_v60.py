"""Audit legacy Train-only D3 against the V6 MPC control manifold."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_train_response_v60 import V60TrainCache


def _load_graph(path):
    with np.load(path,allow_pickle=False) as r:
        return GraphSchema(node_ids=tuple(r["node_ids"].astype(str).tolist()),edge_index=r["edge_index"].astype(np.int64),static_node_features=r["static_node_features"].astype(np.float32),static_node_feature_names=tuple(r["static_node_feature_names"].astype(str).tolist()),actuator_ids=tuple(r["actuator_ids"].astype(str).tolist()),actuator_upstream=r["actuator_upstream"].astype(np.int64),actuator_downstream=r["actuator_downstream"].astype(np.int64),actuator_physics=r["actuator_physics"].astype(np.float32),actuator_physics_feature_names=tuple(r["actuator_physics_feature_names"].astype(str).tolist()),system_units=str(r["system_units"].item()))


def _rank(x):
    x=np.asarray(x,dtype=float); s=np.linalg.svd(x-x.mean(0,keepdims=True),compute_uv=False); e=s*s
    if float(e.sum())<=1e-15: return {"rank":0,"rank_90":0,"rank_95":0,"rank_99":0}
    c=np.cumsum(e)/e.sum(); return {"rank":int((s>1e-9).sum()),**{f"rank_{q}":int(np.searchsorted(c,q/100)+1) for q in (90,95,99)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--graph",required=True); p.add_argument("--cache-manifest",required=True); p.add_argument("--out",required=True); a=p.parse_args(); g=_load_graph(a.graph); basis=build_control_basis_v60(g); cache=V60TrainCache(a.cache_manifest); names=cache.names("D3")
    if not names: raise ValueError("cache contains no D3 groups")
    active_a=[]; active_b=[]; local=[]; residual=[]; all_x=[]; M=basis.design_matrix()
    for name in names:
        e=cache.entry(name); arr=e.arrays; cand=[i for i in e.indices if i!=e.reference_index]; ref=np.asarray(arr["settings"][e.reference_index],dtype=float); delta=np.asarray(arr["settings"][cand],dtype=float)-ref[None]; blocks=delta[:,::basis.horizon.control_block_steps,:]; x=blocks.reshape(len(cand),-1); all_x.append(x); local.append(_rank(x)["rank_95"]); active_a.extend((np.abs(blocks)>1e-6).any(1).sum(1).tolist()); active_b.extend((np.abs(blocks)>1e-6).any(2).sum(1).tolist()); coeff=basis.project_actions_to_coefficients(blocks); recon=(M@coeff.reshape(len(cand),-1).T).T; residual.extend((np.square(x-recon).sum(1)/np.maximum(np.square(x).sum(1),1e-12)).tolist())
    X=np.concatenate(all_x); payload={"contract":"PROJECT7_STEP2_V60_LEGACY_D3_MANIFOLD_AUDIT_V1","legacy_d3_groups":len(names),"legacy_d3_candidates":int(X.shape[0]),"raw_action_dimension":109*36,"v60_coefficient_dimension":basis.coefficient_dimension,"basis":basis_manifest_v60(basis),"global_action_rank":_rank(X),"local_rank95":{"median":float(np.median(local)),"max":int(np.max(local))},"active_actuators":{"median":float(np.median(active_a)),"p90":float(np.percentile(active_a,90)),"max":int(np.max(active_a))},"active_control_blocks":{"median":float(np.median(active_b)),"p90":float(np.percentile(active_b,90)),"max":int(np.max(active_b))},"projection_residual_energy_fraction":{"median":float(np.median(residual)),"p90":float(np.percentile(residual,90)),"p95":float(np.percentile(residual,95))},"interpretation":"Legacy dense-random D3 remains diagnostic-only; high residual means it does not represent the V6 MPC search manifold."}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__": main()
