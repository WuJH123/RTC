from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .acceptance import mae, rank_correlation, rmse
from .contracts import load_priority_nodes
from .flood_volume import trapezoid_node_flood_volume
from .large_model_cli import _device
from .production_cli import _load_graph, _load_step2
from .step2_shards import load_shard_manifest, sha256_file


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accept_step2_large_v2_main() -> None:
    parser=argparse.ArgumentParser(description="Accept Step2 using exact cumulative SWMM flooding-volume truth")
    parser.add_argument("--manifest",required=True); parser.add_argument("--graph",required=True); parser.add_argument("--model",required=True)
    parser.add_argument("--priority"); parser.add_argument("--out",required=True); parser.add_argument("--batch-size",type=int,default=2); parser.add_argument("--device")
    args=parser.parse_args()
    device=_device(args.device); graph=_load_graph(args.graph); model=_load_step2(args.model,device)
    manifest=load_shard_manifest(args.manifest)
    up=torch.as_tensor(graph.actuator_upstream,dtype=torch.long,device=device); down=torch.as_tensor(graph.actuator_downstream,dtype=torch.long,device=device)
    static=torch.as_tensor(graph.static_node_features,dtype=torch.float32,device=device); edges=torch.as_tensor(graph.edge_index,dtype=torch.long,device=device)
    physics=torch.as_tensor(graph.actuator_physics,dtype=torch.float32,device=device)
    pred_depth=[]; true_depth=[]; pred_flow=[]; true_flow=[]; pred_tfv=[]; true_tfv=[]; pred_pfv=[]; true_pfv=[]
    pidx=None
    if args.priority:
        priority=load_priority_nodes(args.priority); missing=sorted(set(priority)-set(graph.node_ids))
        if missing: raise ValueError(f"priority mapping incompatible with graph: {missing}")
        pidx=np.asarray([graph.node_ids.index(n) for n in priority],dtype=int)
    with torch.no_grad():
        for item in manifest["shards"]:
            with np.load(str(item["path"]),allow_pickle=False) as ds:
                if "exact_node_flood_volume_m3" not in ds.files:
                    raise ValueError("Step2 Formal acceptance requires exact SWMM node flooding-volume truth in every shard")
                count=ds["initial_state"].shape[0]
                for start in range(0,count,args.batch_size):
                    end=min(count,start+args.batch_size); b=end-start
                    initial=torch.as_tensor(ds["initial_state"][start:end],dtype=torch.float32,device=device)
                    rain=torch.as_tensor(ds["rainfall"][start:end],dtype=torch.float32,device=device)
                    settings=torch.as_tensor(ds["settings"][start:end],dtype=torch.float32,device=device)
                    prev=torch.as_tensor(ds["previous_actuator_flow"][start:end],dtype=torch.float32,device=device)
                    rollout=model.rollout(initial,rain,settings,prev,up,down,physics.unsqueeze(0).expand(b,-1,-1),static,edges)
                    dt=torch.as_tensor(np.diff(ds["elapsed_seconds"][start:end].astype(np.float32),axis=1),dtype=torch.float32,device=device)
                    pred_node=trapezoid_node_flood_volume(initial,rollout.states,flood_rate_index=2,dt_seconds=dt).cpu().numpy()
                    exact=ds["exact_node_flood_volume_m3"][start:end].astype(float)
                    ps=rollout.states.cpu().numpy(); pf=rollout.actuator_flows.cpu().numpy(); ts=ds["target_states"][start:end]; tf=ds["target_actuator_flows"][start:end]
                    pred_depth.append(ps[...,0]); true_depth.append(ts[...,0]); pred_flow.append(pf); true_flow.append(tf)
                    pred_tfv.extend(pred_node.sum(axis=1)); true_tfv.extend(exact.sum(axis=1))
                    if pidx is not None:
                        pred_pfv.extend(pred_node[:,pidx].sum(axis=1)); true_pfv.extend(exact[:,pidx].sum(axis=1))
    metrics={
        "depth_rmse_m":rmse(np.concatenate(pred_depth),np.concatenate(true_depth)),
        "managed_flow_rmse_m3s":rmse(np.concatenate(pred_flow),np.concatenate(true_flow)),
        "tfv_exact_truth_mae_m3":mae(np.asarray(pred_tfv),np.asarray(true_tfv)),
        "tfv_exact_truth_rank_correlation":rank_correlation(np.asarray(pred_tfv),np.asarray(true_tfv)),
    }
    if pidx is not None:
        metrics["priority_flood_exact_truth_mae_m3"]=mae(np.asarray(pred_pfv),np.asarray(true_pfv))
        metrics["priority_flood_exact_truth_rank_correlation"]=rank_correlation(np.asarray(pred_pfv),np.asarray(true_pfv))
    payload={
        "contract":"STEP2_LARGE_EXACT_TRUTH_ACCEPTANCE_V3_TRAPEZOID",
        "model_sha256":_sha(args.model),"manifest_sha256":sha256_file(args.manifest),
        "metrics":metrics,"priority_diagnostic_only":True,
        "truth_source_tfv_pfv":"SWMM_NODE_STATISTICS_CUMULATIVE_EXACT_HORIZON",
        "prediction_volume_integration":"trapezoid_current_plus_future_flooding_rate",
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": accept_step2_large_v2_main()
