"""Jointly score untouched calibration query sets with the decomposed policy-return critic.

Rows are grouped by query_set_id before scoring.  This is essential: candidate ordering is relative to
other candidates in the same causal prefix, while the best-candidate margin is one query-level value.
The script only appends frozen predictions to existing authoritative calibration truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_policy_return import encode_policy_return_action_token, sha256_file
from rtc.direct_tfv_policy_return_portfolio_admission import validate_policy_return_learning_record
from rtc.direct_tfv_policy_return_query_margin import (
    DIRECT_TFV_QUERY_MARGIN_CONTRACT,
    build_query_margin_features,
    load_query_margin_checkpoint,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph


def _norm(normalization: Any, *, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.as_tensor(normalization.state_mean, dtype=dtype, device=device),
        "state_std": torch.as_tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1e-6),
        "rain_mean": torch.as_tensor(normalization.rainfall_mean, dtype=dtype, device=device),
        "rain_std": torch.as_tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1e-6),
        "flow_mean": torch.as_tensor(normalization.flow_mean, dtype=dtype, device=device),
        "flow_std": torch.as_tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1e-6),
    }


def _load_context(row: dict[str, Any], *, expected_mask_sha: str) -> dict[str, np.ndarray]:
    path = Path(str(row.get("context_npz", "")))
    if not path.is_file() or sha256_file(path).lower() != str(row.get("context_npz_sha256", "")).lower():
        raise ValueError(f"calibration context missing/SHA mismatch: {path}")
    data = np.load(path, allow_pickle=False)
    if str(np.asarray(data["data_role"]).reshape(-1)[0]) != "policy_return_calibration":
        raise ValueError("query-margin scorer accepts calibration contexts only")
    if str(np.asarray(data["supervisory_mask_sha256"]).reshape(-1)[0]).lower() != expected_mask_sha:
        raise ValueError("calibration context uses another supervisory mask")
    return {key: np.asarray(data[key]) for key in (
        "current_state","rainfall_scenarios","active_target","candidate_target",
        "previous_actuator_flow","base_step2_h10_score_m3"
    )}


def _rank_one(
    model: torch.nn.Module, normalization: Any, graph: Any, context: dict[str,np.ndarray], device: torch.device
) -> torch.Tensor:
    state_raw = torch.as_tensor(context["current_state"], dtype=torch.float32, device=device)
    rain_raw = torch.as_tensor(context["rainfall_scenarios"], dtype=torch.float32, device=device)
    active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device)
    target = torch.as_tensor(context["candidate_target"], dtype=torch.float32, device=device)
    flow_raw = torch.as_tensor(context["previous_actuator_flow"], dtype=torch.float32, device=device)
    if int(rain_raw.shape[0]) != 1 or rain_raw.ndim != 5:
        raise ValueError("calibration context must contain one candidate row with scenario rainfall")
    _, scenarios, horizon, nodes, features = rain_raw.shape
    norm = _norm(normalization, dtype=state_raw.dtype, device=device)
    state = ((state_raw - norm["state_mean"])/norm["state_std"]).expand(scenarios,-1,-1)
    rain = ((rain_raw - norm["rain_mean"])/norm["rain_std"]).reshape(scenarios,horizon,nodes,features)
    flow = ((flow_raw - norm["flow_mean"])/norm["flow_std"]).expand(scenarios,-1)
    active = active.expand(scenarios,-1); target = target.expand(scenarios,-1)
    reference, candidate = encode_policy_return_action_token(
        active,target,horizon_steps=int(horizon),first_action_steps=2
    )
    output = model(
        current_state=state,rainfall=rain,reference_settings=reference,candidate_settings=candidate,
        previous_actuator_flow=flow,
        actuator_upstream=torch.as_tensor(graph.actuator_upstream,dtype=torch.long,device=device),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream,dtype=torch.long,device=device),
        actuator_physics=torch.as_tensor(graph.actuator_physics,dtype=state.dtype,device=device),
    )
    return output.total_delta_tfv_m3.mean()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records-jsonl",required=True)
    p.add_argument("--query-margin-checkpoint",required=True)
    p.add_argument("--base-step2",required=True)
    p.add_argument("--graph",required=True)
    p.add_argument("--supervisory-control",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    device=torch.device(args.device if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    graph=_load_graph(args.graph)
    rank_model, normalization, adapter, checkpoint=load_query_margin_checkpoint(
        args.query_margin_checkpoint,graph=graph,base_step2_path=args.base_step2,device=device
    )
    if checkpoint.get("fresh_validation_verified") is not True:
        raise ValueError("calibration cannot score a critic that failed fresh validation")
    _, mask=load_native_supervisory_control(args.supervisory_control,actuator_ids=graph.actuator_ids)
    mask_sha=str(checkpoint["supervisory_mask_sha256"]).lower()
    rows=[]
    for line_number,raw in enumerate(Path(args.records_jsonl).read_text(encoding="utf-8").splitlines(),1):
        if not raw.strip(): continue
        row=json.loads(raw)
        if not isinstance(row,dict): raise ValueError(f"row {line_number} is not an object")
        validate_policy_return_learning_record(row)
        if str(row.get("data_role"))!="policy_return_calibration":
            raise ValueError("query-margin calibration scorer received another role")
        if str(row.get("supervisory_mask_sha256","")).lower()!=mask_sha:
            raise ValueError("calibration truth uses another supervisory mask")
        rows.append(row)
    by_query: dict[str,list[int]]={}
    for i,row in enumerate(rows): by_query.setdefault(str(row["query_set_id"]),[]).append(i)
    checkpoint_sha=sha256_file(args.query_margin_checkpoint)
    with torch.no_grad():
        for query, positions in sorted(by_query.items()):
            if not 1 <= len(positions) <= 3:
                raise ValueError("current query-margin scorer requires 1--3 candidates per query")
            contexts=[_load_context(rows[i],expected_mask_sha=mask_sha) for i in positions]
            first=contexts[0]
            active0=np.asarray(first["active_target"]).reshape(1,109)[0]
            for ctx in contexts[1:]:
                if not np.allclose(np.asarray(ctx["active_target"]).reshape(1,109)[0],active0,atol=1e-7):
                    raise ValueError("same query_set_id contains different active targets")
            raw_scores=torch.stack([_rank_one(rank_model,normalization,graph,ctx,device) for ctx in contexts])
            candidate_targets=torch.cat([
                torch.as_tensor(ctx["candidate_target"],dtype=torch.float32,device=device) for ctx in contexts
            ],dim=0)
            base_scores=torch.as_tensor(
                [float(np.asarray(ctx["base_step2_h10_score_m3"]).reshape(-1)[0]) for ctx in contexts],
                dtype=torch.float32,device=device
            )
            context_features,candidate_features=build_query_margin_features(
                current_state=torch.as_tensor(first["current_state"],dtype=torch.float32,device=device)[0],
                rainfall_scenarios=torch.as_tensor(first["rainfall_scenarios"],dtype=torch.float32,device=device)[0],
                previous_actuator_flow=torch.as_tensor(first["previous_actuator_flow"],dtype=torch.float32,device=device)[0],
                active_target=torch.as_tensor(active0,dtype=torch.float32,device=device),
                candidate_targets=candidate_targets,
                base_step2_scores_m3=base_scores,
                candidate_sources=[str(rows[i]["candidate_source"]) for i in positions],
                supervisory_mask=mask,
                target_scale_m3=float(rank_model.target_scale_m3.detach().cpu()),
            )
            output=adapter(
                raw_rank_scores_m3=raw_scores,context_features=context_features,candidate_features=candidate_features
            )
            predictions=output.predicted_returns_m3.detach().cpu().numpy().astype(float)
            rank=output.relative_rank_normalized.detach().cpu().numpy().astype(float)
            margin=float(output.query_best_margin_m3.detach().cpu())
            for local,pos in enumerate(positions):
                scored=dict(rows[pos])
                scored["predicted_policy_return_delta_tfv_m3"]=float(predictions[local])
                scored["query_conditioned_best_candidate_margin_m3"]=margin
                scored["query_conditioned_relative_rank_normalized"]=float(rank[local])
                scored["query_margin_contract"]=DIRECT_TFV_QUERY_MARGIN_CONTRACT
                scored["policy_return_checkpoint_sha256"]=checkpoint_sha
                validate_policy_return_learning_record(scored)
                rows[pos]=scored
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
    summary={
        "record_count":len(rows),"query_set_count":len(by_query),
        "rainfall_group_count":len({str(r["rainfall_group"]) for r in rows}),
        "query_margin_contract":DIRECT_TFV_QUERY_MARGIN_CONTRACT,
        "policy_return_checkpoint_sha256":checkpoint_sha,
        "supervisory_mask_sha256":mask_sha,
        "fresh_validation_verified":True,
        "calibration_used_for_training":False,
        "scored_records_sha256":sha256_file(out),
    }
    out.with_suffix(".json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
