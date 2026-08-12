"""Targeted D3-v2 design on the low-dimensional V6 control manifold."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data_design import canonical_sequence_sha
from .step2_control_basis_v60 import ControlBasisV60, basis_manifest_v60
from .step2_v60_contract import V60_D3_DATA_CONTRACT

D3_V60_HOLD_ROLE = "D3_HOLD_REFERENCE"
D3_V60_CANDIDATE_ROLE = "D3_V60_MANIFOLD_CANDIDATE"
D3_V60_ACTIVE_ROLE = "D3_V60_ACTIVE_LEARNING_CANDIDATE"
D3_TIME_CONTRACT = "D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1"
D3_FEASIBILITY_CONTRACT = "D3_SEQUENTIAL_SETTING_RATE_FEASIBILITY_V1"


@dataclass(frozen=True)
class D3V60DesignContract:
    candidates_per_checkpoint: int = 24
    single_group_fraction: float = 0.40
    same_zone_fraction: float = 0.25
    cross_zone_fraction: float = 0.15
    sparse_low_discrepancy_fraction: float = 0.20
    coefficient_magnitudes: tuple[float, ...] = (0.35, 0.65, 1.0)
    max_groups_per_candidate: int = 4
    seed: int = 42

    def validate(self) -> None:
        if self.candidates_per_checkpoint < 8:
            raise ValueError("V6 D3 requires at least 8 candidates/checkpoint")
        fractions=(self.single_group_fraction,self.same_zone_fraction,self.cross_zone_fraction,self.sparse_low_discrepancy_fraction)
        if any(x < 0 for x in fractions) or not np.isclose(sum(fractions),1.0,atol=1e-8):
            raise ValueError("D3 family fractions must sum to one")
        if not self.coefficient_magnitudes or any(not 0 < x <= 1 for x in self.coefficient_magnitudes):
            raise ValueError("coefficient magnitudes must lie in (0,1]")
        if not 1 <= self.max_groups_per_candidate <= 8:
            raise ValueError("invalid max_groups_per_candidate")


def _counts(contract: D3V60DesignContract) -> dict[str,int]:
    contract.validate()
    f=np.asarray([contract.single_group_fraction,contract.same_zone_fraction,contract.cross_zone_fraction,contract.sparse_low_discrepancy_fraction])
    raw=f*contract.candidates_per_checkpoint; base=np.floor(raw).astype(int); remainder=contract.candidates_per_checkpoint-int(base.sum())
    for i in np.argsort(-(raw-base))[:remainder]: base[int(i)]+=1
    return dict(zip(("single_group","same_zone","cross_zone","sparse_ld"),base.astype(int).tolist(),strict=True))


def _checkpoint_hash(row: pd.Series) -> int:
    text="::".join(str(row.get(k,"")) for k in ("rainfall_group","event_id","checkpoint_id","checkpoint_minutes"))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)


def _groups_by_zone(basis: ControlBasisV60) -> dict[int,list[int]]:
    result: dict[int,set[int]]={}
    for i,g in enumerate(basis.grouping.group_id_by_actuator): result.setdefault(int(basis.grouping.zone_id_by_actuator[i]),set()).add(int(g))
    return {z:sorted(v) for z,v in result.items()}


def _coefficient_specs(basis: ControlBasisV60, *, checkpoint_seed: int, contract: D3V60DesignContract) -> list[tuple[str,np.ndarray]]:
    rng=np.random.default_rng(int(checkpoint_seed)^int(contract.seed)); counts=_counts(contract); k,g=basis.temporal_basis_count,basis.group_count; zones=_groups_by_zone(basis); zone_ids=sorted(zones); mags=tuple(float(x) for x in contract.coefficient_magnitudes); specs=[]; cursor=checkpoint_seed%max(g*k,1)
    def add(family,entries):
        c=np.zeros((k,g),np.float32)
        for t,group,value in entries: c[int(t)%k,int(group)%g]=float(np.clip(value,-1,1))
        if not np.any(np.abs(c)>1e-8): raise RuntimeError("empty V6 coefficient pattern")
        specs.append((family,c))
    for i in range(counts["single_group"]):
        linear=cursor+i; group=linear%g; temporal=(linear//g+i)%k; mag=mags[i%len(mags)]; add("single_group",[(temporal,group,mag if (checkpoint_seed+i)%2==0 else -mag)])
    same=[z for z in zone_ids if len(zones[z])>=2]
    for i in range(counts["same_zone"]):
        if not same: break
        z=same[(checkpoint_seed+i)%len(same)]; members=zones[z]; start=(checkpoint_seed+3*i)%len(members); g1,g2=members[start],members[(start+1)%len(members)]; mag=mags[(i+1)%len(mags)]; sign=1 if i%2==0 else -1; add("same_zone",[((i+checkpoint_seed)%k,g1,sign*mag),((i+checkpoint_seed)%k,g2,sign*mag)])
    for i in range(counts["cross_zone"]):
        if len(zone_ids)<2: break
        z1=zone_ids[(checkpoint_seed+i)%len(zone_ids)]; z2=zone_ids[(checkpoint_seed+i+max(1,len(zone_ids)//2))%len(zone_ids)]
        if z1==z2: z2=zone_ids[(zone_ids.index(z1)+1)%len(zone_ids)]
        g1=zones[z1][(checkpoint_seed+i)%len(zones[z1])]; g2=zones[z2][(checkpoint_seed+2*i)%len(zones[z2])]; mag=mags[(i+2)%len(mags)]; t=(2*i+checkpoint_seed)%k; add("cross_zone",[(t,g1,mag),(t,g2,-mag)])
    for i in range(counts["sparse_ld"]):
        active=min(contract.max_groups_per_candidate,2+(i%max(contract.max_groups_per_candidate-1,1))); groups=rng.choice(g,size=min(active,g),replace=False); temporal=rng.choice(k,size=min(2,k),replace=False); entries=[]
        for pos,group in enumerate(groups):
            phase=((i+1)*(pos+1)*0.6180339887498949+(checkpoint_seed%997)/997.0)%1.0; mag=mags[(i+pos)%len(mags)]*(0.5+0.5*phase); entries.append((int(temporal[pos%len(temporal)]),int(group),mag if phase>=0.5 else -mag))
        add("sparse_low_discrepancy",entries)
    while len(specs)<contract.candidates_per_checkpoint:
        i=len(specs); add("single_group_fallback",[((checkpoint_seed+5*i)%k,(cursor+7*i)%g,0.35 if i%2==0 else -0.35)])
    return specs[:contract.candidates_per_checkpoint]


def _decode_numpy(basis: ControlBasisV60, base: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    basis.validate(); group_delta=basis.temporal_basis@np.clip(coeff,-1,1); actuator_delta=group_delta[:,basis.grouping.group_id_by_actuator]; max_delta=float(basis.contract.max_setting_delta_per_update); raw=np.clip(base[None]+actuator_delta*max_delta,basis.min_setting[None],basis.max_setting[None]); projected=np.empty_like(raw,dtype=np.float64); previous=base.astype(np.float64,copy=True)
    for block in range(raw.shape[0]): projected[block]=np.clip(previous+np.clip(raw[block]-previous,-max_delta,max_delta),basis.min_setting,basis.max_setting); previous=projected[block]
    return projected.astype(np.float32)


def _sequence(ids: Sequence[str], values: np.ndarray) -> list[dict[str,float]]:
    return [{str(aid):float(step[i]) for i,aid in enumerate(ids)} for step in values]


def design_targeted_d3_v60(checkpoints: pd.DataFrame, graph: Any, basis: ControlBasisV60, *, contract: D3V60DesignContract=D3V60DesignContract(), model_step_seconds: int=300, control_update_seconds: int=600) -> pd.DataFrame:
    contract.validate(); basis.validate(); ids=tuple(str(v) for v in graph.actuator_ids); setting_cols=[f"setting:{aid}" for aid in ids]; missing=[x for x in ("checkpoint_id",*setting_cols) if x not in checkpoints.columns]
    if missing: raise ValueError(f"V6 D3 checkpoints missing: {missing[:10]}")
    if control_update_seconds%model_step_seconds or basis.horizon.control_block_steps != control_update_seconds//model_step_seconds: raise ValueError("V6 D3 cadence differs from control basis")
    metadata=[c for c in checkpoints.columns if c not in setting_cols]; records=[]
    members={group:np.flatnonzero(basis.grouping.group_id_by_actuator==group) for group in range(basis.group_count)}
    for row_number,(_,row) in enumerate(checkpoints.iterrows()):
        base=np.clip(np.asarray([float(row[c]) for c in setting_cols]),basis.min_setting,basis.max_setting); hold_values=np.repeat(base[None],basis.horizon.control_blocks,axis=0).astype(np.float32); hold=_sequence(ids,hold_values)
        common={c:row[c] for c in metadata}; common.update({"model_horizon_steps":basis.horizon.horizon_steps,"model_step_seconds":model_step_seconds,"control_update_seconds":control_update_seconds,"control_block_steps":basis.horizon.control_block_steps,"control_blocks":basis.horizon.control_blocks,"d3_time_contract":D3_TIME_CONTRACT,"d3_feasibility_contract":D3_FEASIBILITY_CONTRACT,"sequence_rate_feasible":True,"all_actuators_eligible":True,"fixed_active_subset":False,"v60_data_contract":V60_D3_DATA_CONTRACT,"v60_control_basis_contract":basis_manifest_v60(basis)["contract"]})
        zero=np.zeros((basis.temporal_basis_count,basis.group_count),np.float32); hold_sha=canonical_sequence_sha(hold); records.append({**common,"data_role":D3_V60_HOLD_ROLE,"sequence_index":0,"candidate_family":"hold_reference","v60_coefficients_json":json.dumps(zero.tolist()),"settings_sequence_json":json.dumps(hold,sort_keys=True),"sequence_sha256":hold_sha,"active_control_groups":0,"active_actuators":0})
        checkpoint_seed=_checkpoint_hash(row)^row_number; seen={hold_sha}
        for seq_idx,(family,coeff) in enumerate(_coefficient_specs(basis,checkpoint_seed=checkpoint_seed,contract=contract),start=1):
            trial=coeff.copy(); sequence=None; sha=""
            for attempt in range(max(4,basis.group_count+1)):
                sequence=_sequence(ids,_decode_numpy(basis,base,trial)); sha=canonical_sequence_sha(sequence)
                if sha not in seen: break
                if attempt==0: trial=-trial
                else: trial[(checkpoint_seed+2*seq_idx+attempt)%basis.temporal_basis_count,(checkpoint_seed+seq_idx+attempt)%basis.group_count]=0.35 if attempt%2 else -0.35
            if sequence is None or sha in seen: raise RuntimeError(f"cannot construct unique V6 candidate for {row['checkpoint_id']}")
            seen.add(sha); active=np.flatnonzero(np.any(np.abs(trial)>1e-8,axis=0)); active_a={int(a) for group in active for a in members[int(group)].tolist()}
            records.append({**common,"data_role":D3_V60_CANDIDATE_ROLE,"sequence_index":seq_idx,"candidate_family":family,"v60_coefficients_json":json.dumps(trial.tolist(),separators=(",",":")),"settings_sequence_json":json.dumps(sequence,sort_keys=True),"sequence_sha256":sha,"active_control_groups":int(len(active)),"active_actuators":int(len(active_a))})
    frame=pd.DataFrame.from_records(records)
    if frame.duplicated(["checkpoint_id","sequence_sha256"]).any(): raise RuntimeError("V6 D3 produced duplicate sequences")
    if not (frame.settings_sequence_json.map(lambda x:len(json.loads(str(x))))==basis.horizon.control_blocks).all(): raise RuntimeError("V6 D3 sequence length invalid")
    return frame


def targeted_d3_design_summary_v60(frame: pd.DataFrame) -> dict[str,object]:
    cand=frame[frame.data_role==D3_V60_CANDIDATE_ROLE]; hold=frame[frame.data_role==D3_V60_HOLD_ROLE]
    return {"contract":V60_D3_DATA_CONTRACT,"rows":int(len(frame)),"checkpoints":int(frame.checkpoint_id.nunique()),"hold_rows":int(len(hold)),"candidate_rows":int(len(cand)),"candidates_per_checkpoint":sorted(cand.groupby("checkpoint_id").size().unique().astype(int).tolist()),"candidate_families":cand.candidate_family.value_counts().sort_index().to_dict(),"active_control_groups":cand.active_control_groups.describe(percentiles=[.5,.9,.95]).to_dict(),"active_actuators":cand.active_actuators.describe(percentiles=[.5,.9,.95]).to_dict(),"all_rate_feasible":bool(frame.sequence_rate_feasible.astype(bool).all()),"roles":sorted(frame.data_role.astype(str).unique().tolist())}


def select_active_learning_pool_v60(candidate_scores: pd.DataFrame, *, budget: int, uncertainty_col: str="ensemble_std_m3", rank_margin_col: str="predicted_rank_margin_m3", gradient_norm_col: str="value_gradient_norm", manifold_distance_col: str="coefficient_novelty", rainfall_group_col: str="rainfall_group") -> pd.DataFrame:
    """P2: label only uncertain/near-rank/gradient-sensitive/novel Train-only manifold cases."""
    if budget<=0: raise ValueError("budget must be positive")
    required={uncertainty_col,rank_margin_col,gradient_norm_col,manifold_distance_col,rainfall_group_col}; missing=sorted(required-set(candidate_scores.columns))
    if missing: raise ValueError(f"active-learning pool missing {missing}")
    frame=candidate_scores.copy()
    def rz(s):
        x=pd.to_numeric(s,errors="raise").astype(float); med=float(x.median()); mad=float((x-med).abs().median()); mad=mad if mad>1e-12 else float(x.std()) or 1.; return (x-med)/mad
    frame["_score"]=rz(frame[uncertainty_col])-rz(frame[rank_margin_col].abs())+0.5*rz(frame[gradient_norm_col])+0.5*rz(frame[manifold_distance_col]); frame=frame.sort_values("_score",ascending=False,kind="mergesort"); selected=[]
    for _,group in frame.groupby(rainfall_group_col,sort=False):
        if len(selected)>=budget: break
        selected.append(int(group.index[0]))
    for i in frame.index:
        if len(selected)>=budget: break
        if int(i) not in selected: selected.append(int(i))
    out=frame.loc[selected].copy(); out["data_role"]=D3_V60_ACTIVE_ROLE; return out.drop(columns=["_score"]).reset_index(drop=True)


__all__=["D3V60DesignContract","D3_V60_ACTIVE_ROLE","D3_V60_CANDIDATE_ROLE","D3_V60_HOLD_ROLE","design_targeted_d3_v60","select_active_learning_pool_v60","targeted_d3_design_summary_v60"]
