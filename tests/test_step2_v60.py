from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import torch

from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import ControlValueSurrogateV60, DualStep2SurrogateV60, HydraulicResponseSurrogateV60, prepare_static_v60
from rtc.step2_d3_design_v60 import D3V60DesignContract, design_targeted_d3_v60
from rtc.step2_train_response_v60 import TargetScalesV60, hydraulic_critical_weights_v60, listwise_loss_v60


def _graph():
    n,a=14,109; edges=[]
    for i in range(n-1): edges += [(i,i+1),(i+1,i)]
    names=("invert_elevation_m","max_depth_m","is_junction","is_outfall","is_storage","is_divider","init_depth_m","surcharge_depth_m","ponded_area_m2","storage_capacity_m3","storage_area_full_m2","conduit_in_count","conduit_out_count","conduit_in_length_sum_m","conduit_out_length_sum_m","conduit_in_roughness_mean","conduit_out_roughness_mean","conduit_in_geom1_mean_m","conduit_out_geom1_mean_m","subcatchment_count","subcatchment_area_m2","subcatchment_impervious_area_m2","subcatchment_width_area_weighted_m","subcatchment_slope_area_weighted_pct","infiltration_max_rate_area_weighted_mmhr","infiltration_min_rate_area_weighted_mmhr")
    st=np.zeros((n,len(names)),np.float32); st[:,1]=3.; st[:,2]=1.; st[:,7]=.5; st[0,4]=1.; st[0,2]=0.; st[0,9]=1000.; st[1,9]=500.
    pn=("is_pump","is_orifice","is_weir","is_outlet","min_setting","max_setting","pump_curve_max_flow_m3s","pump_curve_max_x_m","pump_curve_point_count","offset_or_crest_m","discharge_coefficient","has_flap_gate","xsection_geom1_m","xsection_geom2_m","xsection_geom3_m","xsection_geom4_m","xsection_is_circular","xsection_is_rect_closed","xsection_is_rect_open")
    p=np.zeros((a,len(pn)),np.float32); p[:,5]=1.; p[:57,0]=1.; p[57:99,1]=1.; p[99:,2]=1.; p[:57,6]=2.
    return SimpleNamespace(node_ids=tuple(f"n{i}" for i in range(n)),edge_index=np.asarray(edges,dtype=np.int64).T,static_node_features=st,static_node_feature_names=names,actuator_ids=tuple(f"a{i}" for i in range(a)),actuator_upstream=np.arange(a)%n,actuator_downstream=(np.arange(a)+1)%n,actuator_physics=p,actuator_physics_feature_names=pn,system_units="SI")


def _models(g):
    prep=prepare_static_v60(g); common=dict(state_dim=6,rainfall_dim=1,node_static_dim=g.static_node_features.shape[1],physics_dim=prep.actuator_physics.shape[1],actuator_count=109,hidden_dim=16,latent_dim=8,temporal_dim=6)
    return ControlValueSurrogateV60(tfv_rate_scale_m3s=10.,**common), HydraulicResponseSurrogateV60(state_scale=torch.ones(6),flow_scale=torch.ones(109),**common), prep


def _inputs(c=3):
    init=torch.zeros(1,14,6); init[...,0]=.3; rain=torch.zeros(1,72,14,1); rain[:,10:30,:,0]=.2; ref=torch.full((1,72,109),.4); cand=ref[:,None].expand(1,c,72,109).clone(); cand[:,1,4:20,0]=.7
    if c>2: cand[:,2,10:40,57]=.1
    return init,rain,ref,cand,torch.arange(73,dtype=torch.float32)[None]*300


def test_v60_multi_resolution_and_low_dimensional_basis():
    horizon=MultiResolutionHorizonV60(); assert horizon.indices()[-1]==71 and 10<len(horizon.indices())<72
    basis=build_control_basis_v60(_graph()); assert basis.coefficient_dimension < 109*36
    ref=torch.full((1,2,72,109),.5); coeff=torch.zeros(1,2,basis.temporal_basis_count,basis.group_count,requires_grad=True); coeff.data[0,1,0,0]=.7
    out=basis.decode(ref,coeff); blocks=out[...,::2,:]; assert out.shape==(1,2,72,109); assert out.min()>=0 and out.max()<=1; assert torch.max(torch.abs(blocks[...,1:,:]-blocks[...,:-1,:]))<=.500001


def test_v60_targeted_d3_is_sparse_group_structured_and_unique():
    g=_graph(); basis=build_control_basis_v60(g); row={"checkpoint_id":"c0","event_id":"e0","rainfall_group":"r0","scientific_split":"development","development_fold":"train","checkpoint_minutes":60,"inp_path":"x","trajectory_metadata_path":"y"}
    for aid in g.actuator_ids: row[f"setting:{aid}"]=.5
    frame=design_targeted_d3_v60(pd.DataFrame([row]),g,basis,contract=D3V60DesignContract(candidates_per_checkpoint=24))
    assert len(frame)==25 and (frame.data_role=="D3_HOLD_REFERENCE").sum()==1 and frame.sequence_sha256.nunique()==25
    cand=frame[frame.data_role!="D3_HOLD_REFERENCE"]; assert cand.active_control_groups.max()<=5 and cand.active_actuators.median()<109; assert all(len(json.loads(x))==36 for x in frame.settings_sequence_json)


def test_v60_state_conditioning_exact_zero_causality_and_disjoint_surrogates():
    torch.manual_seed(3); g=_graph(); value,hydraulic,prep=_models(g); init,rain,ref,cand,elapsed=_inputs()
    zero=value(init,rain,ref,ref[:,None],prep,elapsed); assert torch.equal(zero.delta_tfv_m3,torch.zeros_like(zero.delta_tfv_m3))
    hz=hydraulic(init,rain,ref,ref[:,None],prep); assert torch.equal(hz.delta_states_physical,torch.zeros_like(hz.delta_states_physical))
    a=value(init,rain,ref,cand,prep,elapsed); changed=init.clone(); changed[:,0,0]=1.8; b=value(changed,rain,ref,cand,prep,elapsed); assert not torch.allclose(a.joint_context_before_scatter,b.joint_context_before_scatter)
    future=cand.clone(); future[:,1,40:,2]=.9; c=value(init,rain,ref,future,prep,elapsed); assert torch.allclose(a.delta_tfv_prefix_m3[:,:,:40],c.delta_tfv_prefix_m3[:,:,:40],atol=1e-5,rtol=1e-5)
    DualStep2SurrogateV60(value,hydraulic).assert_disjoint_parameters()


def test_v60_critical_hydraulic_weighting_and_source_scales():
    g=_graph(); _,_,prep=_models(g); states=torch.zeros(1,2,3,14,6); states[...,0]=.1; base=hydraulic_critical_weights_v60(states,prep); high=states.clone(); high[...,0,0]=2.5; high[...,0,3]=900.; weighted=hydraulic_critical_weights_v60(high,prep); assert weighted[...,0].mean()>base[...,0].mean()
    pred=torch.tensor([[3.,1.,2.]],requires_grad=True); truth=torch.tensor([[30.,10.,20.]]); loss=listwise_loss_v60(pred,truth,100.); loss.backward(); assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    scales=TargetScalesV60(np.ones(6),np.ones(109),10.,100.,1.); assert scales.tfv_scale("D2")==10 and scales.tfv_scale("D3")==100


def test_v60_mpc_value_is_differentiable_in_control_coefficients():
    torch.manual_seed(4); g=_graph(); basis=build_control_basis_v60(g); value,_,prep=_models(g); init,rain,ref,_,elapsed=_inputs(2); coeff=torch.zeros(1,2,basis.temporal_basis_count,basis.group_count,requires_grad=True); coeff.data[0,1,0,0]=.5; candidate=basis.decode(ref[:,None].expand(1,2,-1,-1),coeff); value(init,rain,ref,candidate,prep,elapsed).delta_tfv_m3.sum().backward(); assert coeff.grad is not None and torch.isfinite(coeff.grad).all() and torch.count_nonzero(coeff.grad)>0
