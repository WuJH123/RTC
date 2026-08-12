from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70, DirectValueOutputV70
from rtc.step2_hydraulic_effect_v70 import retained_onset_targets_v70
from rtc.step2_train_response_v70 import value_loss_v70


def test_v70_report_lineage_reads_source_manifest(monkeypatch):
    import scripts.run_step2_v70 as runner

    monkeypatch.setattr(
        runner,
        "validate_v60_cache_lineage",
        lambda _path: {
            "v60_control_basis_sha256": "basis-sha",
            "v60_design_contract_sha256": "design-sha",
        },
    )
    assert runner._cache_lineage_hashes("cache.json") == {
        "basis_sha256_from_cache_lineage": "basis-sha",
        "design_sha256_from_cache_lineage": "design-sha",
    }


def _graph():
    n, a = 14, 109
    edges = np.asarray([(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)], dtype=np.int64).T
    sn = ("invert_elevation_m","max_depth_m","is_junction","is_outfall","is_storage","is_divider","init_depth_m","surcharge_depth_m","ponded_area_m2","storage_capacity_m3","storage_area_full_m2","conduit_in_count","conduit_out_count","conduit_in_length_sum_m","conduit_out_length_sum_m","conduit_in_roughness_mean","conduit_out_roughness_mean","conduit_in_geom1_mean_m","conduit_out_geom1_mean_m","subcatchment_count","subcatchment_area_m2","subcatchment_impervious_area_m2","subcatchment_width_area_weighted_m","subcatchment_slope_area_weighted_pct","infiltration_max_rate_area_weighted_mmhr","infiltration_min_rate_area_weighted_mmhr")
    static = np.zeros((n, len(sn)), np.float32); static[:,1]=3.; static[:,2]=1.; static[:,7]=.5; static[0,4]=1.; static[0,2]=0.; static[0,9]=1000.
    pn = ("is_pump","is_orifice","is_weir","is_outlet","min_setting","max_setting","pump_curve_max_flow_m3s","pump_curve_max_x_m","pump_curve_point_count","offset_or_crest_m","discharge_coefficient","has_flap_gate","xsection_geom1_m","xsection_geom2_m","xsection_geom3_m","xsection_geom4_m","xsection_is_circular","xsection_is_rect_closed","xsection_is_rect_open")
    physics=np.zeros((a,len(pn)),np.float32); physics[:,5]=1.; physics[:57,0]=1.; physics[57:99,1]=1.; physics[99:,2]=1.; physics[:57,6]=2.
    return SimpleNamespace(node_ids=tuple(f"n{i}" for i in range(n)),edge_index=edges,static_node_features=static,static_node_feature_names=sn,actuator_ids=tuple(f"a{i}" for i in range(a)),actuator_upstream=np.arange(a)%n,actuator_downstream=(np.arange(a)+1)%n,actuator_physics=physics,actuator_physics_feature_names=pn,system_units="SI")


def _case():
    graph=_graph(); basis=build_control_basis_v60(graph); prepared=prepare_static_v60(graph)
    model=ControlValueSurrogateV70(state_dim=6,rainfall_dim=1,physics_dim=prepared.actuator_physics.shape[1],actuator_count=109,temporal_basis=basis.temporal_basis,control_block_steps=basis.horizon.control_block_steps,tfv_scale_m3=10000.,hidden_dim=32,actuator_embedding_dim=8)
    initial=torch.zeros(1,14,6); initial[...,0]=.3
    rain=torch.zeros(1,72,14,1); rain[:,10:30,:,0]=.2
    ref=torch.full((1,72,109),.4); cand=ref[:,None].expand(1,3,72,109).clone(); cand[:,1,4:20,0]=.7; cand[:,2,10:40,57]=.1
    flow=torch.linspace(0.,1.,109)[None]
    return basis,prepared,model,initial,rain,ref,cand,flow


def test_v70_exact_zero_and_action_gradient():
    _,p,m,x,r,ref,cand,flow=_case(); out=m(x,r,ref,cand,flow,p)
    assert torch.equal(out.delta_tfv_m3[:,0],torch.zeros_like(out.delta_tfv_m3[:,0]))
    action=cand[:,1:2].clone().requires_grad_(True); value=m(x,r,ref,action,flow,p).delta_tfv_m3
    g=torch.autograd.grad(value.sum(),action)[0]; assert torch.isfinite(g).all() and torch.count_nonzero(g)>0


def test_v70_loss_does_not_reward_tiny_response():
    truth=torch.tensor([[-20000.,-5000.,4000.,18000.]]); scale=10000.
    def output(pred):
        return DirectValueOutputV70(pred,torch.asinh(pred/scale),torch.zeros(1,4,1,1),torch.zeros(1,4,1))
    zero,_=value_loss_v70(output(torch.zeros_like(truth)),truth,scale_m3=scale)
    tiny,_=value_loss_v70(output(1e-5*truth),truth,scale_m3=scale)
    perfect,_=value_loss_v70(output(truth),truth,scale_m3=scale)
    assert perfect < .25*zero and tiny > .90*zero


def test_v70_onset_means_dry_to_flood_transition():
    states=torch.zeros(1,1,3,2,6); states[0,0,:,0,2]=1.; states[0,0,1:,1,2]=1.
    target=retained_onset_targets_v70(states,torch.tensor([0,1,2]),initial_flood_m3s=torch.tensor([[1.,0.]]),epsilon_m3s=1e-7)
    assert target[0,0,:,0].sum().item()==0
    assert target[0,0,:,1].tolist()==[0.,1.,0.]


def test_v70_previous_flow_and_coefficient_paths_are_active():
    basis,p,m,x,r,ref,cand,flow=_case()
    first=m(x,r,ref,cand[:,1:2],flow,p).delta_tfv_m3; changed=flow.clone(); changed[:,0]+=2.
    second=m(x,r,ref,cand[:,1:2],changed,p).delta_tfv_m3; assert not torch.allclose(first,second)
    coeff=torch.zeros(1,2,basis.temporal_basis_count,basis.group_count,requires_grad=True)
    with torch.no_grad(): coeff[0,0,0,0]=.4; coeff[0,1,2,1]=-.3
    val=m.forward_coefficients(initial_state=x,rainfall=r,reference_settings=ref,coefficients=coeff,previous_actuator_flow=flow,prepared=p,basis=basis).delta_tfv_m3
    g=torch.autograd.grad(val.sum(),coeff)[0]; assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
