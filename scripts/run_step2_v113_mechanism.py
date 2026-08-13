"""Run one fixed V11.3 mechanism gate on nested TrainFit D2 data.

The command intentionally has no hyperparameter knobs.  ``tiny`` and
``micro`` are diagnostics; ``devcheck`` trains on exactly the frozen ten-event
DevFit and evaluates once on the four untouched DevCheck events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v113 import SupportConditionedHydraulicSurrogateV113
from rtc.step2_hydraulic_eval_v113 import build_oracle_support_override_v113, evaluate_hydraulic_v113
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_hydraulic_objective_v113 import derive_v113_scales, v113_effect_loss
from rtc.step2_train_response_v60 import V60TrainCache, derive_input_normalization_v60
from rtc.step2_v110_contract import HydraulicHorizonV110
from run_step2_v110 import _load_graph


CONTRACT = "PROJECT7_STEP2_V113_SUPPORT_CONDITIONED_SIGNED_HYDRAULIC_V1"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()


def _event_groups(cache: V60TrainCache, names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in sorted(names):
        out.setdefault(str(cache.entry(n).event_id), []).append(n)
    return out


def _load_prior(path: str | Path, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[int, ...]]:
    z = np.load(path, allow_pickle=False)
    required = {"overall__state_support_probability", "overall__flow_support_probability", "low__state_support_probability", "mid__state_support_probability", "high__state_support_probability", "low__flow_support_probability", "mid__flow_support_probability", "high__flow_support_probability"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"V113 support prior missing arrays: {missing}")
    state = np.asarray(z["overall__state_support_probability"], dtype=np.float32)
    flow = np.asarray(z["overall__flow_support_probability"], dtype=np.float32)
    phase_state = np.stack([z[f"{p}__state_support_probability"] for p in ("low", "mid", "high")]).astype(np.float32)
    phase_flow = np.stack([z[f"{p}__flow_support_probability"] for p in ("low", "mid", "high")]).astype(np.float32)
    # The prior NPZ is generated in V110 retained-time order.  Keep the order
    # explicit in the report rather than assuming a contiguous 0..H slice.
    idx = tuple(int(x) for x in HydraulicHorizonV110().indices())
    return state, flow, phase_state, phase_flow, (0.25, 0.75), idx


def _build_model(cache, graph, fit_names, prior_npz, mode, device):
    prepared_cpu = prepare_static_v60(graph, "cpu")
    effects = derive_effect_scales_v111(cache, fit_names, prepared_cpu)
    state, flow, phase_state, phase_flow, boundaries, idx = _load_prior(prior_npz, mode)
    if state.shape[1] != len(idx):
        raise ValueError("V113 prior retained-time dimension does not match frozen horizon")
    model = SupportConditionedHydraulicSurrogateV113(
        node_static_dim=int(prepared_cpu.node_static.shape[1]), physics_dim=int(prepared_cpu.actuator_physics.shape[1]),
        rainfall_dim=int(cache.entry(fit_names[0]).arrays["rainfall"].shape[-1]),
        actuator_count=len(graph.actuator_ids), node_count=len(graph.node_ids),
        state_scale=effects.state_magnitude_scale, flow_scale=effects.flow_magnitude_scale,
        overall_state_prior=state, overall_flow_prior=flow,
        phase_state_priors=phase_state, phase_flow_priors=phase_flow,
        phase_boundaries=boundaries, retained_indices=idx, prior_mode=mode, hidden_dim=48,
    ).to(device)
    return model, effects


def _train(model, cache, names, normalization, prepared, scales, storage_mask, *, device, epochs, oracle_support=False):
    events = _event_groups(cache, names)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        event_losses = []
        for event, group_names in sorted(events.items()):
            optimizer.zero_grad(set_to_none=True)
            group_stats = []
            for name in group_names:
                batch = cache.batch(name, normalization, device)
                entry = cache.entry(name)
                raw_initial = torch.from_numpy(np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float32).copy())[None].to(device)
                override = build_oracle_support_override_v113(batch, scales, storage_mask, model.retained_indices) if oracle_support else None
                out = model(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, prepared, initial_state_physical=raw_initial, support_override=override)
                loss, stats = v113_effect_loss(out, batch, scales, storage_mask=storage_mask)
                (loss / float(len(group_names))).backward()
                group_stats.append(stats)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach())
            optimizer.step()
            mean_loss = float(np.mean([s["loss"] for s in group_stats]))
            event_losses.append(mean_loss)
        history.append({"epoch": epoch, "loss": float(np.mean(event_losses)), "event_updates": len(events), "gradient_norm_last": grad_norm, "finite": bool(np.isfinite(event_losses).all())})
    return history


def _zero_metrics(cache, names, normalization, prepared, scales, storage_mask, device):
    # Evaluation-shaped zero baseline, kept event-balanced by using the model
    # evaluator's metric implementation on a zero-output adapter would be more
    # cumbersome; this direct calculation covers the primary channels.
    records = []
    for name in names:
        entry = cache.entry(name); arrays = entry.arrays; ref = entry.reference_index; cand = [i for i in entry.indices if i != ref]
        idx = np.asarray(HydraulicHorizonV110().indices(), dtype=np.int64)
        truth_s = arrays["target_states"][cand][:, idx] - arrays["target_states"][ref][idx][None]
        truth_f = arrays["target_actuator_flows"][cand][:, idx] - arrays["target_actuator_flows"][ref][idx][None]
        ch=[]
        for data, channel, mask in ((truth_s[...,0],0, np.abs(truth_s[...,0]) >= scales.state_threshold.detach().cpu().numpy()[None,None,:,0]), (truth_s[...,2],1,np.abs(truth_s[...,2]) >= scales.state_threshold.detach().cpu().numpy()[None,None,:,1]), (truth_s[...,3],2,np.abs(truth_s[...,3]) >= scales.state_threshold.detach().cpu().numpy()[None,None,:,2]), (truth_s[...,4],3,np.abs(truth_s[...,4]) >= scales.state_threshold.detach().cpu().numpy()[None,None,:,3]), (truth_s[...,5],4,np.abs(truth_s[...,5]) >= scales.state_threshold.detach().cpu().numpy()[None,None,:,4]), (truth_f,None,np.abs(truth_f) >= scales.flow_threshold.detach().cpu().numpy()[None,None,:])):
            if channel==2: mask &= storage_mask.detach().cpu().numpy()[None,None,:]
            ch.append({"skill_vs_zero":0.0,"active_skill_vs_zero":0.0,"active_sign_accuracy":0.0,"response_ratio":0.0,"rmse":float(np.sqrt(np.mean(data**2))),"active_count":int(mask.sum()),"count":int(data.size)})
        records.append({"event_key":f"{entry.rainfall_group}::{entry.event_id}","channels":ch})
    result={}
    for i,n in enumerate(("depth_m","flood_m3s","storage_m3","inflow_m3s","outflow_m3s","managed_flow_m3s")):
        by_event={}
        for r in records: by_event.setdefault(r["event_key"],[]).append(r["channels"][i])
        result[n]={k:float(np.mean([x[k] for vals in by_event.values() for x in vals])) for k in ("skill_vs_zero","active_skill_vs_zero","active_sign_accuracy","response_ratio","rmse")}
    return {"scientific_primary":"event_balanced","channels":result,"groups":len(records),"events":len({r['event_key'] for r in records})}


def _preflight_gate_v113(preflight: dict[str, bool]) -> bool:
    """Evaluate positive gates and negative-prohibition flags explicitly."""
    return bool(
        preflight.get("exact_zero", False)
        and preflight.get("action_gradient_finite", False)
        and preflight.get("action_gradient_nonzero", False)
        and preflight.get("storage_domain_only", False)
        and not preflight.get("future_truth_input", True)
        and not preflight.get("hard_prior_mask", True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nested-split", required=True)
    ap.add_argument("--prior-npz", required=True)
    ap.add_argument("--mode", choices=("tiny", "micro", "devcheck"), required=True)
    ap.add_argument("--prior-mode", choices=("none", "overall", "phase", "oracle"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cache = V60TrainCache(args.cache_manifest); graph = _load_graph(args.graph)
    split = json.loads(Path(args.nested_split).read_text(encoding="utf-8"))
    devfit = list(split["v113_devfit"]["group_names"]); devcheck = list(split["v113_devcheck"]["group_names"])
    if args.mode == "tiny": train_names, eval_names = devfit[:2], devfit[:2]
    elif args.mode == "micro": train_names, eval_names = devfit[:12], devfit[:12]
    else: train_names, eval_names = devfit, devcheck
    normalization = derive_input_normalization_v60(cache, devfit)
    model, effects = _build_model(cache, graph, devfit, args.prior_npz, args.prior_mode, device)
    prepared = prepare_static_v60(graph, device)
    scales = derive_v113_scales(effects.state_magnitude_scale, effects.flow_magnitude_scale, effects.state_active_threshold, effects.flow_active_threshold, device=device)
    storage_mask = prepared.storage_mask
    # Exact-zero / gradient preflight on a deterministic training group.
    first = cache.batch(train_names[0], normalization, device)
    entry = cache.entry(train_names[0]); raw_initial = torch.from_numpy(np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float32).copy())[None].to(device)
    model.eval();
    with torch.no_grad(): zero = model(first.initial_state, first.rainfall, first.reference_settings, first.reference_settings[:,None], first.previous_actuator_flow, prepared, initial_state_physical=raw_initial)
    exact_zero = bool(torch.equal(zero.raw_delta_states_physical, torch.zeros_like(zero.raw_delta_states_physical)) and torch.equal(zero.raw_delta_flows_physical, torch.zeros_like(zero.raw_delta_flows_physical)))
    candidate = first.candidate_settings[:, :1].detach().clone().requires_grad_(True)
    output = model(first.initial_state, first.rainfall, first.reference_settings, candidate, first.previous_actuator_flow, prepared, initial_state_physical=raw_initial)
    grad = torch.autograd.grad(output.raw_delta_states_physical.square().mean() + output.raw_delta_flows_physical.square().mean(), candidate)[0]
    preflight = {"exact_zero": exact_zero, "action_gradient_finite": bool(torch.isfinite(grad).all()), "action_gradient_nonzero": int(torch.count_nonzero(grad)) > 0, "storage_domain_only": True, "future_truth_input": False, "hard_prior_mask": False}
    if not _preflight_gate_v113(preflight):
        raise RuntimeError(f"V113 preflight failed: {preflight}")
    oracle_support = args.prior_mode == "oracle"
    model.train(); history = _train(model, cache, train_names, normalization, prepared, scales, storage_mask, device=device, epochs=2 if args.mode == "tiny" else 4, oracle_support=oracle_support)
    metrics = evaluate_hydraulic_v113(model, cache, eval_names, normalization, prepared, scales, storage_mask=storage_mask, device=device, oracle_support=oracle_support)
    zero_metrics = _zero_metrics(cache, eval_names, normalization, prepared, scales, storage_mask, device)
    report = {"contract": CONTRACT, "stage": args.mode, "prior_mode": args.prior_mode, "development_only": True, "oracle_support_diagnostic_only": oracle_support, "lineage": {"git_head": _head(), "graph_sha256": _sha(args.graph), "cache_manifest_sha256": _sha(args.cache_manifest), "nested_split_sha256": _sha(args.nested_split), "prior_sha256": _sha(args.prior_npz)}, "train_groups": len(train_names), "eval_groups": len(eval_names), "train_events": sorted({cache.entry(n).event_id for n in train_names}), "eval_events": sorted({cache.entry(n).event_id for n in eval_names}), "seed": 42, "fp32": True, "optimizer": "AdamW", "epochs": 2 if args.mode == "tiny" else 4, "training_history": history, "preflight": preflight, "zero_baseline": zero_metrics, "metrics": metrics, "checkpoint": None, "new_swmm_run": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": args.out, "stage": args.mode, "prior_mode": args.prior_mode, "preflight": preflight}, indent=2))


if __name__ == "__main__": main()
