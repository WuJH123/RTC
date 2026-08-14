"""Fit the frozen TrainFit one-sided TFV/PFV false-benefit admission budget."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70
from rtc.step2_priority_value_v123 import PriorityValueCacheV123
from rtc.step2_train_response_v60 import V60TrainCache, InputNormalizationV60, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step3_calibration_v123 import fit_one_sided_value_calibration_v123


def _stats(values: list[np.ndarray], channels: int) -> tuple[np.ndarray, np.ndarray]:
    flat = [np.asarray(v, dtype=np.float64).reshape(-1, channels) for v in values]
    x = np.concatenate(flat, axis=0)
    return x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), 1e-6).astype(np.float32)


def _causal_norm(base: V60TrainCache, store, fit: list[str]) -> InputNormalizationV60:
    idx = store.index()
    states = [base.entry(n).arrays["initial_state"][base.entry(n).reference_index] for n in fit]
    flows = [base.entry(n).arrays["previous_actuator_flow"][base.entry(n).reference_index] for n in fit]
    rains = [store.forecast_mmhr[idx[n]] for n in fit]
    sm, ss = _stats(states, states[0].shape[-1]); rm, rs = _stats(rains, rains[0].shape[-1]); fm, fs = _stats(flows, flows[0].shape[-1])
    return InputNormalizationV60(sm, ss, rm, rs, fm, fs)


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 TrainFit admission calibration")
    parser.add_argument("--graph", required=True); parser.add_argument("--cache-manifest", required=True); parser.add_argument("--causal-store", required=True); parser.add_argument("--tfv-checkpoint", required=True); parser.add_argument("--pfv-checkpoint", required=True); parser.add_argument("--priority-nodes", required=True); parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    base = V60TrainCache(args.cache_manifest); store = load_causal_forecast_store_v123(args.causal_store); causal = CausalForecastValueCacheV123(base, store)
    priority = tuple(x.strip() for x in Path(args.priority_nodes).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")); pfv_cache = PriorityValueCacheV123(causal, priority)
    names = sorted(base.names("D2") + base.targeted_d3_names()); fit, _ = deterministic_rainfall_split_v60(base, names=names, holdout_fraction=0.20)
    graph = load_graph_v120(args.graph); basis = build_control_basis_v60(graph); device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"; prepared = prepare_static_v60(graph, device); norm = _causal_norm(base, store, fit); base_scales = derive_target_scales_v70(base, fit)
    tfv_scale = float(json.loads(Path(args.tfv_checkpoint).with_name("STEP2_V123_TFV_CAUSAL_ABLATION.json").read_text(encoding="utf-8"))["target_scales"]["direct_tfv_scale_m3"]) if Path(args.tfv_checkpoint).with_name("STEP2_V123_TFV_CAUSAL_ABLATION.json").exists() else float(base_scales.direct_tfv_scale_m3)
    pfv_scale = float(json.loads(Path(args.pfv_checkpoint).with_name("STEP2_V123_PFV_VALUE_REPORT.json").read_text(encoding="utf-8"))["target_scale_pfv_m3"]) if Path(args.pfv_checkpoint).with_name("STEP2_V123_PFV_VALUE_REPORT.json").exists() else 100.0
    first = base.entry(fit[0]).arrays
    def model(scale: float) -> ControlValueSurrogateV70:
        return ControlValueSurrogateV70(state_dim=first["initial_state"].shape[-1], rainfall_dim=first["rainfall"].shape[-1], physics_dim=prepared.actuator_physics.shape[1], actuator_count=len(graph.actuator_ids), temporal_basis=basis.temporal_basis, control_block_steps=basis.horizon.control_block_steps, tfv_scale_m3=scale, hidden_dim=96, actuator_embedding_dim=16)
    tfv = model(tfv_scale); pfv = model(pfv_scale); tfv.load_state_dict(torch.load(args.tfv_checkpoint, map_location=device)["state_dict"]); pfv.load_state_dict(torch.load(args.pfv_checkpoint, map_location=device)["state_dict"]); tfv.to(device).eval(); pfv.to(device).eval()
    truth_tfv=[]; pred_tfv=[]; truth_pfv=[]; pred_pfv=[]; groups=[]
    with torch.no_grad():
        for name in fit:
            bt = causal.batch(name, norm, device); bp = pfv_cache.batch(name, norm, device)
            pt = tfv(bt.initial_state, bt.rainfall, bt.reference_settings, bt.candidate_settings, bt.previous_actuator_flow, prepared).delta_tfv_m3[0].cpu().numpy()
            pp = pfv(bp.initial_state, bp.rainfall, bp.reference_settings, bp.candidate_settings, bp.previous_actuator_flow, prepared).delta_tfv_m3[0].cpu().numpy()
            truth_tfv.extend(bt.true_delta_tfv_m3[0].cpu().numpy().tolist()); pred_tfv.extend(pt.tolist()); truth_pfv.extend(bp.true_delta_tfv_m3[0].cpu().numpy().tolist()); pred_pfv.extend(pp.tolist()); groups.extend([str(base.entry(name).rainfall_group)] * len(pt))
    calibration = fit_one_sided_value_calibration_v123(tfv_truth_m3=truth_tfv, tfv_prediction_m3=pred_tfv, pfv_truth_m3=truth_pfv, pfv_prediction_m3=pred_pfv, rainfall_groups=groups, quantile=0.90)
    payload = {"contract": "PROJECT7_V123_TRAINFIT_ADMISSION_CALIBRATION_V1", "calibration": calibration.as_payload(), "fit_groups": len(fit), "fit_events": sorted({base.entry(n).event_id for n in fit}), "tfv_checkpoint_sha256": hashlib.sha256(Path(args.tfv_checkpoint).read_bytes()).hexdigest(), "pfv_checkpoint_sha256": hashlib.sha256(Path(args.pfv_checkpoint).read_bytes()).hexdigest(), "causal_store_sha256": hashlib.sha256(Path(args.causal_store).read_bytes()).hexdigest(), "future_realized_rainfall_used_as_model_input": False, "boundary": {"new_swmm": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False}}
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n",encoding="utf-8"); out.with_suffix(".md").write_text(f"# V123 admission calibration\n\nTrainFit groups: {len(fit)}\nTFV false-benefit margin (q90): {calibration.tfv_false_benefit_margin_m3:.3f} m3\nPFV residual q90: {calibration.pfv_truth_minus_prediction_q_m3:.3f} m3\n\nFrozen from TrainFit only; no Holdout tuning.\n",encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
