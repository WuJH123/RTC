"""Run the isolated Project7 Step2 NG1 Development experiment once.

This runner intentionally does not alter production routing and consumes only
the already prepared V5 Development caches.  It never invokes SWMM.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from rtc.development_profile_v128 import get_execution_profile, profile_groups
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_state_store_v127 import (
    CausalStep1StateCacheV127,
    derive_v127_input_normalization,
    load_causal_state_store_v127,
)
from rtc.step2_tfv_support import derive_direct_tfv_action_support
from rtc.step2_tfv_value import DirectTFVValueDesign
from rtc.step2_tfv_value_ng1 import NG1_CONTRACT, NG1ProcessAwareDirectTFVValueModel, build_control_interaction_graph
from rtc.step2_tfv_value_training import evaluate_direct_tfv_value_model
from rtc.step2_tfv_value_training_ng1 import (
    NG1_TRAINING_CONTRACT,
    NG1TrainingDesign,
    train_ng1_model,
)
from rtc.step2_tfv_value_training_v4 import derive_direct_tfv_target_scale_m3
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


EXPECTED_COUNTS = (112, 112, 32, 32, 33, 15)
REPORT_FILENAME = "STEP2_NG1_D2D3_PROCESS_AWARE_REPORT.json"
CHECKPOINT_FILENAME = "step2_ng1.pt"
D2_CHECKPOINT_FILENAME = "step2_d2_main.pt"
D2_REPORT_FILENAME = "STEP2_NG1_D2_MAIN_REPORT.json"
SEED = 42


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "dev"), required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--d4-fit-cache", required=True)
    parser.add_argument("--d4-audit-cache", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--causal-state-store", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def _selected_groups(profile: str, *, fit_d2: list[str], fit_d3: list[str], hold_d2: list[str], hold_d3: list[str], d4_fit: list[str], d4_audit: list[str]) -> dict[str, list[str]]:
    if profile == "smoke":
        return profile_groups(
            get_execution_profile("smoke"), fit_d2=fit_d2, fit_d3=fit_d3,
            hold_d2=hold_d2, hold_d3=hold_d3, d4_fit=d4_fit, d4_audit=d4_audit,
            one_group=False,
        )
    return {"fit_d2": sorted(fit_d2), "fit_d3": sorted(fit_d3), "hold_d2": sorted(hold_d2), "hold_d3": sorted(hold_d3), "d4_fit": sorted(d4_fit), "d4_audit": sorted(d4_audit)}


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _normalization_payload(normalization: object) -> dict[str, list[float]]:
    return {name: np.asarray(getattr(normalization, name)).astype(float).tolist() for name in ("state_mean", "state_std", "rainfall_mean", "rainfall_std", "flow_mean", "flow_std")}


def _model(*, state_dim: int, rainfall_dim: int, physics_dim: int, target_scale: float, graph_payload: dict[str, object], device: torch.device) -> NG1ProcessAwareDirectTFVValueModel:
    return NG1ProcessAwareDirectTFVValueModel(
        state_dim=state_dim,
        rainfall_dim=rainfall_dim,
        actuator_physics_dim=physics_dim,
        target_scale_m3=target_scale,
        interaction_graph=graph_payload,
        design=DirectTFVValueDesign(),
    ).to(device)


def main() -> None:
    started = time.perf_counter()
    args = _parser().parse_args()
    if int(args.seed) != SEED:
        raise ValueError("NG1 first Development run is frozen to seed=42")
    _seed_everything(SEED)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(args.graph)
    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rainfall_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rainfall_store.validate()
    state_store.validate()
    fit, holdout = deterministic_rainfall_split_v60(
        base, names=sorted(base.names("D2") + base.targeted_d3_names()), holdout_fraction=0.20
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]
    d4_fit = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit = d4_audit_raw.names(D4_SOURCE_KIND)
    counts = tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3, d4_fit, d4_audit)))
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"NG1 canonical split mismatch: expected {EXPECTED_COUNTS}, got {counts}")
    selected = _selected_groups(profile=str(args.profile), fit_d2=fit_d2, fit_d3=fit_d3, hold_d2=hold_d2, hold_d3=hold_d3, d4_fit=d4_fit, d4_audit=d4_audit)
    if str(args.profile) == "dev" and tuple(len(selected[key]) for key in ("fit_d2", "fit_d3", "hold_d2", "hold_d3", "d4_fit", "d4_audit")) != EXPECTED_COUNTS:
        raise RuntimeError("NG1 dev must use all existing canonical Development groups")

    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rainfall_store), state_store)
    d4_fit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rainfall_store), state_store)
    d4_audit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rainfall_store), state_store)
    normalization = derive_v127_input_normalization(base_cache=base, causal_rainfall=rainfall_store, causal_state=state_store, fit_names=selected["fit_d2"] + selected["fit_d3"])
    raw_caches = {"D2": base, "D3": base, "D4": d4_fit_raw}
    raw_groups = {"D2": selected["fit_d2"], "D3": selected["fit_d3"], "D4": selected["d4_fit"]}
    target_scale = derive_direct_tfv_target_scale_m3(raw_caches, raw_groups)
    support = derive_direct_tfv_action_support(raw_caches, raw_groups, actuator_ids=graph.actuator_ids, control_block_steps=2)
    if str(args.profile) == "dev" and int(support["single_facility_coverage_count"]) != 109:
        raise RuntimeError("NG1 requires complete 109-facility D2 TrainFit coverage")
    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    graph_payload = build_control_interaction_graph(graph)
    graph_path = out / "STEP2_NG1_CONTROL_INTERACTION_GRAPH.json"
    graph_path.write_text(json.dumps(_jsonable(graph_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = _model(state_dim=int(first_state.shape[-1]), rainfall_dim=int(rainfall_store.forecast_mmhr.shape[-1]), physics_dim=int(np.asarray(graph.actuator_physics).shape[1]), target_scale=target_scale, graph_payload=graph_payload, device=device)
    design = NG1TrainingDesign(seed=SEED)
    history = train_ng1_model(model, source_cache=base_online, fit_d2_names=selected["fit_d2"], fit_d3_names=selected["fit_d3"], normalization=normalization, graph=graph, device=device, target_scale_m3=target_scale, design=design)
    main_state = history.pop("main_state_dict")

    main_model = _model(state_dim=int(first_state.shape[-1]), rainfall_dim=int(rainfall_store.forecast_mmhr.shape[-1]), physics_dim=int(np.asarray(graph.actuator_physics).shape[1]), target_scale=target_scale, graph_payload=graph_payload, device=device)
    main_model.load_state_dict(main_state)
    main_evaluations = {
        "trainfit_d2": evaluate_direct_tfv_value_model(main_model, cache=base_online, names=selected["fit_d2"], normalization=normalization, graph=graph, device=device),
        "internal_holdout_d2": evaluate_direct_tfv_value_model(main_model, cache=base_online, names=selected["hold_d2"], normalization=normalization, graph=graph, device=device),
    }
    d2_checkpoint = out / D2_CHECKPOINT_FILENAME
    torch.save({"contract": NG1_CONTRACT, "checkpoint_stage": "D2_MAIN_ONLY", "development_only": True, "seed": SEED, "target_scale_m3": float(target_scale), "model_design": asdict(DirectTFVValueDesign()), "interaction_graph_sha256": graph_payload["sha256"], "model_state_dict": main_state}, d2_checkpoint)
    d2_report = {"contract": NG1_CONTRACT, "training_contract": NG1_TRAINING_CONTRACT, "development_only": True, "new_swmm_runs": 0, "checkpoint": str(d2_checkpoint), "checkpoint_sha256": _sha(d2_checkpoint), "training_source": "existing D2 TrainFit exact SWMM records only", "main_evaluations": main_evaluations, "d2_magnitude_strata": history["d2_magnitude_strata"], "interaction_graph_sha256": graph_payload["sha256"]}
    (out / D2_REPORT_FILENAME).write_text(json.dumps(_jsonable(d2_report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evaluations = {
        "trainfit_d2": evaluate_direct_tfv_value_model(model, cache=base_online, names=selected["fit_d2"], normalization=normalization, graph=graph, device=device),
        "trainfit_d3": evaluate_direct_tfv_value_model(model, cache=base_online, names=selected["fit_d3"], normalization=normalization, graph=graph, device=device),
        "internal_holdout_d2": evaluate_direct_tfv_value_model(model, cache=base_online, names=selected["hold_d2"], normalization=normalization, graph=graph, device=device),
        "internal_holdout_d3": evaluate_direct_tfv_value_model(model, cache=base_online, names=selected["hold_d3"], normalization=normalization, graph=graph, device=device),
        "d4_fit": evaluate_direct_tfv_value_model(model, cache=d4_fit_online, names=selected["d4_fit"], normalization=normalization, graph=graph, device=device),
        "d4_audit": evaluate_direct_tfv_value_model(model, cache=d4_audit_online, names=selected["d4_audit"], normalization=normalization, graph=graph, device=device),
    }
    lineage = {"graph_sha256": _sha(args.graph), "base_cache_sha256": _sha(args.cache_manifest), "d4_fit_cache_sha256": _sha(args.d4_fit_cache), "d4_audit_cache_sha256": _sha(args.d4_audit_cache), "causal_rainfall_sha256": _sha(args.causal_store), "causal_state_store_sha256": _sha(args.causal_state_store), "step1_model_semantic_sha256": str(state_store.step1_model_semantic_sha256), "sensor_layout_semantic_sha256": str(state_store.sensor_layout_semantic_sha256)}
    resources = {"wall_seconds": float(time.perf_counter() - started), "device": str(device), "cuda_available": bool(torch.cuda.is_available()), "peak_cuda_allocated_gb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0}
    checkpoint = out / CHECKPOINT_FILENAME
    checkpoint_payload = {"contract": NG1_CONTRACT, "training_contract": NG1_TRAINING_CONTRACT, "development_only": True, "seed": SEED, "target_scale_m3": float(target_scale), "state_dim": int(first_state.shape[-1]), "rainfall_dim": int(rainfall_store.forecast_mmhr.shape[-1]), "actuator_physics_dim": int(np.asarray(graph.actuator_physics).shape[1]), "model_design": asdict(DirectTFVValueDesign()), "normalization": _normalization_payload(normalization), "interaction_graph_sha256": graph_payload["sha256"], "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "history": history, "evaluations": evaluations, "lineage": lineage, "selected_group_counts": {key: len(value) for key, value in selected.items()}, "resources": resources}
    torch.save(checkpoint_payload, checkpoint)
    report = {"contract": NG1_CONTRACT, "training_contract": NG1_TRAINING_CONTRACT, "development_only": True, "publication_controller_changed": False, "validation_accessed": False, "final_accessed": False, "formal_used_for_tuning": False, "policy_lock_accessed": False, "new_swmm_runs": 0, "new_training_examples": 0, "new_policy_return_truth": 0, "seed": SEED, "primary_target": "authoritative exact SWMM delta TFV", "architecture": "V5 facility main effect plus complete 5886-pair process-aware interaction value", "d2_main_trainfit_only": True, "d2_facility_balanced": True, "d3_main_parameters_frozen": True, "interaction_graph_path": str(graph_path), "interaction_graph_sha256": graph_payload["sha256"], "interaction_graph_pair_count": 5886, "interaction_graph_label_independent": True, "main_parameter_sha256_before_d3": history["main_parameter_sha256_before_d3"], "main_parameter_sha256_after_d3": history["main_parameter_sha256_after_d3"], "selected_group_counts": {key: len(value) for key, value in selected.items()}, "canonical_group_counts": {"fit_d2": 112, "fit_d3": 112, "hold_d2": 32, "hold_d3": 32, "d4_fit": 33, "d4_audit": 15}, "rainfall_fit_groups": 14, "rainfall_holdout_groups": 4, "rainfall_group_overlap": 0, "target_scale_m3": float(target_scale), "history": history, "evaluations": evaluations, "lineage": lineage, "resources": resources, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "d2_main_checkpoint": str(d2_checkpoint), "d2_main_report": str(out / D2_REPORT_FILENAME), "runtime_promoted": False, "next_gate": "Development-only NG1 frontier comparison; no Step3/SWMM/Formal access"}
    (out / REPORT_FILENAME).write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
