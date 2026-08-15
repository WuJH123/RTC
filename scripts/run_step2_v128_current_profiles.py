"""Current Project7 V128 Step2 runner with smoke/dev/full profiles and stage resume.

This is the single current training orchestration.  It does not change the frozen scientific
model/time/action contract.  ``smoke`` and ``dev`` reduce Development-data coverage and
repetitions only; only ``full`` may emit the strict final V128 base checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from rtc.checkpoint_v128 import save_step2_v128
from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.development_profile_v128 import (
    V128_EXECUTION_PROFILE_CONTRACT,
    apply_profile_to_design,
    get_execution_profile,
    profile_groups,
)
from rtc.development_telemetry_v128 import JsonlTelemetry, TimedStage
from rtc.production_cli import _load_graph
from rtc.stage_checkpoint_v128 import load_stage_checkpoint_v128, save_stage_checkpoint_v128
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_differentiable_v128 import V128_STEP2_CONTRACT, build_v128_model_from_graph
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, derive_v127_input_normalization, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import configure_model_normalization_v127
from rtc.step2_train_v127_streaming import (
    V127StreamingMemoryDesign,
    derive_residual_scales_streaming_v127,
    evaluate_objective_groups_streaming_v127,
    train_truncated_rollout_stage_streaming_v127,
)
from rtc.step2_train_v128_exact import V128_OBJECTIVE_TRAINING_CONTRACT, train_objective_stage_streaming_v128
from rtc.step2_train_v128_hydraulic import V128_HYDRAULIC_TRAINING_CONTRACT, train_hydraulic_stage_streaming_v128
from rtc.v128_control_profile import V128_CONTROL_PROFILE_CONTRACT, build_v128_control_training_design, configure_v128_cuda_matmul_precision

CURRENT_RUN_CONTRACT = "PROJECT7_V128_CURRENT_SMOKE_DEV_FULL_STAGED_V1"
REPORT_FILENAME = "STEP2_V128_CURRENT_REPORT.json"
FINAL_CHECKPOINT_FILENAME = "step2_v128_control_base.pt"
_STAGE_ORDER = {"initialized": 0, "stage_a": 1, "stage_b0": 2, "objective": 3, "evaluation": 4}


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rain(cache: V60TrainCache, names: list[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def _finite_metrics(name: str, metrics: dict[str, float | str]) -> None:
    required = (
        "rank", "pairwise", "top1", "tfv_mae_m3", "selected_regret_m3",
        "hard_tfv_rank", "hard_tfv_informative_pairwise", "smooth_tfv_rank",
        "hard_tfv_delta_mae_m3", "smooth_tfv_delta_mae_m3",
    )
    bad = [key for key in required if key not in metrics or not math.isfinite(float(metrics[key]))]
    if bad:
        raise RuntimeError(f"{name}: current V128 evaluation invalid for {bad}")


def _cache_time_engine(path: str | Path) -> tuple[int, int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    value = (int(payload.get("model_step_seconds", -1)), int(payload.get("horizon_steps", -1)), str(payload.get("swmm_engine_version", "")).strip())
    if value[0] != 300 or value[1] != 72 or not value[2]:
        raise ValueError(f"cache {path} does not satisfy current 300-s/H72/engine lineage")
    return value


def _hardware(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {"device": str(device), "amp_enabled": False, "activation_checkpointing_enabled": False}
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        payload.update({"system_ram_total_gb": float(vm.total / 1024**3), "system_ram_available_gb_at_start": float(vm.available / 1024**3)})
    except Exception:
        payload["system_ram_probe"] = "psutil unavailable"
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        props = torch.cuda.get_device_properties(device)
        payload.update({"cuda_name": str(props.name), "cuda_total_gb": float(total / 1024**3), "cuda_free_gb_at_start": float(free / 1024**3), "cuda_allocator_backend": str(torch.cuda.get_allocator_backend())})
        if total < 7.0 * 1024**3:
            raise RuntimeError("current V128 profile expects an approximately 8-GB CUDA GPU")
        if free < 4.0 * 1024**3:
            raise RuntimeError("less than 4 GB CUDA memory is free; close other GPU jobs")
    return payload


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke", "dev", "full"), required=True,
                   help="Required explicit cost tier. smoke/dev cannot create a final checkpoint.")
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--hydraulic-branch-chunk", type=int, default=4)
    p.add_argument("--rollout-candidates-per-group", type=int, default=2)
    p.add_argument("--objective-candidate-chunk", type=int, default=2)
    p.add_argument("--evaluation-branch-chunk", type=int, default=4)
    p.add_argument("--residual-sample-rows", type=int, default=131072)
    p.add_argument("--resume-from", default="")
    p.add_argument("--stop-after-stage", choices=("stage_a", "stage_b0", "objective", "evaluation"), default="evaluation")
    p.add_argument("--profile-one-group", action="store_true",
                   help="Diagnostic-only: use one deterministic group from each source; forbidden with full.")
    p.add_argument("--torch-profiler", action="store_true",
                   help="Export a CPU/CUDA profiler trace; intended only with smoke/dev profile-one-group.")
    return p


def _should_run(completed: str, stage: str) -> bool:
    return _STAGE_ORDER[completed] < _STAGE_ORDER[stage]


def main() -> None:
    args = _parser().parse_args()
    profile = get_execution_profile(args.profile)
    if args.profile_one_group and profile.is_full:
        raise ValueError("--profile-one-group is development-only and forbidden with --profile full")
    if args.torch_profiler and profile.is_full:
        raise ValueError("torch profiler is forbidden on the expensive full scientific run")

    execution_profile = configure_v128_cuda_matmul_precision()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    hardware = _hardware(device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    telemetry = JsonlTelemetry(out / "TRAINING_TELEMETRY.jsonl", device=device)
    telemetry.emit("run_start", profile=profile.name, one_group=bool(args.profile_one_group))

    memory = V127StreamingMemoryDesign(
        hydraulic_branch_chunk=int(args.hydraulic_branch_chunk),
        rollout_candidates_per_group=int(args.rollout_candidates_per_group),
        objective_candidate_chunk=int(args.objective_candidate_chunk),
        evaluation_branch_chunk=int(args.evaluation_branch_chunk),
        residual_sample_rows=int(args.residual_sample_rows),
    )
    memory.validate()

    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("current V128 training requires exactly 109 actuators")
    cache_contracts = {
        "D2_D3": _cache_time_engine(args.cache_manifest),
        "D4_FIT": _cache_time_engine(args.d4_fit_cache),
        "D4_AUDIT": _cache_time_engine(args.d4_audit_cache),
    }
    engines = {value[2] for value in cache_contracts.values()}
    if len(engines) != 1:
        raise ValueError(f"current caches mix SWMM engines: {cache_contracts}")
    swmm_engine = next(iter(engines))

    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rain_store.validate(); state_store.validate()
    if rain_store.forecast_mmhr.shape[1] != 72 or rain_store.forecast_mmhr.shape[2] != len(graph.node_ids):
        raise ValueError("causal rainfall store must match current H72 graph")

    fit, holdout = deterministic_rainfall_split_v60(base, names=sorted(base.names("D2") + base.targeted_d3_names()), holdout_fraction=0.20)
    fit_d2 = [x for x in fit if x.startswith("D2::")]
    fit_d3 = [x for x in fit if x.startswith("D3::")]
    hold_d2 = [x for x in holdout if x.startswith("D2::")]
    hold_d3 = [x for x in holdout if x.startswith("D3::")]
    if tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3))) != (112, 112, 32, 32):
        raise ValueError("canonical D2/D3 split differs from 112/112/32/32")
    d4_fit_names = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit_raw.names(D4_SOURCE_KIND)
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("current V128 requires canonical D4 FIT/AUDIT census 33/15")
    if _rain(d4_fit_raw, d4_fit_names) & _rain(d4_audit_raw, d4_audit_names):
        raise ValueError("D4 FIT/AUDIT rainfall leakage")

    selected = profile_groups(profile, fit_d2=fit_d2, fit_d3=fit_d3, hold_d2=hold_d2, hold_d3=hold_d3,
                              d4_fit=d4_fit_names, d4_audit=d4_audit_names, one_group=bool(args.profile_one_group))
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    d4_fit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain_store), state_store)
    d4_audit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain_store), state_store)

    normalization = derive_v127_input_normalization(base_cache=base, causal_rainfall=rain_store, causal_state=state_store, fit_names=selected["fit_d2"] + selected["fit_d3"])
    sample_rows = int(args.residual_sample_rows)
    if not profile.is_full:
        sample_rows = min(sample_rows, 8192 if profile.name == "smoke" else 32768)
    if args.profile_one_group:
        sample_rows = min(sample_rows, 4096)
    state_scale, flow_scale, scale_telemetry = derive_residual_scales_streaming_v127(
        ((base, selected["fit_d2"] + selected["fit_d3"]), (d4_fit_raw, selected["d4_fit"])), sample_rows=sample_rows)
    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    model = build_v128_model_from_graph(graph, state_dim=int(first_state.shape[-1]), rainfall_dim=int(rain_store.forecast_mmhr.shape[-1]),
                                        delta_state_scale=state_scale, delta_flow_scale=flow_scale)
    configure_model_normalization_v127(model, normalization=normalization, graph=graph, state_delta_scale=state_scale, flow_delta_scale=flow_scale)

    design = build_v128_control_training_design(
        hydraulic_branch_chunk=memory.hydraulic_branch_chunk,
        rollout_candidates_per_group=memory.rollout_candidates_per_group,
        objective_candidate_chunk=memory.objective_candidate_chunk,
        evaluation_branch_chunk=memory.evaluation_branch_chunk,
    )
    design = apply_profile_to_design(design, profile)
    source_caches = {"D2": base_online, "D3": base_online, "D4": d4_fit_online}
    source_groups = {"D2": selected["fit_d2"], "D3": selected["fit_d3"], "D4": selected["d4_fit"]}

    lineage = {
        "graph_sha256": _sha(args.graph), "base_cache_sha256": _sha(args.cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache), "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "causal_rainfall_sha256": _sha(args.causal_store), "causal_state_store_sha256": _sha(args.causal_state_store),
        "causal_state_step1_file_sha256": str(state_store.step1_sha256), "causal_state_sensor_file_sha256": str(state_store.sensor_sha256),
        "causal_state_graph_file_sha256": str(state_store.graph_sha256), "causal_state_step1_model_semantic_sha256": str(state_store.step1_model_semantic_sha256),
        "causal_state_sensor_layout_semantic_sha256": str(state_store.sensor_layout_semantic_sha256), "causal_rainfall_forecast_contract": str(rain_store.forecast_contract),
        "swmm_engine_version": swmm_engine,
    }
    design_dict = asdict(design)
    history: dict[str, object] = {"hydraulic": [], "rollout": [], "objective": []}
    completed = "initialized"
    if args.resume_from:
        payload = load_stage_checkpoint_v128(args.resume_from, model=model, expected_profile=profile.name, graph_path=args.graph,
                                             expected_lineage=lineage, expected_training_design=design_dict)
        completed = str(payload["completed_stage"])
        history.update(dict(payload.get("history") or {}))
        telemetry.emit("resume_loaded", completed_stage=completed, path=str(Path(args.resume_from).resolve()))

    profiler = nullcontext()
    prof = None
    if args.torch_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda": activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=True, with_stack=False)
        profiler = prof

    with profiler:
        if _should_run(completed, "stage_a"):
            with TimedStage(telemetry, "stage_a"):
                history["hydraulic"] = train_hydraulic_stage_streaming_v128(model, source_caches=source_caches, source_groups=source_groups,
                    normalization=normalization, graph=graph, device=device, depth_index=args.depth_index, flood_rate_index=args.flood_rate_index, design=design)
            completed = "stage_a"
            save_stage_checkpoint_v128(out / "stage_a.pt", model=model, completed_stage=completed, profile=profile.name,
                graph_path=args.graph, lineage=lineage, training_design=design_dict, history=history)
            if args.stop_after_stage == completed: return

        if _should_run(completed, "stage_b0"):
            with TimedStage(telemetry, "stage_b0"):
                history["rollout"] = train_truncated_rollout_stage_streaming_v127(model, source_caches=source_caches, source_groups=source_groups,
                    normalization=normalization, graph=graph, device=device, depth_index=args.depth_index, flood_rate_index=args.flood_rate_index, design=design)
            completed = "stage_b0"
            save_stage_checkpoint_v128(out / "stage_b0.pt", model=model, completed_stage=completed, profile=profile.name,
                graph_path=args.graph, lineage=lineage, training_design=design_dict, history=history)
            if args.stop_after_stage == completed: return

        if _should_run(completed, "objective"):
            with TimedStage(telemetry, "objective"):
                history["objective"] = train_objective_stage_streaming_v128(model, source_caches=source_caches, source_groups=source_groups,
                    normalization=normalization, graph=graph, device=device, flood_rate_index=args.flood_rate_index, design=design)
            completed = "objective"
            save_stage_checkpoint_v128(out / "stage_objective.pt", model=model, completed_stage=completed, profile=profile.name,
                graph_path=args.graph, lineage=lineage, training_design=design_dict, history=history)
            if args.stop_after_stage == completed: return

    if prof is not None:
        prof.export_chrome_trace(str(out / "TORCH_PROFILER_TRACE.json"))

    specs = {
        "trainfit_d2": (base_online, selected["fit_d2"]), "trainfit_d3": (base_online, selected["fit_d3"]),
        "internal_holdout_d2": (base_online, selected["hold_d2"]), "internal_holdout_d3": (base_online, selected["hold_d3"]),
        "d4_fit": (d4_fit_online, selected["d4_fit"]), "d4_audit": (d4_audit_online, selected["d4_audit"]),
    }
    evaluations: dict[str, dict[str, float | str]] = {}
    with TimedStage(telemetry, "evaluation"):
        for key, (cache, names) in specs.items():
            metrics = evaluate_objective_groups_streaming_v127(model, cache=cache, names=names, normalization=normalization, graph=graph,
                device=device, flood_rate_index=args.flood_rate_index, design=design)
            _finite_metrics(key, metrics); evaluations[key] = metrics
    completed = "evaluation"

    report = {
        "contract": CURRENT_RUN_CONTRACT,
        "execution_profile_contract": V128_EXECUTION_PROFILE_CONTRACT,
        "profile": profile.name,
        "scientific_claim_allowed": bool(profile.scientific_claim_allowed),
        "final_checkpoint_allowed": bool(profile.final_checkpoint_allowed),
        "policy_lock_allowed": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False,
        "step2_contract": V128_STEP2_CONTRACT, "hydraulic_training_contract": V128_HYDRAULIC_TRAINING_CONTRACT,
        "objective_training_contract": V128_OBJECTIVE_TRAINING_CONTRACT, "control_profile_contract": V128_CONTROL_PROFILE_CONTRACT,
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "hardware": {**hardware, **execution_profile}, "memory_execution": asdict(memory), "training_design": design_dict,
        "selected_groups": selected, "selected_group_counts": {key: len(value) for key, value in selected.items()},
        "residual_scale_sampling": scale_telemetry, "history": history, "evaluations": evaluations, "lineage": lineage,
        "boundary": {"new_swmm": False, "d4_audit_used_for_training": False, "internal_holdout_used_for_training": False,
                     "d5_used_for_training": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False},
    }
    report_path = out / REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    if profile.final_checkpoint_allowed:
        checkpoint = save_step2_v128(out / FINAL_CHECKPOINT_FILENAME, model=model, graph=graph, input_normalization=normalization, training_report=report, lineage=lineage)
        report["checkpoint"] = str(checkpoint.resolve()); report["checkpoint_sha256"] = _sha(checkpoint)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    else:
        report["checkpoint"] = None
        report["checkpoint_note"] = "smoke/dev artifacts are NONFINAL and cannot enter D5/runtime/Policy Lock"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    telemetry.emit("run_complete", profile=profile.name, strict_final_checkpoint=bool(profile.final_checkpoint_allowed))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
