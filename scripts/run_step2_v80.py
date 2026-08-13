"""Canonical Train-only Project7 Step2 V8.0 direct hydraulic-effect development runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import HydraulicResponseSurrogateV70
from rtc.step2_control_response_v80 import DirectHydraulicEffectSurrogateV80, prepare_static_v80
from rtc.step2_hydraulic_eval_v80 import (
    calibrate_onset_threshold_v80,
    evaluate_hydraulic_effect_v80,
    evaluate_onset_threshold_v80,
)
from rtc.step2_hydraulic_objective_v80 import derive_onset_sqrt_positive_weight_v80
from rtc.step2_optimization_v80 import train_hydraulic_effect_v80
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v70_contract import V70_CONTRACT
from rtc.step2_v80_contract import (
    DirectHydraulicEffectLossContractV80,
    V80_CONTRACT,
    V80_HYDRAULIC_CONTRACT,
)


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _load_checkpoint(path: str | Path, expected_kind: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} is not a dictionary")
    if str(payload.get("contract")) != V70_CONTRACT:
        raise ValueError(f"checkpoint {path} is not a V7 canonical checkpoint")
    if str(payload.get("kind")) != expected_kind:
        raise ValueError(f"checkpoint {path}: kind {payload.get('kind')} != {expected_kind}")
    if bool(payload.get("production_compatible", True)):
        raise ValueError("development checkpoint unexpectedly claims production compatibility")
    if "state_dict" not in payload:
        raise ValueError("checkpoint has no state_dict")
    return payload


def _build_reference_hydraulic(graph: GraphSchema, cache: V60TrainCache, fit_names, scales):
    entry = cache.entry(fit_names[0])
    state_dim = int(entry.arrays["initial_state"].shape[-1])
    rainfall_dim = int(entry.arrays["rainfall"].shape[-1])
    if state_dim != 6:
        raise ValueError("V8 requires the frozen six-state hydraulic contract")
    prepared = prepare_static_v60(graph, "cpu")
    return HydraulicResponseSurrogateV70(
        state_scale=scales.base.state_scale,
        flow_scale=scales.base.flow_scale,
        horizon_contract=MultiResolutionHorizonV60(),
        state_dim=state_dim,
        rainfall_dim=rainfall_dim,
        node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=64,
        latent_dim=32,
        temporal_dim=12,
    )


def _preflight(model, cache, name, normalization, graph, device):
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    prepared = prepare_static_v80(graph, target)
    batch = cache.batch(name, normalization, target)
    reference = batch.reference_settings[:, None]
    output = model(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        reference,
        batch.previous_actuator_flow,
        prepared,
    )
    if not torch.equal(output.delta_states_physical, torch.zeros_like(output.delta_states_physical)):
        raise RuntimeError("V8 exact-zero state-effect contract failed")
    if not torch.equal(output.delta_flows_physical, torch.zeros_like(output.delta_flows_physical)):
        raise RuntimeError("V8 exact-zero flow-effect contract failed")
    candidate = batch.candidate_settings[:, :1].clone().requires_grad_(True)
    output = model(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        candidate,
        batch.previous_actuator_flow,
        prepared,
    )
    objective = (
        output.delta_states_physical[..., 0].abs().sum()
        + output.delta_states_physical[..., 2].abs().sum()
        + output.delta_states_physical[..., 3].abs().sum()
        + output.delta_flows_physical.abs().sum()
    )
    gradient = torch.autograd.grad(objective, candidate)[0]
    reference_trainable = any(p.requires_grad for p in model.reference_model.parameters())
    if reference_trainable:
        raise RuntimeError("V8 frozen V7 reference model unexpectedly has trainable parameters")
    if not torch.isfinite(gradient).all() or torch.count_nonzero(gradient).item() == 0:
        raise RuntimeError("V8 direct hydraulic action gradient is invalid")
    return {
        "state_exact_zero": True,
        "flow_exact_zero": True,
        "action_gradient_finite": True,
        "action_gradient_nonzero": True,
        "action_gradient_median_abs": float(gradient.abs().median()),
        "action_gradient_max_abs": float(gradient.abs().max()),
        "reference_model_frozen": True,
        "graph_blocks": len(model.graph_blocks),
    }


def _report_markdown(payload: dict[str, Any]) -> str:
    h = payload["hydraulic_effect_metrics"]["holdout_d3"]
    onset = payload["onset_metrics"]["holdout_d3"]
    return "\n".join(
        [
            "# PROJECT7 STEP2 V8.0 DIRECT HYDRAULIC EFFECT",
            "",
            "This Markdown is a summary of the authoritative JSON report.",
            "",
            "## Boundary",
            "",
            f"- Git HEAD: `{payload['lineage']['git_head']}`",
            "- SWMM: not run; existing D2/D3 reused; Validation/Final/Formal not accessed.",
            "- V7 Value checkpoint is preserved unchanged; V7 hydraulic checkpoint is frozen as reference only.",
            "",
            "## Holdout D3 direct effect",
            "",
            f"- Delta depth response ratio: `{h['delta_depth_m_response_ratio']:.4f}`; skill vs zero: `{h['delta_depth_m_skill_vs_zero']:.4f}`.",
            f"- Delta flood response ratio: `{h['delta_flood_m3s_response_ratio']:.4f}`; skill vs zero: `{h['delta_flood_m3s_skill_vs_zero']:.4f}`.",
            f"- Delta storage response ratio: `{h['delta_storage_m3_response_ratio']:.4f}`; skill vs zero: `{h['delta_storage_m3_skill_vs_zero']:.4f}`.",
            f"- Delta managed-flow response ratio: `{h['delta_managed_flow_m3s_response_ratio']:.4f}`; skill vs zero: `{h['delta_managed_flow_m3s_skill_vs_zero']:.4f}`.",
            "",
            "## Onset",
            "",
            f"- TrainFit-calibrated logit threshold: `{payload['onset_threshold']['threshold_logit']:.4f}`.",
            f"- Holdout F1: `{onset['f1']:.4f}`; precision: `{onset['precision']:.4f}`; recall: `{onset['recall']:.4f}`; precision lift: `{onset['precision_lift']:.2f}x`.",
            "",
            "## Decision",
            "",
            f"- Direct hydraulic effect supported: **{payload['direct_hydraulic_effect_supported']}**.",
            f"- V8 architecture-lock candidate: **{payload['architecture_lock_candidate']}** (development only).",
            "- Ready for Formal/production: **NO**.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project7 Step2 V8 direct hydraulic effect development")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--v70-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError(f"V8 frozen actuator count requires 109, got {len(graph.actuator_ids)}")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V8 refuses legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    selected = sorted(d2 + d3)
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V8 split must contain D2 and D3 on both sides")
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V8 split leakage detected")

    cache_lineage = validate_v60_cache_lineage(args.cache_manifest)
    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    basis = build_control_basis_v60(graph)
    value_checkpoint = _load_checkpoint(args.v70_value_checkpoint, "control_value")
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    v70_report = json.loads(Path(args.v70_report).read_text(encoding="utf-8"))
    if bool(v70_report.get("catastrophic_value_collapse", True)):
        raise ValueError("V8 requires the successful non-collapsed V7 Value checkpoint")

    reference = _build_reference_hydraulic(graph, cache, fit, scales)
    reference.load_state_dict(hydraulic_checkpoint["state_dict"], strict=True)
    contract = DirectHydraulicEffectLossContractV80()
    prepared_cpu = prepare_static_v60(graph, "cpu")
    model = DirectHydraulicEffectSurrogateV80(
        reference_model=reference,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=scales.state_delta_scale,
        flow_delta_scale=scales.flow_delta_scale,
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_count=len(graph.actuator_ids),
        contract=contract,
    )
    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    preflight = _preflight(model, cache, fit_d2[0], normalization, graph, str(target))
    onset_positive_weight = derive_onset_sqrt_positive_weight_v80(
        cache,
        fit,
        normalization,
        contract=contract,
    )

    training_started = time.perf_counter()
    history = train_hydraulic_effect_v80(
        model,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=normalization,
        scales=scales,
        onset_positive_weight=onset_positive_weight,
        graph=graph,
        device=str(target),
        seed=args.seed,
        contract=contract,
    )
    training_seconds = time.perf_counter() - training_started

    evaluation_started = time.perf_counter()
    prepared = prepare_static_v80(graph, target)
    effect_metrics = {
        "fit_d2": evaluate_hydraulic_effect_v80(model, cache, fit_d2, normalization, prepared, scales, device=target, contract=contract),
        "fit_d3": evaluate_hydraulic_effect_v80(model, cache, fit_d3, normalization, prepared, scales, device=target, contract=contract),
        "holdout_d2": evaluate_hydraulic_effect_v80(model, cache, holdout_d2, normalization, prepared, scales, device=target, contract=contract),
        "holdout_d3": evaluate_hydraulic_effect_v80(model, cache, holdout_d3, normalization, prepared, scales, device=target, contract=contract),
    }
    onset_threshold = calibrate_onset_threshold_v80(
        model, cache, fit_d3, normalization, prepared, device=target, contract=contract
    )
    onset_metrics = {
        "fit_d3": evaluate_onset_threshold_v80(
            model, cache, fit_d3, normalization, prepared,
            threshold_logit=onset_threshold["threshold_logit"], device=target, contract=contract
        ),
        "holdout_d3": evaluate_onset_threshold_v80(
            model, cache, holdout_d3, normalization, prepared,
            threshold_logit=onset_threshold["threshold_logit"], device=target, contract=contract
        ),
    }
    evaluation_seconds = time.perf_counter() - evaluation_started

    holdout_primary = effect_metrics["holdout_d3"]
    response_keys = (
        "delta_depth_m_response_ratio",
        "delta_flood_m3s_response_ratio",
        "delta_storage_m3_response_ratio",
    )
    response_not_collapsed = all(float(holdout_primary[key]) >= 0.10 for key in response_keys)
    direct_supported = bool(
        holdout_primary["all_primary_effect_skills_positive"] and response_not_collapsed
    )
    value_supported = bool(
        float(v70_report["value_metrics"]["holdout_d3"]["spread_ratio"]) >= 0.10
        and not bool(v70_report.get("catastrophic_value_collapse", True))
    )
    architecture_lock_candidate = bool(value_supported and direct_supported)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lineage = {
        "git_head": _git_head(),
        "graph_sha256": _sha256(args.graph),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        "v70_value_checkpoint_sha256": _sha256(args.v70_value_checkpoint),
        "v70_hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
        "v70_report_sha256": _sha256(args.v70_report),
        "basis_sha256_from_cache_lineage": str(cache_lineage["v60_control_basis_sha256"]),
        "design_sha256_from_cache_lineage": str(cache_lineage["v60_design_contract_sha256"]),
    }
    checkpoint_payload = {
        "contract": V80_CONTRACT,
        "kind": "direct_hydraulic_effect",
        "state_dict": model.state_dict(),
        "lineage": lineage,
        "basis_manifest": basis_manifest_v60(basis),
        "seed": int(args.seed),
        "production_compatible": False,
    }
    checkpoint_path = out / "step2_v80_direct_hydraulic_effect_development.pt"
    torch.save(checkpoint_payload, checkpoint_path)

    total_seconds = time.perf_counter() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    report = {
        "contract": V80_CONTRACT,
        "hydraulic_contract": V80_HYDRAULIC_CONTRACT,
        "boundary": {
            "swmm_run": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "production_wiring_modified": False,
        },
        "lineage": lineage,
        "data": {
            "d2_groups": len(d2),
            "d3_groups": len(d3),
            "fit_events": len(fit_events),
            "holdout_events": len(holdout_events),
            "event_overlap": 0,
            "rainfall_overlap": 0,
        },
        "value_v70_preserved": True,
        "value_v70_metrics": v70_report.get("value_metrics", {}),
        "preflight": preflight,
        "loss_contract": contract.__dict__,
        "onset_positive_weight": float(onset_positive_weight),
        "training_history": history,
        "hydraulic_effect_metrics": effect_metrics,
        "onset_threshold": onset_threshold,
        "onset_metrics": onset_metrics,
        "direct_hydraulic_effect_supported": direct_supported,
        "value_supported": value_supported,
        "architecture_lock_candidate": architecture_lock_candidate,
        "ready_for_formal": False,
        "ready_for_production": False,
        "performance": {
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": total_seconds,
            "torch_cuda_peak_allocated_bytes": peak_vram,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
    }
    _write_json(out / "STEP2_V80_DEVELOPMENT_REPORT.json", report)
    (out / "STEP2_V80_DEVELOPMENT_REPORT.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "PROJECT7_STEP2_V80": {
            "direct_hydraulic_effect_supported": direct_supported,
            "architecture_lock_candidate": architecture_lock_candidate,
            "holdout_d3": effect_metrics["holdout_d3"],
            "onset_holdout_d3": onset_metrics["holdout_d3"],
            "performance": report["performance"],
            "report": str(out / "STEP2_V80_DEVELOPMENT_REPORT.json"),
        }
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
