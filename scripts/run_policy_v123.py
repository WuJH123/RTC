"""Run the V123 causal TFV/PFV knowledge-anchored finite policy.

The validated V122 execution shell is reused for clock/readback/score==execute. V123
uses a sparse-state Auto-RBC engineering anchor when learned Value evidence is not
strong enough to justify an override. TFV Value may be the frozen causal V70 model or
the interaction-aware V124 development model after external acceptance; PFV remains a
separate soft diagnostic/value model.

Continuous gradient search remains disabled until the authoritative gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.controller_v123 import V123TorchMPCController
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import (
    _controller_config,
    _controls_disabled_runtime,
    _load_graph,
    _load_lines,
    _load_step1,
)
from rtc.runtime_controller_guard import ContinuityGuardController
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70
from rtc.step2_control_value_v124 import ControlValueSurrogateV124, V124_VALUE_CONTRACT
from rtc.step2_policy_v120 import RuntimeNormalizationV120
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_causal_rainfall_v123 import (
    derive_causal_input_normalization_v123,
    load_causal_forecast_store_v123,
)
from rtc.step2_control_value_v123 import DualVolumeValueV123
from rtc.step2_policy_v123 import FirstMoveTFVPFVPolicyV123, V123_POLICY_CONTRACT
from rtc.step3_knowledge_seeds_v123 import V123_SPARSE_RBC_ANCHOR_CONTRACT
from rtc.step3_objective_v123 import TFVPFVObjectiveV123


def _causal_normalization(
    cache: V60TrainCache, store, fit_names: list[str]
) -> RuntimeNormalizationV120:
    norm = derive_causal_input_normalization_v123(cache, store, fit_names)
    result = RuntimeNormalizationV120(
        norm.state_mean,
        norm.state_std,
        norm.rainfall_mean,
        norm.rainfall_std,
        norm.flow_mean,
        norm.flow_std,
    )
    result.validate()
    return result


def _tfv_scale_from_report(payload: dict) -> float:
    if "target_scales" in payload:
        return float(payload["target_scales"]["direct_tfv_scale_m3"])
    if "target_scale_tfv_m3" in payload:
        return float(payload["target_scale_tfv_m3"])
    raise ValueError("V123 runtime TFV report lacks direct TFV target scale")


def _v70_model(*, first, prepared, graph, basis, scale: float) -> ControlValueSurrogateV70:
    return ControlValueSurrogateV70(
        state_dim=int(first["initial_state"].shape[-1]),
        rainfall_dim=int(first["rainfall"].shape[-1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=scale,
        hidden_dim=96,
        actuator_embedding_dim=16,
    )


def _load_tfv_model(
    *,
    checkpoint_path: str,
    first,
    prepared,
    graph,
    basis,
    scale: float,
    device: torch.device,
):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("contract") == V124_VALUE_CONTRACT:
        model = ControlValueSurrogateV124(
            state_dim=int(first["initial_state"].shape[-1]),
            rainfall_dim=int(first["rainfall"].shape[-1]),
            physics_dim=int(prepared.actuator_physics.shape[1]),
            actuator_count=len(graph.actuator_ids),
            temporal_basis=basis.temporal_basis,
            control_block_steps=basis.horizon.control_block_steps,
            tfv_scale_m3=scale,
            hidden_dim=int(payload.get("hidden_dim", 96)),
            actuator_embedding_dim=16,
            attention_heads=int(payload.get("attention_heads", 4)),
        )
        architecture = "V124_INTERACTION_AWARE"
    elif payload.get("arm") == "B_CAUSAL":
        model = _v70_model(
            first=first,
            prepared=prepared,
            graph=graph,
            basis=basis,
            scale=scale,
        )
        architecture = "V70_CAUSAL"
    else:
        raise ValueError("V123 runtime TFV checkpoint is not an accepted causal V70/V124 artifact")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, architecture


def _load_policy(
    *,
    graph,
    cache_manifest: str,
    causal_store_path: str,
    tfv_checkpoint: str,
    pfv_checkpoint: str,
    objective_report: str,
    calibration_report: str,
    tfv_report: str,
    pfv_report: str,
    device: torch.device,
):
    cache = V60TrainCache(cache_manifest)
    store = load_causal_forecast_store_v123(causal_store_path)
    names = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, _ = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    if len(fit_d2) != 112:
        raise ValueError(f"V123 runtime requires 112 TrainFit D2 groups, got {len(fit_d2)}")
    normalization = _causal_normalization(cache, store, fit)
    basis = build_control_basis_v60(graph)
    prepared = prepare_static_v60(graph, device)
    first = cache.entry(fit_d2[0]).arrays
    tfv_report_raw = json.loads(Path(tfv_report).read_text(encoding="utf-8"))
    pfv_report_raw = json.loads(Path(pfv_report).read_text(encoding="utf-8"))
    tfv_scale = _tfv_scale_from_report(tfv_report_raw)
    pfv_scale = float(pfv_report_raw["target_scale_pfv_m3"])

    tfv_model, tfv_architecture = _load_tfv_model(
        checkpoint_path=tfv_checkpoint,
        first=first,
        prepared=prepared,
        graph=graph,
        basis=basis,
        scale=tfv_scale,
        device=device,
    )
    pfv_payload = torch.load(pfv_checkpoint, map_location="cpu", weights_only=False)
    if pfv_payload.get("target") != "PFV":
        raise ValueError("V123 runtime PFV checkpoint has unexpected target lineage")
    pfv_model = _v70_model(
        first=first,
        prepared=prepared,
        graph=graph,
        basis=basis,
        scale=pfv_scale,
    )
    pfv_model.load_state_dict(pfv_payload["state_dict"], strict=True)
    pfv_model.to(device).float().eval()
    for parameter in pfv_model.parameters():
        parameter.requires_grad_(False)

    objective_raw = json.loads(Path(objective_report).read_text(encoding="utf-8"))["objective"]
    calibration = json.loads(Path(calibration_report).read_text(encoding="utf-8"))["calibration"]
    if "tfv_false_benefit_margin_m3" not in calibration:
        raise ValueError("V123 calibration lacks TFV false-benefit margin")
    if "pfv_false_safety_margin_m3" not in calibration:
        raise ValueError("V123 calibration lacks PFV false-safety margin")
    objective = TFVPFVObjectiveV123(
        pfv_soft_margin_m3=float(objective_raw["pfv_soft_margin_m3"]),
        pfv_scale_m3=float(objective_raw["pfv_scale_m3"]),
        tfv_scale_m3=float(objective_raw["tfv_scale_m3"]),
        pfv_penalty_weight=float(objective_raw["pfv_penalty_weight"]),
        pfv_model_error_margin_m3=float(calibration["pfv_false_safety_margin_m3"]),
    )
    dual = DualVolumeValueV123(tfv_model=tfv_model, pfv_model=pfv_model).to(device).eval()
    false_benefit = float(calibration["tfv_false_benefit_margin_m3"])
    false_safety = float(calibration["pfv_false_safety_margin_m3"])
    return FirstMoveTFVPFVPolicyV123(
        model=dual,
        basis=basis,
        prepared=prepared,
        normalization=normalization,
        objective=objective,
        false_benefit_margin_m3=false_benefit,
        graph=graph,
        use_sparse_rbc_anchor=True,
        knowledge_anchor_fallback=True,
    ), {
        "fit_d2_groups": len(fit_d2),
        "causal_store_sha256": hashlib.sha256(Path(causal_store_path).read_bytes()).hexdigest(),
        "tfv_checkpoint_sha256": hashlib.sha256(Path(tfv_checkpoint).read_bytes()).hexdigest(),
        "pfv_checkpoint_sha256": hashlib.sha256(Path(pfv_checkpoint).read_bytes()).hexdigest(),
        "tfv_architecture": tfv_architecture,
        "tfv_scale_m3": tfv_scale,
        "pfv_scale_m3": pfv_scale,
        "tfv_false_benefit_margin_m3": false_benefit,
        "pfv_false_safety_margin_m3": false_safety,
        "sparse_state_rbc_anchor": True,
        "sparse_state_rbc_anchor_contract": V123_SPARSE_RBC_ANCHOR_CONTRACT,
        "knowledge_anchor_fallback": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 causal knowledge-anchored development runner")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step1", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--tfv-checkpoint", required=True)
    parser.add_argument("--pfv-checkpoint", required=True)
    parser.add_argument("--tfv-report", required=True)
    parser.add_argument("--pfv-report", required=True)
    parser.add_argument("--objective-report", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("v120_contract") != "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1":
        raise ValueError("V123 development runner requires the frozen V120 timing/config contract")
    forecast_cfg = cfg.get("forecast", {})
    if tuple(float(x) for x in forecast_cfg.get("scenario_multipliers", [])) != (1.0,):
        raise ValueError("V123 runtime requires the single central rainfall forecast")
    if abs(float(forecast_cfg.get("decay_per_step", -1.0)) - 0.92) > 1.0e-12:
        raise ValueError("V123 runtime rainfall decay differs from the frozen contract")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    step1 = _load_step1(args.step1, device)
    policy, lineage = _load_policy(
        graph=graph,
        cache_manifest=args.cache_manifest,
        causal_store_path=args.causal_store,
        tfv_checkpoint=args.tfv_checkpoint,
        pfv_checkpoint=args.pfv_checkpoint,
        objective_report=args.objective_report,
        calibration_report=args.calibration_report,
        tfv_report=args.tfv_report,
        pfv_report=args.pfv_report,
        device=device,
    )
    controller_cfg = cfg["controller"]
    controller = V123TorchMPCController(
        step1=step1,
        mpc=policy,
        graph=graph,
        sensor_nodes=sensors,
        forecast=PersistenceDecayForecast(
            decay_per_step=0.92,
            scenario_multipliers=(1.0,),
            history_steps_for_level=1,
        ),
        config=_controller_config(controller_cfg, control_block_steps=2),
        device=device,
    )
    controller = ContinuityGuardController(
        controller, max_delta_per_update=0.5, allow_projection=False
    )
    source_inp = Path(args.inp)
    runtime_inp = _controls_disabled_runtime(
        source_inp=source_inp,
        cache_dir=Path(args.out_dir) / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=args.out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=60,
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=bool(cfg.get("exact_global_peak", False)),
    )
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    metadata.update(
        {
            "v123_policy_contract": V123_POLICY_CONTRACT,
            "v123_runtime_causal_rainfall": True,
            "future_realized_rainfall_used_as_model_input": False,
            "continuous_gradient_search": False,
            "score_only_executable_sequences": True,
            "first_move_bound_to_current_and_target_readback": True,
            "knowledge_data_fusion": True,
            "knowledge_anchor_default_when_value_uncertain": True,
            "v123_lineage": lineage,
        }
    )
    Path(result.metadata_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "strategy": "proposed_v123",
                "metadata_path": result.metadata_path,
                "decision_path": result.decision_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
                "flow_routing_error_pct": result.flow_routing_error_pct,
                "lineage": lineage,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
