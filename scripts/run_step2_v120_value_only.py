"""Train the final simplified Project7 Step2: direct joint-action Delta-TFV only.

No nodewise Hydraulic model is trained. Existing authoritative D2 + targeted D3-v2
are reused; the internal rainfall/event holdout is read once after fitting for a
control-oriented gate. Validation/Final/SWMM are not accessed by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70
from rtc.step2_optimization_v70 import train_value_event_balanced_v70
from rtc.step2_policy_v120 import RuntimeNormalizationV120, candidate_coefficients_v120
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import (
    derive_target_scales_v70,
    evaluate_value_strata_v70,
    evaluate_value_v70,
)
from rtc.step2_v120_contract import (
    Step2V120Contract,
    V120_BUNDLE_CONTRACT,
    V120_CONTRACT,
)
from rtc.step2_v70_contract import DirectValueLossContractV70, V70_CONTRACT


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


def _cache_hashes(path: str | Path) -> dict[str, str]:
    lineage = validate_v60_cache_lineage(path)
    return {
        "basis_sha256_from_cache_lineage": str(lineage["v60_control_basis_sha256"]),
        "design_sha256_from_cache_lineage": str(lineage["v60_design_contract_sha256"]),
    }


def _value_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required = {
        "spearman": 0.35,
        "pairwise_accuracy": 0.60,
        "sign_accuracy": 0.60,
        "top1_accuracy": 0.25,
    }
    for key, floor in required.items():
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value) or value < floor:
            reasons.append(f"{key}={value} < {floor}")
    for key in ("spread_ratio", "response_ratio"):
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value) or not 0.25 <= value <= 2.50:
            reasons.append(f"{key}={value} outside [0.25,2.50]")
    regret = float(metrics.get("mean_regret_m3", float("nan")))
    if not np.isfinite(regret):
        reasons.append("mean_regret_m3 non-finite")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Project7 V120 TFV-value-only Step2")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--swmm-engine-version", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    contract = Step2V120Contract(seed=int(args.seed))
    contract.validate()
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != contract.actuator_count:
        raise ValueError("V120 graph violates frozen actuator count")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V120 rejects legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    if not d2 or not d3:
        raise ValueError("V120 requires existing D2 plus targeted D3-v2")
    selected = sorted(d2 + d3)
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [x for x in fit if x.startswith("D2::")]
    fit_d3 = [x for x in fit if x.startswith("D3::")]
    holdout_d2 = [x for x in holdout if x.startswith("D2::")]
    holdout_d3 = [x for x in holdout if x.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V120 split must contain D2/D3 on both sides")
    fit_events = sorted({cache.entry(x).event_id for x in fit})
    holdout_events = sorted({cache.entry(x).event_id for x in holdout})
    fit_rain = sorted({cache.entry(x).rainfall_group for x in fit})
    holdout_rain = sorted({cache.entry(x).rainfall_group for x in holdout})
    if set(fit_events) & set(holdout_events) or set(fit_rain) & set(holdout_rain):
        raise ValueError("V120 event/rainfall leakage")

    norm = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    basis = build_control_basis_v60(graph)
    entry = cache.entry(fit[0])
    state_dim = int(entry.arrays["initial_state"].shape[-1])
    rainfall_dim = int(entry.arrays["rainfall"].shape[-1])
    if state_dim != 6:
        raise ValueError("V120 requires frozen six-state Step1 interface")
    prepared_cpu = prepare_static_v60(graph, "cpu")
    model = ControlValueSurrogateV70(
        state_dim=state_dim,
        rainfall_dim=rainfall_dim,
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(scales.direct_tfv_scale_m3),
        hidden_dim=96,
        actuator_embedding_dim=16,
    )
    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    history = train_value_event_balanced_v70(
        model,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=norm,
        scales=scales,
        graph=graph,
        device=str(target),
        seed=int(args.seed),
        contract=DirectValueLossContractV70(),
    )
    prepared = prepare_static_v60(graph, target)
    metrics = {
        "fit_d2": evaluate_value_v70(model, cache, fit_d2, norm, prepared, device=target),
        "fit_d3": evaluate_value_v70(model, cache, fit_d3, norm, prepared, device=target),
        "holdout_d2": evaluate_value_v70(model, cache, holdout_d2, norm, prepared, device=target),
        "holdout_d3": evaluate_value_v70(model, cache, holdout_d3, norm, prepared, device=target),
    }
    magnitude = {
        "holdout_d3": evaluate_value_strata_v70(
            model, cache, holdout_d3, norm, prepared, device=target,
            q33_m3=float(np.quantile(np.concatenate([
                np.abs(np.asarray(cache.entry(n).arrays["exact_node_flood_volume_m3"][[i for i in cache.entry(n).indices if i != cache.entry(n).reference_index]], dtype=np.float64).sum(axis=1)
                - np.asarray(cache.entry(n).arrays["exact_node_flood_volume_m3"][cache.entry(n).reference_index], dtype=np.float64).sum())
                for n in fit_d3
            ]), 1.0 / 3.0)),
            q67_m3=float(np.quantile(np.concatenate([
                np.abs(np.asarray(cache.entry(n).arrays["exact_node_flood_volume_m3"][[i for i in cache.entry(n).indices if i != cache.entry(n).reference_index]], dtype=np.float64).sum(axis=1)
                - np.asarray(cache.entry(n).arrays["exact_node_flood_volume_m3"][cache.entry(n).reference_index], dtype=np.float64).sum())
                for n in fit_d3
            ]), 2.0 / 3.0)),
        )
    }
    gate_ok, gate_reasons = _value_gate(metrics["holdout_d3"])
    coefficient_bank = candidate_coefficients_v120(basis, contract=contract)
    normalization = RuntimeNormalizationV120(
        state_mean=norm.state_mean,
        state_std=norm.state_std,
        rainfall_mean=norm.rainfall_mean,
        rainfall_std=norm.rainfall_std,
        flow_mean=norm.flow_mean,
        flow_std=norm.flow_std,
    )
    lineage = {
        "git_head": _git_head(),
        "graph_sha256": _sha256(args.graph),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        **_cache_hashes(args.cache_manifest),
    }
    model_config = {
        "state_dim": state_dim,
        "rainfall_dim": rainfall_dim,
        "physics_dim": int(prepared_cpu.actuator_physics.shape[1]),
        "actuator_count": len(graph.actuator_ids),
        "hidden_dim": 96,
        "actuator_embedding_dim": 16,
        "direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3),
        "swmm_engine_version": str(args.swmm_engine_version).strip(),
    }
    if not model_config["swmm_engine_version"]:
        raise ValueError("V120 requires a non-empty SWMM engine version")
    bundle = {
        "bundle_contract": V120_BUNDLE_CONTRACT,
        "step2_contract": V120_CONTRACT,
        "base_value_contract": V70_CONTRACT,
        "kind": "control_value_only",
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "input_normalization": normalization.as_payload(),
        "basis_manifest": basis_manifest_v60(basis),
        "candidate_policy": {
            "contract": "D3_V2_DISTRIBUTION_MATCHED_FINITE_CANDIDATE_POLICY_V1",
            "candidate_count_including_hold": int(coefficient_bank.shape[0]),
            "candidate_banks": contract.candidate_banks,
            "candidates_per_bank": contract.candidates_per_bank,
            "families": ["single_group", "same_zone", "cross_zone", "sparse_low_discrepancy"],
            "continuous_gradient_search": False,
        },
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "lineage": lineage,
        "split": {
            "fit_event_ids": fit_events,
            "holdout_event_ids": holdout_events,
            "fit_rainfall_groups": fit_rain,
            "holdout_rainfall_groups": holdout_rain,
            "event_overlap": 0,
            "rainfall_overlap": 0,
        },
        "value_horizon_minutes": contract.value_horizon_minutes,
        "control_update_seconds": contract.control_update_seconds,
        "runtime_compatible": bool(gate_ok),
        "production_compatible": False,
        "value_gate": {"passed": bool(gate_ok), "reasons": gate_reasons},
        "metrics": metrics,
        "magnitude_strata": magnitude,
        "training_history": history,
        "boundary": {
            "new_swmm": False,
            "hydraulic_model_trained": False,
            "nodewise_action_effect_required": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out / "step2_v120_tfv_value_only_bundle.pt")
    report = {k: v for k, v in bundle.items() if k not in {"state_dict"}}
    (out / "STEP2_V120_VALUE_ONLY_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gate_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
