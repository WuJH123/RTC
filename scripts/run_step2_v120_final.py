"""Canonical execution-bound Project7 V120 trainer.

This entrypoint intentionally separates the upstream authoritative D2 source census
(4,800 branches) from the split-eligible V120 training cache.  Only events in the
frozen 18-event development-train cohort may enter fitting.  Targeted D3 must be
the complete 3,600-branch Train view.
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
    derive_magnitude_strata_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import (
    derive_target_scales_v70,
    evaluate_value_strata_v70,
    evaluate_value_v70,
)
from rtc.step2_v120_contract import (
    SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS,
    TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS,
    Step2V120Contract,
    V120_BUNDLE_CONTRACT,
    V120_CANDIDATE_POLICY_CONTRACT,
    V120_CONTRACT,
    V120_DATA_ELIGIBILITY_CONTRACT,
    v120_graph_semantic_sha256,
    v120_runtime_contract_sha256,
)
from rtc.step2_v70_contract import DirectValueLossContractV70, V70_CONTRACT

FROZEN_SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"


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
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    rules = {"rank": 0.35, "pairwise": 0.60, "sign_accuracy": 0.60, "top1_rate": 0.25}
    reasons: list[str] = []
    for key, floor in rules.items():
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value) or value < floor:
            reasons.append(f"{key}={value} < {floor}")
    for key in ("spread_ratio", "response_ratio"):
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value) or not 0.25 <= value <= 2.50:
            reasons.append(f"{key}={value} outside [0.25,2.50]")
    if not np.isfinite(float(metrics.get("mean_regret_m3", float("nan")))):
        reasons.append("mean_regret_m3 is non-finite")
    return not reasons, reasons


def _branch_count(cache: V60TrainCache, names: list[str]) -> int:
    return int(sum(len(cache.entry(name).indices) for name in names))


def _candidate_count(cache: V60TrainCache, names: list[str]) -> int:
    return int(sum(max(len(cache.entry(name).indices) - 1, 0) for name in names))


def _load_frozen_train_events(path: str | Path) -> tuple[dict[str, Any], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != FROZEN_SPLIT_CONTRACT:
        raise ValueError("V120 requires the frozen Project7 18/6/6 split contract")
    train = {str(x) for x in payload.get("development_train", [])}
    validation = {str(x) for x in payload.get("development_validation", [])}
    final = {str(x) for x in payload.get("final", [])}
    if len(train) != 18 or len(validation) != 6 or len(final) != 6:
        raise ValueError("V120 split contract must remain 18 Train / 6 Validation / 6 Final")
    if train & validation or train & final or validation & final:
        raise ValueError("V120 frozen scientific event splits overlap")
    return payload, train


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train execution-bound Project7 Step2 V120")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--swmm-engine-version", required=True)
    parser.add_argument(
        "--split-contract",
        default=str(repo / "configs" / "project7_v069_split_contract.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    contract = Step2V120Contract(seed=int(args.seed))
    contract.validate()
    split_payload, frozen_train = _load_frozen_train_events(args.split_contract)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != contract.actuator_count:
        raise ValueError("V120 requires frozen 109 actuators")

    cache = V60TrainCache(args.cache_manifest)
    legacy = cache.legacy_d3_names()
    if legacy:
        raise ValueError(f"V120 rejects legacy dense D3 groups: {legacy[:3]}")
    d2 = cache.names("D2")
    d3 = cache.targeted_d3_names()
    if not d2 or not d3:
        raise ValueError("V120 requires existing D2 plus targeted D3-v2")

    cache_events = {cache.entry(name).event_id for name in sorted(d2 + d3)}
    extra = sorted(cache_events - frozen_train)
    missing = sorted(frozen_train - cache_events)
    if extra:
        raise ValueError(
            "P0 V120 data leakage: cache contains Validation/Final/non-Train events: "
            + ", ".join(extra)
        )
    if missing:
        raise ValueError(
            "V120 canonical cache is incomplete for frozen Train18; missing: "
            + ", ".join(missing)
        )
    if len(cache_events) != contract.training_event_count:
        raise ValueError("V120 cache must contain exactly 18 frozen Train events")

    d2_branches = _branch_count(cache, d2)
    d3_branches = _branch_count(cache, d3)
    if d3_branches != TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS:
        raise ValueError(
            f"V120 requires complete targeted D3 Train view: {d3_branches} != "
            f"{TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS} branches"
        )

    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=sorted(d2 + d3), holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V120 internal split must contain D2/D3 on both sides")
    fit_events = sorted({cache.entry(name).event_id for name in fit})
    holdout_events = sorted({cache.entry(name).event_id for name in holdout})
    fit_rain = sorted({cache.entry(name).rainfall_group for name in fit})
    holdout_rain = sorted({cache.entry(name).rainfall_group for name in holdout})
    if set(fit_events) & set(holdout_events) or set(fit_rain) & set(holdout_rain):
        raise ValueError("V120 event/rainfall internal split leakage")
    if len(holdout_events) != contract.internal_holdout_event_count:
        raise ValueError(
            f"V120 frozen internal holdout must contain four events, got {len(holdout_events)}"
        )
    if len(fit_events) != contract.training_event_count - contract.internal_holdout_event_count:
        raise ValueError("V120 TrainFit must contain 14 events")

    norm = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    strata = derive_magnitude_strata_v60(cache, fit_d3)
    basis = build_control_basis_v60(graph)
    first = cache.entry(fit[0])
    state_dim = int(first.arrays["initial_state"].shape[-1])
    rainfall_dim = int(first.arrays["rainfall"].shape[-1])
    if state_dim != 6:
        raise ValueError("V120 requires the frozen six-state Step1 interface")
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
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    history = train_value_event_balanced_v70(
        model,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=norm,
        scales=scales,
        graph=graph,
        device=str(device),
        seed=int(args.seed),
        contract=DirectValueLossContractV70(),
    )
    prepared = prepare_static_v60(graph, device)
    metrics = {
        "fit_d2": evaluate_value_v70(model, cache, fit_d2, norm, prepared, device=device),
        "fit_d3": evaluate_value_v70(model, cache, fit_d3, norm, prepared, device=device),
        "holdout_d2": evaluate_value_v70(model, cache, holdout_d2, norm, prepared, device=device),
        "holdout_d3": evaluate_value_v70(model, cache, holdout_d3, norm, prepared, device=device),
    }
    magnitude = evaluate_value_strata_v70(
        model,
        cache,
        holdout_d3,
        norm,
        prepared,
        device=device,
        q33_m3=float(strata["q33_m3"]),
        q67_m3=float(strata["q67_m3"]),
    )
    gate_ok, reasons = _gate(metrics["holdout_d3"])
    cache_lineage = validate_v60_cache_lineage(args.cache_manifest)
    normalization = RuntimeNormalizationV120(
        norm.state_mean,
        norm.state_std,
        norm.rainfall_mean,
        norm.rainfall_std,
        norm.flow_mean,
        norm.flow_std,
    )
    engine = str(args.swmm_engine_version).strip()
    if not engine:
        raise ValueError("V120 requires a non-empty SWMM engine version")

    model_config = {
        "state_dim": state_dim,
        "rainfall_dim": rainfall_dim,
        "physics_dim": int(prepared_cpu.actuator_physics.shape[1]),
        "actuator_count": len(graph.actuator_ids),
        "hidden_dim": 96,
        "actuator_embedding_dim": 16,
        "direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3),
        "swmm_engine_version": engine,
    }
    candidate_count = int(candidate_coefficients_v120(basis, contract=contract).shape[0])
    data_census = {
        "contract": V120_DATA_ELIGIBILITY_CONTRACT,
        "source_d2_authoritative_branch_census": SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS,
        "source_d2_census_role": "upstream intervention census; not pooled across frozen split",
        "eligible_cache_d2_groups": len(d2),
        "eligible_cache_d2_branches": d2_branches,
        "eligible_cache_d2_candidates": _candidate_count(cache, d2),
        "targeted_d3_groups": len(d3),
        "targeted_d3_branches": d3_branches,
        "targeted_d3_candidates": _candidate_count(cache, d3),
        "eligible_event_count": len(cache_events),
        "eligible_events": sorted(cache_events),
        "non_train_events_loaded": [],
    }
    bundle = {
        "bundle_contract": V120_BUNDLE_CONTRACT,
        "step2_contract": V120_CONTRACT,
        "base_value_contract": V70_CONTRACT,
        "kind": "control_value_only_execution_bound",
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "input_normalization": normalization.as_payload(),
        "basis_manifest": basis_manifest_v60(basis),
        # Legacy global contract remains for frozen Step1 compatibility. V120 has
        # a separate content-addressed runtime hash below.
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "v120_runtime_contract_sha256": v120_runtime_contract_sha256(),
        "lineage": {
            "git_head": _git_head(),
            "graph_sha256": _sha256(args.graph),
            "graph_semantic_sha256": v120_graph_semantic_sha256(graph),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "split_contract_sha256": _sha256(args.split_contract),
            "training_entrypoint_sha256": _sha256(Path(__file__)),
            "basis_sha256_from_cache_lineage": str(cache_lineage["v60_control_basis_sha256"]),
            "design_sha256_from_cache_lineage": str(cache_lineage["v60_design_contract_sha256"]),
        },
        "data_census": data_census,
        "split": {
            "scientific_split_contract": str(split_payload["contract"]),
            "fit_event_ids": fit_events,
            "holdout_event_ids": holdout_events,
            "fit_rainfall_groups": fit_rain,
            "holdout_rainfall_groups": holdout_rain,
            "event_overlap": 0,
            "rainfall_overlap": 0,
            "validation_events_loaded": 0,
            "final_events_loaded": 0,
        },
        "candidate_policy": {
            "contract": V120_CANDIDATE_POLICY_CONTRACT,
            "candidate_count_including_hold": candidate_count,
            "candidate_banks": contract.candidate_banks,
            "families": [
                "hold",
                "single_group",
                "same_zone",
                "cross_zone",
                "sparse_low_discrepancy",
            ],
            "continuous_gradient_search": False,
            "score_only_executable_sequences": True,
            "first_move_bound_to_current_and_target_readback": True,
        },
        "value_horizon_minutes": 360,
        "control_update_seconds": 600,
        "runtime_compatible": bool(gate_ok),
        "production_compatible": False,
        "value_gate": {"passed": bool(gate_ok), "reasons": reasons},
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
    bundle_path = out / "step2_v120_execution_bound_bundle.pt"
    torch.save(bundle, bundle_path)
    report = {key: value for key, value in bundle.items() if key != "state_dict"}
    (out / "STEP2_V120_EXECUTION_BOUND_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not gate_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
