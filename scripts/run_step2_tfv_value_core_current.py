"""Current Project7 core Direct-TFV Step2 runner.

The scientific target is unchanged: causal current state + rainfall + reference/candidate actions ->
authoritative exact SWMM delta TFV. V4 training separates facility-effect learning from the runtime
HOLD decision: MAIN/JOINT learn values from all TrainFit reference families, while CONTROL uses only
D3 HOLD-reference groups. D4 remains a reference-shift stress diagnostic and never blocks Step3.
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
from rtc.step2_tfv_value import DIRECT_TFV_VALUE_CONTRACT, DirectFacilityTFVValueModel, DirectTFVValueDesign
from rtc.step2_tfv_value_training_v4 import (
    DIRECT_TFV_TRAINING_CONTRACT,
    DirectTFVTrainingDesignV4,
    derive_direct_tfv_target_scale_m3,
    evaluate_direct_tfv_value_model,
    train_direct_tfv_value_model_v4,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


CURRENT_DIRECT_TFV_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_CORE_SMOKE_DEV_V4"
DIRECT_DEV_PROFILE_CONTRACT = "PROJECT7_DIRECT_TFV_ALL_EXISTING_DEVELOPMENT_GROUPS_V1"
REPORT_FILENAME = "STEP2_DIRECT_TFV_VALUE_REPORT.json"
CHECKPOINT_FILENAME = "step2_direct_tfv_value_dev.pt"
SEED = 42
EXPECTED_COUNTS = (112, 112, 32, 32, 33, 15)


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke", "dev"), required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--main-epochs", type=int, default=0)
    p.add_argument("--interaction-epochs", type=int, default=0)
    p.add_argument("--control-epochs", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=5.0e-4)
    p.add_argument("--control-learning-rate", type=float, default=1.0e-4)
    return p


def _training_design(profile: str, args: argparse.Namespace) -> DirectTFVTrainingDesignV4:
    if profile == "smoke":
        main_epochs, interaction_epochs, control_epochs = 2, 1, 1
    else:
        main_epochs, interaction_epochs, control_epochs = 8, 5, 3
    if int(args.main_epochs) > 0:
        main_epochs = int(args.main_epochs)
    if int(args.interaction_epochs) > 0:
        interaction_epochs = int(args.interaction_epochs)
    if int(args.control_epochs) > 0:
        control_epochs = int(args.control_epochs)
    design = DirectTFVTrainingDesignV4(
        main_epochs=main_epochs,
        interaction_epochs=interaction_epochs,
        control_epochs=control_epochs,
        learning_rate=float(args.learning_rate),
        interaction_learning_rate=float(args.learning_rate),
        control_learning_rate=float(args.control_learning_rate),
        seed=SEED,
    )
    design.validate()
    return design


def _select_groups(
    profile: str,
    *,
    fit_d2: list[str],
    fit_d3: list[str],
    hold_d2: list[str],
    hold_d3: list[str],
    d4_fit: list[str],
    d4_audit: list[str],
) -> dict[str, list[str]]:
    if profile == "smoke":
        return profile_groups(
            get_execution_profile("smoke"),
            fit_d2=fit_d2,
            fit_d3=fit_d3,
            hold_d2=hold_d2,
            hold_d3=hold_d3,
            d4_fit=d4_fit,
            d4_audit=d4_audit,
            one_group=False,
        )
    if profile != "dev":
        raise ValueError(f"unsupported Direct-TFV profile: {profile!r}")
    return {
        "fit_d2": sorted(fit_d2),
        "fit_d3": sorted(fit_d3),
        "hold_d2": sorted(hold_d2),
        "hold_d3": sorted(hold_d3),
        "d4_fit": sorted(d4_fit),
        "d4_audit": sorted(d4_audit),
    }


def _finite_metrics(label: str, metrics: dict[str, float | int]) -> None:
    required = (
        "rank",
        "pairwise",
        "sign",
        "top1_fraction",
        "selected_harmful_fraction",
        "selected_true_delta_tfv_m3",
        "delta_tfv_mae_m3",
        "selected_regret_m3",
    )
    bad = [name for name in required if name not in metrics or not np.isfinite(float(metrics[name]))]
    if bad:
        raise RuntimeError(f"{label}: non-finite Direct-TFV metrics {bad}")


def main() -> None:
    started = time.perf_counter()
    args = _parser().parse_args()
    _seed_everything(SEED)
    profile_name = str(args.profile)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("core Direct-TFV requires exactly 109 actuators")
    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rain_store.validate()
    state_store.validate()
    if rain_store.forecast_mmhr.shape[1] != 72 or rain_store.forecast_mmhr.shape[2] != len(graph.node_ids):
        raise ValueError("causal rainfall store must match H72 graph")

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]
    d4_fit_names = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit_raw.names(D4_SOURCE_KIND)
    canonical_counts = tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3, d4_fit_names, d4_audit_names)))
    if canonical_counts != EXPECTED_COUNTS:
        raise ValueError(f"canonical Direct-TFV split differs from {EXPECTED_COUNTS}: got {canonical_counts}")

    selected = _select_groups(
        profile_name,
        fit_d2=fit_d2,
        fit_d3=fit_d3,
        hold_d2=hold_d2,
        hold_d3=hold_d3,
        d4_fit=d4_fit_names,
        d4_audit=d4_audit_names,
    )
    if profile_name == "dev" and tuple(
        len(selected[key])
        for key in ("fit_d2", "fit_d3", "hold_d2", "hold_d3", "d4_fit", "d4_audit")
    ) != EXPECTED_COUNTS:
        raise RuntimeError("Direct-TFV DEV must use every admitted existing Development group")

    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    d4_fit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain_store), state_store)
    d4_audit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain_store), state_store)
    normalization = derive_v127_input_normalization(
        base_cache=base,
        causal_rainfall=rain_store,
        causal_state=state_store,
        fit_names=selected["fit_d2"] + selected["fit_d3"],
    )
    raw_train_caches = {"D2": base, "D3": base, "D4": d4_fit_raw}
    train_groups = {
        "D2": selected["fit_d2"],
        "D3": selected["fit_d3"],
        "D4": selected["d4_fit"],
    }
    target_scale = derive_direct_tfv_target_scale_m3(raw_train_caches, train_groups)
    action_support = derive_direct_tfv_action_support(
        raw_train_caches,
        train_groups,
        actuator_ids=graph.actuator_ids,
        control_block_steps=2,
    )
    if profile_name == "dev" and int(action_support["single_facility_coverage_count"]) != 109:
        raise RuntimeError(
            "full Direct-TFV DEV requires 109/109 exact single-facility TrainFit coverage: "
            f"uncovered={action_support['uncovered_facilities']}"
        )

    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    model_design = DirectTFVValueDesign()
    model = DirectFacilityTFVValueModel(
        state_dim=int(first_state.shape[-1]),
        rainfall_dim=int(rain_store.forecast_mmhr.shape[-1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        target_scale_m3=target_scale,
        design=model_design,
    ).to(device)
    training_design = _training_design(profile_name, args)
    online_train_caches = {"D2": base_online, "D3": base_online, "D4": d4_fit_online}
    history = train_direct_tfv_value_model_v4(
        model,
        source_caches=online_train_caches,
        source_groups=train_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        design=training_design,
    )

    evaluations = {
        "trainfit_d2": evaluate_direct_tfv_value_model(
            model, cache=base_online, names=selected["fit_d2"], normalization=normalization, graph=graph, device=device
        ),
        "trainfit_d3": evaluate_direct_tfv_value_model(
            model, cache=base_online, names=selected["fit_d3"], normalization=normalization, graph=graph, device=device
        ),
        "internal_holdout_d2": evaluate_direct_tfv_value_model(
            model, cache=base_online, names=selected["hold_d2"], normalization=normalization, graph=graph, device=device
        ),
        "internal_holdout_d3": evaluate_direct_tfv_value_model(
            model, cache=base_online, names=selected["hold_d3"], normalization=normalization, graph=graph, device=device
        ),
        "d4_fit": evaluate_direct_tfv_value_model(
            model, cache=d4_fit_online, names=selected["d4_fit"], normalization=normalization, graph=graph, device=device
        ),
        "d4_audit": evaluate_direct_tfv_value_model(
            model, cache=d4_audit_online, names=selected["d4_audit"], normalization=normalization, graph=graph, device=device
        ),
    }
    for label, metrics in evaluations.items():
        _finite_metrics(label, metrics)

    lineage = {
        "graph_sha256": _sha(args.graph),
        "base_cache_sha256": _sha(args.cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
        "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "causal_rainfall_sha256": _sha(args.causal_store),
        "causal_state_store_sha256": _sha(args.causal_state_store),
        "step1_model_semantic_sha256": str(state_store.step1_model_semantic_sha256),
        "sensor_layout_semantic_sha256": str(state_store.sensor_layout_semantic_sha256),
        "causal_rainfall_forecast_contract": str(rain_store.forecast_contract),
    }
    resources = {
        "wall_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "peak_cuda_allocated_gb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
        ),
    }
    normalization_payload = {
        "state_mean": normalization.state_mean,
        "state_std": normalization.state_std,
        "rainfall_mean": normalization.rainfall_mean,
        "rainfall_std": normalization.rainfall_std,
        "flow_mean": normalization.flow_mean,
        "flow_std": normalization.flow_std,
    }
    selected_counts = {key: len(value) for key, value in selected.items()}
    checkpoint = {
        "contract": DIRECT_TFV_VALUE_CONTRACT,
        "training_contract": DIRECT_TFV_TRAINING_CONTRACT,
        "run_contract": CURRENT_DIRECT_TFV_RUN_CONTRACT,
        "direct_dev_profile_contract": DIRECT_DEV_PROFILE_CONTRACT,
        "profile": profile_name,
        "development_only": True,
        "seed": SEED,
        "model_design": asdict(model_design),
        "training_design": asdict(training_design),
        "target_scale_m3": float(target_scale),
        "state_dim": int(first_state.shape[-1]),
        "rainfall_dim": int(rain_store.forecast_mmhr.shape[-1]),
        "actuator_physics_dim": int(np.asarray(graph.actuator_physics).shape[1]),
        "normalization": normalization_payload,
        "lineage": lineage,
        "selected_group_counts": selected_counts,
        "action_support": action_support,
        "model_state_dict": model.state_dict(),
        "evaluations": evaluations,
        "history": history,
        "resources": resources,
    }
    checkpoint_path = out / CHECKPOINT_FILENAME
    torch.save(checkpoint, checkpoint_path)
    report = {
        "contract": CURRENT_DIRECT_TFV_RUN_CONTRACT,
        "step2_contract": DIRECT_TFV_VALUE_CONTRACT,
        "training_contract": DIRECT_TFV_TRAINING_CONTRACT,
        "direct_dev_profile_contract": DIRECT_DEV_PROFILE_CONTRACT,
        "profile": profile_name,
        "development_only": True,
        "seed": SEED,
        "primary_target": "authoritative SWMM exact delta TFV",
        "architecture": "shared pairwise sequence value V(candidate)-V(reference); 109 facility values + interaction value",
        "facility_balanced_main_regression": True,
        "control_training_reference_family": "D3 HOLD only",
        "d4_role": "reference-shift stress diagnostic only; never a Step3 admission gate",
        "top1_is_diagnostic_only": True,
        "primary_control_metrics": ["selected_true_delta_tfv_m3", "selected_regret_m3", "selected_harmful_fraction", "pairwise", "sign"],
        "complete_reference_sequence_encoded": True,
        "candidate_reference_antisymmetry_by_construction": True,
        "dev_uses_all_existing_development_groups": profile_name == "dev",
        "hydraulic_trajectory_primary_target": False,
        "gradient_label_used": False,
        "meaningful_1pct_threshold_used_for_training": False,
        "target_scale_m3": float(target_scale),
        "canonical_group_counts": {
            "fit_d2": 112,
            "fit_d3": 112,
            "hold_d2": 32,
            "hold_d3": 32,
            "d4_fit": 33,
            "d4_audit": 15,
        },
        "selected_group_counts": selected_counts,
        "action_support": action_support,
        "history": history,
        "evaluations": evaluations,
        "lineage": lineage,
        "resources": resources,
        "checkpoint": str(checkpoint_path),
        "runtime_promoted": False,
        "next_gate": "all-109 Direct-TFV Step3 solver audit on HOLD-reference D3 holdout; no threshold-calibration stage",
    }
    (out / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
