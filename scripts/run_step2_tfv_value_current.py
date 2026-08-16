"""Current Project7 Step2: learn 109-facility action value directly from exact SWMM delta TFV.

This runner is intentionally smaller than the legacy V128 hydraulic-world-model curriculum.
It preserves the online causal inputs (Step1 reconstructed current state and causal rainfall
forecast) but trains the primary control surrogate on the quantity Step3 actually needs:
state/action-conditioned delta TFV.

Development flow:
  MAIN  = single-actuator exact counterfactuals -> per-facility main effects
  JOINT = multi-actuator exact counterfactuals -> interaction residual only
  EVAL  = held-out rainfall groups + untouched D4 audit ranking/regret

No SWMM is launched here. Validation/Final/Formal/Policy-Lock are not accessed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

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
from rtc.step2_tfv_value import DIRECT_TFV_VALUE_CONTRACT, DirectFacilityTFVValueModel, DirectTFVValueDesign
from rtc.step2_tfv_value_training import (
    DIRECT_TFV_TRAINING_CONTRACT,
    DirectTFVTrainingDesign,
    derive_direct_tfv_target_scale_m3,
    evaluate_direct_tfv_value_model,
    train_direct_tfv_value_model,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


CURRENT_DIRECT_TFV_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_VALUE_SMOKE_DEV_V1"
REPORT_FILENAME = "STEP2_DIRECT_TFV_VALUE_REPORT.json"
CHECKPOINT_FILENAME = "step2_direct_tfv_value_dev.pt"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    p.add_argument("--learning-rate", type=float, default=5.0e-4)
    return p


def _training_design(profile: str, args: argparse.Namespace) -> DirectTFVTrainingDesign:
    if profile == "smoke":
        main_epochs, interaction_epochs = 2, 1
    else:
        main_epochs, interaction_epochs = 8, 5
    if int(args.main_epochs) > 0:
        main_epochs = int(args.main_epochs)
    if int(args.interaction_epochs) > 0:
        interaction_epochs = int(args.interaction_epochs)
    result = DirectTFVTrainingDesign(
        main_epochs=main_epochs,
        interaction_epochs=interaction_epochs,
        learning_rate=float(args.learning_rate),
        interaction_learning_rate=float(args.learning_rate),
    )
    result.validate()
    return result


def _finite_metrics(label: str, metrics: dict[str, float | int]) -> None:
    required = ("rank", "pairwise", "sign", "top1_fraction", "delta_tfv_mae_m3", "selected_regret_m3")
    bad = [name for name in required if name not in metrics or not np.isfinite(float(metrics[name]))]
    if bad:
        raise RuntimeError(f"{label}: direct TFV evaluation is non-finite for {bad}")


def main() -> None:
    args = _parser().parse_args()
    profile = get_execution_profile(str(args.profile))
    if profile.name not in {"smoke", "dev"}:
        raise ValueError("direct TFV current runner is Development-only until ranking promotion")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("direct TFV current runner requires exactly 109 actuators")
    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rain_store.validate()
    state_store.validate()
    if rain_store.forecast_mmhr.shape[1] != 72 or rain_store.forecast_mmhr.shape[2] != len(graph.node_ids):
        raise ValueError("causal rainfall store must match current H72 graph")

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
    if tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3))) != (112, 112, 32, 32):
        raise ValueError("canonical direct TFV D2/D3 split differs from 112/112/32/32")
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("canonical direct TFV D4 FIT/AUDIT census differs from 33/15")

    selected = profile_groups(
        profile,
        fit_d2=fit_d2,
        fit_d3=fit_d3,
        hold_d2=hold_d2,
        hold_d3=hold_d3,
        d4_fit=d4_fit_names,
        d4_audit=d4_audit_names,
        one_group=False,
    )
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
    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    design = DirectTFVValueDesign()
    model = DirectFacilityTFVValueModel(
        state_dim=int(first_state.shape[-1]),
        rainfall_dim=int(rain_store.forecast_mmhr.shape[-1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        target_scale_m3=target_scale,
        design=design,
    ).to(device)
    training_design = _training_design(profile.name, args)
    online_train_caches = {"D2": base_online, "D3": base_online, "D4": d4_fit_online}
    history = train_direct_tfv_value_model(
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
    checkpoint = {
        "contract": DIRECT_TFV_VALUE_CONTRACT,
        "training_contract": DIRECT_TFV_TRAINING_CONTRACT,
        "run_contract": CURRENT_DIRECT_TFV_RUN_CONTRACT,
        "profile": profile.name,
        "development_only": True,
        "model_design": asdict(design),
        "training_design": asdict(training_design),
        "target_scale_m3": float(target_scale),
        "state_dim": int(first_state.shape[-1]),
        "rainfall_dim": int(rain_store.forecast_mmhr.shape[-1]),
        "actuator_physics_dim": int(np.asarray(graph.actuator_physics).shape[1]),
        "normalization": {
            "state_mean": normalization.state_mean,
            "state_std": normalization.state_std,
            "rainfall_mean": normalization.rainfall_mean,
            "rainfall_std": normalization.rainfall_std,
            "flow_mean": normalization.flow_mean,
            "flow_std": normalization.flow_std,
        },
        "lineage": lineage,
        "model_state_dict": model.state_dict(),
        "evaluations": evaluations,
    }
    checkpoint_path = out / CHECKPOINT_FILENAME
    torch.save(checkpoint, checkpoint_path)
    report = {
        "contract": CURRENT_DIRECT_TFV_RUN_CONTRACT,
        "step2_contract": DIRECT_TFV_VALUE_CONTRACT,
        "training_contract": DIRECT_TFV_TRAINING_CONTRACT,
        "profile": profile.name,
        "development_only": True,
        "primary_target": "authoritative SWMM exact delta TFV",
        "architecture": "109 facility main effects + multi-actuator interaction residual",
        "hydraulic_trajectory_primary_target": False,
        "gradient_label_used": False,
        "meaningful_1pct_threshold_used_for_training": False,
        "target_scale_m3": float(target_scale),
        "selected_group_counts": {key: len(value) for key, value in selected.items()},
        "history": history,
        "evaluations": evaluations,
        "lineage": lineage,
        "checkpoint": str(checkpoint_path.resolve()),
        "runtime_promoted": False,
        "next_gate": "held-out delta-TFV ranking/top1/regret before Step3 runtime promotion",
    }
    (out / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
