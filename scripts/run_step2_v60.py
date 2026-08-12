"""Bounded development runner for the Project7 Step2 V6 dual-surrogate rebuild.

Requires a dedicated V6 shard cache containing existing D2 plus targeted D3-v2.
It never loads legacy V4/V5 surrogate checkpoints and it refuses legacy dense D3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_control_response_v60 import (
    ControlValueSurrogateV60,
    DualStep2SurrogateV60,
    HydraulicResponseSurrogateV60,
    prepare_static_v60,
)
from rtc.step2_optimization_v60 import (
    event_balance_summary_v60,
    train_hydraulic_event_balanced_v60,
    train_value_event_balanced_v60,
)
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    derive_target_scales_v60,
    deterministic_rainfall_split_v60,
    evaluate_hydraulic_v60,
    evaluate_value_v60,
)
from rtc.step2_v60_contract import MultiResolutionHorizonV60, V60_CONTRACT


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


def _model_dimensions(cache: V60TrainCache, name: str) -> tuple[int, int]:
    entry = cache.entry(name)
    arrays = entry.arrays
    return int(arrays["initial_state"].shape[-1]), int(arrays["rainfall"].shape[-1])


def _build_models(
    graph: GraphSchema,
    cache: V60TrainCache,
    fit_names: list[str],
    scales,
) -> tuple[ControlValueSurrogateV60, HydraulicResponseSurrogateV60]:
    state_dim, rain_dim = _model_dimensions(cache, fit_names[0])
    if state_dim != 6:
        raise ValueError(f"V6 requires frozen six-state hydraulic contract, got {state_dim}")
    safe_static = prepare_static_v60(graph, "cpu")
    common: dict[str, Any] = {
        "state_dim": state_dim,
        "rainfall_dim": rain_dim,
        "node_static_dim": int(graph.static_node_features.shape[1]),
        "physics_dim": int(safe_static.actuator_physics.shape[1]),
        "actuator_count": len(graph.actuator_ids),
        "hidden_dim": 64,
        "latent_dim": 32,
        "temporal_dim": 12,
    }
    value = ControlValueSurrogateV60(
        tfv_rate_scale_m3s=scales.tfv_rate_scale_m3s,
        **common,
    )
    hydraulic = HydraulicResponseSurrogateV60(
        state_scale=scales.state_scale,
        flow_scale=scales.flow_scale,
        horizon_contract=MultiResolutionHorizonV60(),
        **common,
    )
    DualStep2SurrogateV60(value, hydraulic).assert_disjoint_parameters()
    return value, hydraulic


def _coefficient_gradient_audit(value, graph, basis, cache, name, norm, device: str) -> dict[str, float]:
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    value.to(target).eval()
    prepared = prepare_static_v60(graph, target)
    batch = cache.batch(name, norm, target)
    reference = batch.reference_settings
    seed = torch.zeros(
        1,
        4,
        basis.temporal_basis_count,
        basis.group_count,
        dtype=reference.dtype,
        device=target,
    )
    for candidate_index in range(4):
        seed[
            0,
            candidate_index,
            candidate_index % basis.temporal_basis_count,
            (3 * candidate_index) % basis.group_count,
        ] = 0.5 if candidate_index % 2 == 0 else -0.5
    coefficients = seed.requires_grad_(True)
    reference_expanded = reference[:, None].expand(1, 4, -1, -1)
    candidate = basis.decode(reference_expanded, coefficients)
    output = value(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        candidate,
        prepared,
        batch.elapsed_seconds,
    )
    output.delta_tfv_m3.sum().backward()
    gradient = coefficients.grad
    if gradient is None:
        raise RuntimeError("V6 control-latent objective produced no coefficient gradient")
    finite = torch.isfinite(gradient)
    return {
        "finite_fraction": float(finite.float().mean()),
        "nonzero_fraction": float((gradient.abs() > 1e-12).float().mean()),
        "median_abs": float(gradient.abs().median()),
        "max_abs": float(gradient.abs().max()),
        "coefficient_dimension": int(basis.coefficient_dimension),
        "raw_action_dimension": int(len(graph.actuator_ids) * basis.horizon.control_blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded Project7 Step2 V6 dual-surrogate Train-only development"
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--cache-manifest",
        required=True,
        help="dedicated V6 D2 + targeted D3-v2 cache manifest",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError(
            f"V6 frozen actuator contract requires 109, got {len(graph.actuator_ids)}"
        )
    cache = V60TrainCache(args.cache_manifest)
    d2_names = cache.names("D2")
    targeted_d3_names = cache.targeted_d3_names()
    legacy_d3_names = cache.legacy_d3_names()
    if not d2_names:
        raise ValueError("V6 cache contains no D2 groups")
    if not targeted_d3_names:
        raise ValueError(
            "V6 cache contains no targeted D3-v2 groups. "
            "Design/run D3-v2 and build the dedicated V6 index/cache first."
        )
    if legacy_d3_names:
        raise ValueError(
            f"V6 dedicated cache contains {len(legacy_d3_names)} legacy dense D3 groups; "
            "rebuild from D2 + targeted D3-v2 only"
        )

    selected = sorted(d2_names + targeted_d3_names)
    fit, holdout = deterministic_rainfall_split_v60(
        cache,
        names=selected,
        holdout_fraction=float(args.holdout_fraction),
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V6 rainfall-group split must contain D2 and targeted D3 on both sides")

    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v60(cache, fit)
    value, hydraulic = _build_models(graph, cache, fit, scales)
    value_history = train_value_event_balanced_v60(
        value,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=normalization,
        scales=scales,
        graph=graph,
        device=args.device,
        seed=args.seed,
    )
    hydraulic_history = train_hydraulic_event_balanced_v60(
        hydraulic,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=normalization,
        scales=scales,
        graph=graph,
        device=args.device,
        seed=args.seed,
    )
    target = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    prepared = prepare_static_v60(graph, target)
    value_metrics = {
        "fit_d2": evaluate_value_v60(value, cache, fit_d2, normalization, prepared, device=target),
        "fit_d3": evaluate_value_v60(value, cache, fit_d3, normalization, prepared, device=target),
        "holdout_d2": evaluate_value_v60(value, cache, holdout_d2, normalization, prepared, device=target),
        "holdout_d3": evaluate_value_v60(value, cache, holdout_d3, normalization, prepared, device=target),
    }
    hydraulic_metrics = {
        "holdout_d2": evaluate_hydraulic_v60(
            hydraulic, cache, holdout_d2, normalization, graph, device=target
        ),
        "holdout_d3": evaluate_hydraulic_v60(
            hydraulic, cache, holdout_d3, normalization, graph, device=target
        ),
    }
    control_basis = build_control_basis_v60(graph)
    coefficient_gradient = _coefficient_gradient_audit(
        value,
        graph,
        control_basis,
        cache,
        holdout_d3[0],
        normalization,
        str(target),
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_contract": "PROJECT7_STEP2_V60_DEVELOPMENT_VALUE_ONLY",
            "scientific_split": "development",
            "training_fold": "TrainFit",
            "model_state_dict": value.state_dict(),
            "basis_manifest": basis_manifest_v60(control_basis),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "production_compatible": False,
        },
        out / "step2_v60_control_value_development.pt",
    )
    torch.save(
        {
            "checkpoint_contract": "PROJECT7_STEP2_V60_DEVELOPMENT_HYDRAULIC_ONLY",
            "scientific_split": "development",
            "training_fold": "TrainFit",
            "model_state_dict": hydraulic.state_dict(),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "production_compatible": False,
        },
        out / "step2_v60_hydraulic_response_development.pt",
    )
    report = {
        "contract": V60_CONTRACT,
        "graph_actuators": len(graph.actuator_ids),
        "fit_groups": len(fit),
        "holdout_groups": len(holdout),
        "fit_d2": len(fit_d2),
        "fit_d3": len(fit_d3),
        "holdout_d2": len(holdout_d2),
        "holdout_d3": len(holdout_d3),
        "legacy_dense_d3_present": False,
        "control_basis": basis_manifest_v60(control_basis),
        "event_balance": {
            "fit_d2": event_balance_summary_v60(cache, fit_d2),
            "fit_d3": event_balance_summary_v60(cache, fit_d3),
        },
        "value_metrics": value_metrics,
        "hydraulic_metrics": hydraulic_metrics,
        "control_latent_gradient": coefficient_gradient,
        "value_history": value_history,
        "hydraulic_history": hydraulic_history,
        "boundaries": {
            "validation_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "production_wiring_modified": False,
        },
    }
    report_path = out / "STEP2_V60_DEVELOPMENT_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
