"""Canonical Train-only Project7 Step2 V7.0 development runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import (
    ControlValueSurrogateV70,
    DualStep2SurrogateV70,
    HydraulicResponseSurrogateV70,
)
from rtc.step2_hydraulic_effect_v70 import (
    derive_onset_positive_weight_v70,
    evaluate_onset_transition_v70,
)
from rtc.step2_hydraulic_eval_v70 import evaluate_hydraulic_effect_v70
from rtc.step2_optimization_v70 import (
    train_hydraulic_event_balanced_v70,
    train_value_event_balanced_v70,
)
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    derive_magnitude_strata_v60,
    deterministic_rainfall_split_v60,
    evaluate_hydraulic_v60,
)
from rtc.step2_train_response_v70 import (
    derive_target_scales_v70,
    evaluate_value_strata_v70,
    evaluate_value_v70,
)
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v70_contract import (
    DirectValueLossContractV70,
    HydraulicEffectLossContractV70,
    V70_CONTRACT,
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


def _build_models(graph, cache, fit_names, scales, basis):
    entry = cache.entry(fit_names[0])
    state_dim = int(entry.arrays["initial_state"].shape[-1])
    rainfall_dim = int(entry.arrays["rainfall"].shape[-1])
    if state_dim != 6:
        raise ValueError(f"V7 requires frozen six-state contract, got {state_dim}")
    prepared = prepare_static_v60(graph, "cpu")
    value = ControlValueSurrogateV70(
        state_dim=state_dim,
        rainfall_dim=rainfall_dim,
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=scales.direct_tfv_scale_m3,
        hidden_dim=96,
        actuator_embedding_dim=16,
    )
    common = {
        "state_dim": state_dim,
        "rainfall_dim": rainfall_dim,
        "node_static_dim": int(graph.static_node_features.shape[1]),
        "physics_dim": int(prepared.actuator_physics.shape[1]),
        "actuator_count": len(graph.actuator_ids),
        "hidden_dim": 64,
        "latent_dim": 32,
        "temporal_dim": 12,
    }
    hydraulic = HydraulicResponseSurrogateV70(
        state_scale=scales.base.state_scale,
        flow_scale=scales.base.flow_scale,
        horizon_contract=MultiResolutionHorizonV60(),
        **common,
    )
    DualStep2SurrogateV70(value, hydraulic).assert_disjoint_parameters()
    return value, hydraulic


def _value_preflight(value, cache, name, norm, graph, device):
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    value.to(target).eval()
    prepared = prepare_static_v60(graph, target)
    batch = cache.batch(name, norm, target)
    reference = batch.reference_settings[:, None]
    output = value(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        reference,
        batch.previous_actuator_flow,
        prepared,
    )
    if not torch.equal(output.delta_tfv_m3, torch.zeros_like(output.delta_tfv_m3)):
        raise RuntimeError("V7 direct value exact-zero contract failed")
    candidate = batch.candidate_settings[:, :1].clone().requires_grad_(True)
    output = value(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        candidate,
        batch.previous_actuator_flow,
        prepared,
    )
    gradient = torch.autograd.grad(output.delta_tfv_m3.sum(), candidate)[0]
    if not torch.isfinite(gradient).all() or torch.count_nonzero(gradient).item() == 0:
        raise RuntimeError("V7 direct value action gradient is non-finite or zero")
    return {
        "exact_zero": True,
        "action_gradient_finite": True,
        "action_gradient_nonzero": True,
        "action_gradient_median_abs": float(gradient.abs().median()),
        "action_gradient_max_abs": float(gradient.abs().max()),
    }


def _coefficient_gradient_audit(value, graph, basis, cache, names, norm, device):
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    value.to(target).eval()
    prepared = prepare_static_v60(graph, target)
    records = []
    for name in list(names)[:4]:
        batch = cache.batch(name, norm, target)
        coeff = torch.zeros(
            1,
            4,
            basis.temporal_basis_count,
            basis.group_count,
            dtype=batch.reference_settings.dtype,
            device=target,
        )
        for index in range(4):
            coeff[0, index, index % basis.temporal_basis_count, (3 * index) % basis.group_count] = 0.5 if index % 2 == 0 else -0.5
        coeff.requires_grad_(True)
        output = value.forward_coefficients(
            initial_state=batch.initial_state,
            rainfall=batch.rainfall,
            reference_settings=batch.reference_settings,
            coefficients=coeff,
            previous_actuator_flow=batch.previous_actuator_flow,
            prepared=prepared,
            basis=basis,
        )
        gradient = torch.autograd.grad(output.delta_tfv_m3.sum(), coeff)[0]
        records.append(gradient.detach().abs().cpu().numpy().reshape(-1))
    values = np.concatenate(records) if records else np.asarray([], dtype=np.float64)
    return {
        "groups": int(min(len(names), 4)),
        "finite_fraction": float(np.isfinite(values).mean()) if values.size else float("nan"),
        "nonzero_fraction": float((values > 1e-12).mean()) if values.size else float("nan"),
        "median_abs": float(np.median(values)) if values.size else float("nan"),
        "p90_abs": float(np.quantile(values, 0.90)) if values.size else float("nan"),
        "max_abs": float(values.max()) if values.size else float("nan"),
        "coefficient_dimension": int(basis.coefficient_dimension),
        "raw_action_dimension": int(len(graph.actuator_ids) * basis.horizon.control_blocks),
    }


def _checkpoint_payload(model, *, kind, lineage, basis_manifest, scales, split_sha, seed):
    return {
        "contract": V70_CONTRACT,
        "kind": kind,
        "state_dict": model.state_dict(),
        "lineage": lineage,
        "basis_manifest": basis_manifest,
        "direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3),
        "state_delta_scale": np.asarray(scales.state_delta_scale).tolist(),
        "flow_delta_scale": np.asarray(scales.flow_delta_scale).tolist(),
        "split_manifest_sha256": split_sha,
        "seed": int(seed),
        "production_compatible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project7 Step2 V7 canonical Train-only development")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError(f"V7 frozen actuator count must be 109, got {len(graph.actuator_ids)}")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V7 cache must reject legacy dense D3")
    d2 = cache.names("D2")
    d3 = cache.targeted_d3_names()
    if not d2 or not d3:
        raise ValueError("V7 requires existing D2 plus targeted D3-v2")
    selected = sorted(d2 + d3)
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [x for x in fit if x.startswith("D2::")]
    fit_d3 = [x for x in fit if x.startswith("D3::")]
    holdout_d2 = [x for x in holdout if x.startswith("D2::")]
    holdout_d3 = [x for x in holdout if x.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V7 event split must contain D2/D3 on both sides")

    fit_rain = sorted({cache.entry(x).rainfall_group for x in fit})
    holdout_rain = sorted({cache.entry(x).rainfall_group for x in holdout})
    fit_events = sorted({cache.entry(x).event_id for x in fit})
    holdout_events = sorted({cache.entry(x).event_id for x in holdout})
    if set(fit_rain) & set(holdout_rain) or set(fit_events) & set(holdout_events):
        raise ValueError("V7 split leakage detected")

    norm = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    onset_pos_weight = derive_onset_positive_weight_v70(
        cache,
        fit,
        cap=HydraulicEffectLossContractV70().onset_max_pos_weight,
    )
    magnitude = derive_magnitude_strata_v60(cache, fit_d3)
    basis = build_control_basis_v60(graph)
    basis_payload = basis_manifest_v60(basis)
    value, hydraulic = _build_models(graph, cache, fit, scales, basis)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    git_head = _git_head()
    lineage = {
        "git_head": git_head,
        "graph_sha256": _sha256(args.graph),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        "basis_sha256_from_cache_lineage": json.loads(Path(args.cache_manifest).read_text()).get("v60_control_basis_sha256", ""),
        "design_sha256_from_cache_lineage": json.loads(Path(args.cache_manifest).read_text()).get("v60_design_contract_sha256", ""),
    }
    split_payload = {
        "contract": "PROJECT7_STEP2_V70_SAME_V60_RAINFALL_HOLDOUT_V1",
        "fit_rainfall_groups": fit_rain,
        "holdout_rainfall_groups": holdout_rain,
        "fit_event_ids": fit_events,
        "holdout_event_ids": holdout_events,
        "fit_d2_groups": sorted(fit_d2),
        "fit_d3_groups": sorted(fit_d3),
        "holdout_d2_groups": sorted(holdout_d2),
        "holdout_d3_groups": sorted(holdout_d3),
        "rainfall_overlap": 0,
        "event_overlap": 0,
        "holdout_fraction": float(args.holdout_fraction),
        "seed": int(args.seed),
        **lineage,
    }
    split_sha = _write_json(out / "STEP2_V70_SPLIT_MANIFEST.json", split_payload)
    _write_json(
        out / "STEP2_V70_SCALES.json",
        {
            "contract": "PROJECT7_STEP2_V70_TRAINFIT_ONLY_SCALES_V1",
            "direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3),
            "state_scale": scales.base.state_scale.tolist(),
            "state_delta_scale": scales.state_delta_scale.tolist(),
            "flow_scale": scales.base.flow_scale.tolist(),
            "flow_delta_scale": scales.flow_delta_scale.tolist(),
            "onset_positive_weight": float(onset_pos_weight),
            "magnitude_q33_m3": float(magnitude["q33_m3"]),
            "magnitude_q67_m3": float(magnitude["q67_m3"]),
            "fit_groups": sorted(fit),
            **lineage,
        },
    )

    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    preflight = _value_preflight(value, cache, fit_d3[0], norm, graph, str(target))
    value_history = train_value_event_balanced_v70(
        value,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=norm,
        scales=scales,
        graph=graph,
        device=str(target),
        seed=args.seed,
    )
    prepared = prepare_static_v60(graph, target)
    value_metrics = {
        "fit_d2": evaluate_value_v70(value, cache, fit_d2, norm, prepared, device=target),
        "fit_d3": evaluate_value_v70(value, cache, fit_d3, norm, prepared, device=target),
        "holdout_d2": evaluate_value_v70(value, cache, holdout_d2, norm, prepared, device=target),
        "holdout_d3": evaluate_value_v70(value, cache, holdout_d3, norm, prepared, device=target),
    }
    value_strata = {
        "fit_d3": evaluate_value_strata_v70(
            value, cache, fit_d3, norm, prepared, device=target,
            q33_m3=float(magnitude["q33_m3"]), q67_m3=float(magnitude["q67_m3"]),
        ),
        "holdout_d3": evaluate_value_strata_v70(
            value, cache, holdout_d3, norm, prepared, device=target,
            q33_m3=float(magnitude["q33_m3"]), q67_m3=float(magnitude["q67_m3"]),
        ),
    }
    coefficient_gradient = _coefficient_gradient_audit(
        value, graph, basis, cache, holdout_d3, norm, str(target)
    )
    torch.save(
        _checkpoint_payload(
            value,
            kind="control_value",
            lineage=lineage,
            basis_manifest=basis_payload,
            scales=scales,
            split_sha=split_sha,
            seed=args.seed,
        ),
        out / "step2_v70_control_value_development.pt",
    )

    catastrophic_value_collapse = bool(value_metrics["fit_d3"]["spread_ratio"] < 1e-3)
    hydraulic_history: list[dict[str, Any]] = []
    hydraulic_metrics: dict[str, Any] = {"status": "SKIPPED_VALUE_COLLAPSE"}
    hydraulic_effect_metrics: dict[str, Any] = {"status": "SKIPPED_VALUE_COLLAPSE"}
    onset_metrics: dict[str, Any] = {"status": "SKIPPED_VALUE_COLLAPSE"}
    if not catastrophic_value_collapse:
        hydraulic_history = train_hydraulic_event_balanced_v70(
            hydraulic,
            cache,
            fit_d2_names=fit_d2,
            fit_d3_names=fit_d3,
            normalization=norm,
            scales=scales,
            onset_positive_weight=onset_pos_weight,
            graph=graph,
            device=str(target),
            seed=args.seed,
        )
        hydraulic_metrics = {
            "fit_d3": evaluate_hydraulic_v60(hydraulic, cache, fit_d3, norm, graph, device=target),
            "holdout_d3": evaluate_hydraulic_v60(hydraulic, cache, holdout_d3, norm, graph, device=target),
            "note": "V60 occurrence-style onset field is diagnostic only; V7 transition metric below is scientific primary for onset.",
        }
        hydraulic_effect_metrics = {
            "fit_d3": evaluate_hydraulic_effect_v70(hydraulic, cache, fit_d3, norm, prepared, scales, device=target),
            "holdout_d3": evaluate_hydraulic_effect_v70(hydraulic, cache, holdout_d3, norm, prepared, scales, device=target),
        }
        onset_metrics = {
            "fit_d3": evaluate_onset_transition_v70(hydraulic, cache, fit_d3, norm, prepared, device=target),
            "holdout_d3": evaluate_onset_transition_v70(hydraulic, cache, holdout_d3, norm, prepared, device=target),
        }
        torch.save(
            _checkpoint_payload(
                hydraulic,
                kind="hydraulic_response",
                lineage=lineage,
                basis_manifest=basis_payload,
                scales=scales,
                split_sha=split_sha,
                seed=args.seed,
            ),
            out / "step2_v70_hydraulic_development.pt",
        )

    report = {
        "contract": V70_CONTRACT,
        "boundary": {
            "swmm_run": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "production_wiring_modified": False,
            "legacy_dense_d3_used": False,
        },
        "lineage": lineage,
        "data": {
            "d2_groups": len(d2),
            "targeted_d3_groups": len(d3),
            "fit_groups": len(fit),
            "holdout_groups": len(holdout),
            "fit_events": len(fit_events),
            "holdout_events": len(holdout_events),
        },
        "control_dimension": {
            "raw": len(graph.actuator_ids) * basis.horizon.control_blocks,
            "basis_coefficients": basis.coefficient_dimension,
            "value_action_temporal_features_per_actuator": basis.temporal_basis_count,
        },
        "value_contract": {
            "prediction_target": "direct_signed_authoritative_delta_TFV_m3",
            "rate_head_in_primary_path": False,
            "uses_previous_actuator_flow": True,
            "structural_exact_zero": True,
            "loss": DirectValueLossContractV70().__dict__,
            "direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3),
        },
        "hydraulic_contract": {
            "absolute_plus_counterfactual_delta_supervision": True,
            "onset_target": "dry_to_flood_transition",
            "onset_positive_weight": float(onset_pos_weight),
            "loss": HydraulicEffectLossContractV70().__dict__,
        },
        "preflight": preflight,
        "value_history": value_history,
        "value_metrics": value_metrics,
        "value_magnitude_strata": value_strata,
        "coefficient_gradient": coefficient_gradient,
        "catastrophic_value_collapse": catastrophic_value_collapse,
        "hydraulic_history": hydraulic_history,
        "hydraulic_metrics": hydraulic_metrics,
        "hydraulic_effect_metrics": hydraulic_effect_metrics,
        "onset_transition_metrics": onset_metrics,
        "verdict": "RED_VALUE_COLLAPSE" if catastrophic_value_collapse else "DEVELOPMENT_COMPLETE_EXTERNAL_REVIEW_REQUIRED",
        "ready_for_formal": False,
        "ready_for_production": False,
    }
    _write_json(out / "STEP2_V70_DEVELOPMENT_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
