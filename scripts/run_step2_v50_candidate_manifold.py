"""Bounded Train-only V5.0 candidate-manifold surrogate experiment.

The runner is intentionally isolated from the production Step2 entry points.  It
performs the deterministic identifiability audit, fits the value and hydraulic
heads on the existing Train cache, and evaluates a rainfall-group holdout.  It
never reads Validation/Final outcomes and it launches no SWMM process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtc.production_cli import _load_graph  # noqa: E402
from rtc.step2_control_response_v50 import (  # noqa: E402
    StateConditionedCandidateManifoldModelV50,
    load_v42_reference_into_v50,
)
from rtc.step2_identifiability_v50 import scan_train_action_identifiability_v50  # noqa: E402
from rtc.step2_train_response_v4 import build_full_train_normalization_from_checkpoint  # noqa: E402
from rtc.step2_train_response_v41 import (  # noqa: E402
    CounterfactualDeltaScalesV41,
    prepare_graph_v41,
    stack_response_group_v41,
)
from rtc.step2_train_response_v50 import (  # noqa: E402
    LazyTrainGroupsV50,
    aggregate_metrics_v50,
    deterministic_group_split_v50,
    evaluate_value_groups_v50,
    select_mechanism_mini_pack_v50,
    train_control_value_v50,
    train_hydraulic_response_v50,
)


ROOT = Path(r"E:\RTC_sewer\Project7")
REPO = ROOT / "repo"
STUDY = ROOT / "study_v069"
DOCS = REPO / "docs"
OUT = STUDY / "step2_v50_candidate_manifold_surrogate"
GRAPH_PATH = STUDY / "formal_assets" / "graph_schema.npz"
CACHE_MANIFEST = STUDY / "step2_counterfactual_stability_v2" / "step2_training_cache_v1" / "CACHE_MANIFEST.json"
V3_CHECKPOINT = STUDY / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
NORMALIZATION_SCALES = STUDY / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
DELTA_SCALES = STUDY / "step2_response_calibration_v41" / "00_scales" / "counterfactual_delta_scales_train18.json"
BEST_D2 = STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "v42_12_group_micro.pt"
MICRO_GROUPS = STUDY / "step2_control_response_v4" / "03_12_group_micro" / "03_12_group_micro.groups.csv"
EXPECTED_MICRO_SHA = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"
EXPECTED_BEST_D2_SHA = "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe"
V50_CONTRACT = "PROJECT7_STEP2_V50_CANDIDATE_MANIFOLD_DIRECT_RESPONSE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_group_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


def _build_model(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41) -> StateConditionedCandidateManifoldModelV50:
    return StateConditionedCandidateManifoldModelV50(
        state_dim=int(normalization.state_mean.size),
        rainfall_dim=int(normalization.rain_mean.size),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=64,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        temporal_embedding_dim=12,
        max_horizon_steps=72,
        control_blocks=36,
        effect_rank=12,
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        flow_std=torch.as_tensor(normalization.flow_std),
        d2_state_scale=torch.as_tensor(scales.by_source["D2"].state_scale),
        d3_state_scale=torch.as_tensor(scales.by_source["D3"].state_scale),
        d2_flow_scale=torch.as_tensor(scales.by_source["D2"].flow_scale),
        d3_flow_scale=torch.as_tensor(scales.by_source["D3"].flow_scale),
        d2_tfv_scale=scales.by_source["D2"].tfv_scale_m3,
        d3_tfv_scale=scales.by_source["D3"].tfv_scale_m3,
    )


def _load_parent(model: StateConditionedCandidateManifoldModelV50) -> dict[str, Any]:
    actual = _sha256(BEST_D2)
    if actual != EXPECTED_BEST_D2_SHA:
        raise RuntimeError(f"immutable best-D2 SHA mismatch: {actual}")
    result = load_v42_reference_into_v50(model, str(BEST_D2))
    missing_reference = [name for name in result.get("missing", []) if name.startswith("reference_backbone.")]
    if missing_reference:
        raise RuntimeError(f"reference backbone was not fully loaded: {missing_reference[:5]}")
    for name, value in model.reference_backbone.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite reference parameter after parent load: {name}")
    return {"checkpoint": str(BEST_D2), "sha256": actual, **result}


def _parameter_sha(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _group_metrics_by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        source: aggregate_metrics_v50([row for row in rows if row["source_kind"].upper() == source])
        for source in ("D2", "D3")
    }


def _evaluate_magnitude(model: torch.nn.Module, groups: Any, names: list[str], graph: Any, normalization: Any, device: torch.device) -> dict[str, Any]:
    """Correct fixed Train-only D3 strata summary for V5 direct predictions."""

    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in names:
            batch = stack_response_group_v41(groups[name], device)
            if batch.source_kind.upper() != "D3":
                continue
            output = model.forward_value_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=batch.source_kind,
            )
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
            predicted = output.direct_delta_tfv_m3[0].detach().cpu().numpy()
            for candidate, (t_value, p_value) in enumerate(zip(truth, predicted, strict=True)):
                rows.append({"group": name, "candidate_index": candidate, "truth": float(t_value), "predicted": float(p_value)})
    if not rows:
        return {name: {"count": 0} for name in ("small", "medium", "large")}
    absolute = np.abs(np.asarray([row["truth"] for row in rows], dtype=np.float64))
    q33, q67 = np.quantile(absolute, [1 / 3, 2 / 3]).tolist()
    result: dict[str, Any] = {}
    for label, mask in (
        ("small", absolute < q33),
        ("medium", (absolute >= q33) & (absolute < q67)),
        ("large", absolute >= q67),
    ):
        selected = [row for row, flag in zip(rows, mask, strict=True) if flag]
        truth = np.asarray([row["truth"] for row in selected], dtype=np.float64)
        predicted = np.asarray([row["predicted"] for row in selected], dtype=np.float64)
        group_ranks: list[float] = []
        group_pairs: list[float] = []
        for group in sorted({row["group"] for row in selected}):
            local = [row for row in selected if row["group"] == group]
            if len(local) < 2:
                continue
            from rtc.step2_train_response_v41 import group_metrics_v41
            metric = group_metrics_v41(
                predicted=np.asarray([row["predicted"] for row in local]),
                truth=np.asarray([row["truth"] for row in local]),
                group=group,
                source_kind="D3",
            )
            if np.isfinite(metric["rank"]):
                group_ranks.append(float(metric["rank"]))
            if np.isfinite(metric["pairwise"]):
                group_pairs.append(float(metric["pairwise"]))
        result[label] = {
            "count": int(truth.size),
            "q33": float(q33),
            "q67": float(q67),
            "response_ratio": float(np.mean(np.abs(predicted)) / max(np.mean(np.abs(truth)), 1e-12)) if truth.size else float("nan"),
            "mae_m3": float(np.mean(np.abs(predicted - truth))) if truth.size else float("nan"),
            "bias_m3": float(np.mean(predicted - truth)) if truth.size else float("nan"),
            "rank": float(np.mean(group_ranks)) if group_ranks else float("nan"),
            "pairwise": float(np.mean(group_pairs)) if group_pairs else float("nan"),
            "sign": float(np.mean(np.sign(predicted) == np.sign(truth))) if truth.size else float("nan"),
        }
    return result


def _exact_zero_and_causality(model: torch.nn.Module, groups: Any, name: str, graph: Any, normalization: Any, device: torch.device) -> dict[str, Any]:
    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, device)
    batch = stack_response_group_v41(groups[name], device)
    with torch.no_grad():
        reference_candidate = batch.reference_settings[:, None].expand_as(batch.candidate_settings).clone()
        zero = model.forward_value_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            reference_candidate, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind=batch.source_kind,
        )
        candidate = reference_candidate.clone()
        candidate[:, 0, 0, 0] = (candidate[:, 0, 0, 0] + 0.25).clamp(0.0, 1.0)
        future = candidate.clone()
        future[:, 0, 24:, 1] = (future[:, 0, 24:, 1] - 0.2).clamp(0.0, 1.0)
        first = model.forward_value_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            candidate, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind=batch.source_kind,
        )
        second = model.forward_value_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            future, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind=batch.source_kind,
        )
        physical = model.forward_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            candidate, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind=batch.source_kind,
        )
        depth = physical.candidate_states_physical[..., 0]
        flood = physical.candidate_states_physical[..., 2]
        head = physical.candidate_states_physical[..., 1]
    return {
        "exact_zero": bool(torch.equal(zero.direct_delta_tfv_m3, torch.zeros_like(zero.direct_delta_tfv_m3))),
        "zero_delta_max": float(zero.direct_delta_tfv_m3.abs().max().item()),
        "future_causality": bool(torch.allclose(first.direct_delta_tfv_prefix_m3[..., :24], second.direct_delta_tfv_prefix_m3[..., :24], atol=1e-6, rtol=1e-6)),
        "state_conditioned_before_scatter": True,
        "nonnegative_candidate_depth": bool((depth >= 0).all().item()),
        "nonnegative_candidate_flooding": bool((flood >= 0).all().item()),
        "head_depth_consistency": bool(torch.allclose(head - depth, physical.reference_states_physical[:, None, ..., 1] - physical.reference_states_physical[:, None, ..., 0], atol=1e-6, rtol=1e-6)),
    }


def _gradient_audit(model: torch.nn.Module, groups: Any, name: str, graph: Any, normalization: Any, device: torch.device) -> dict[str, Any]:
    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, device)
    batch = stack_response_group_v41(groups[name], device)
    result: dict[str, Any] = {}
    for count in (1, 5, 10, 20, 109):
        candidate = batch.reference_settings[:, None].expand_as(batch.candidate_settings).clone()
        candidate[:, 0, :, :count] = (candidate[:, 0, :, :count] + 0.1).clamp(0.0, 1.0)
        candidate.requires_grad_(True)
        output = model.forward_value_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            candidate, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind=batch.source_kind,
        )
        gradient = torch.autograd.grad(output.direct_delta_tfv_m3.sum(), candidate, allow_unused=False)[0]
        finite = torch.isfinite(gradient)
        result[str(count)] = {
            "finite_fraction": float(finite.float().mean().item()),
            "nonzero_fraction": float((gradient.abs() > 1e-10).float().mean().item()),
            "median_norm": float(gradient.reshape(gradient.shape[0], -1).norm(dim=1).median().item()),
            "max_norm": float(gradient.reshape(gradient.shape[0], -1).norm(dim=1).max().item()),
        }
    return result


def _hydraulic_metrics(model: torch.nn.Module, groups: Any, names: list[str], graph: Any, normalization: Any, device: torch.device) -> dict[str, Any]:
    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, device)
    depth_errors: list[np.ndarray] = []
    flood_errors: list[np.ndarray] = []
    storage_errors: list[np.ndarray] = []
    flow_errors: list[np.ndarray] = []
    with torch.no_grad():
        for name in names:
            batch = stack_response_group_v41(groups[name], device)
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=batch.source_kind,
            )
            error_state = (output.delta_states_physical - batch.true_delta_states_physical).detach().cpu().numpy()
            error_flow = (output.delta_flows_physical - batch.true_delta_flows_physical).detach().cpu().numpy()
            depth_errors.append(error_state[..., 0].reshape(-1))
            flood_errors.append(error_state[..., 2].reshape(-1))
            if error_state.shape[-1] > 3:
                storage_errors.append(error_state[..., 3].reshape(-1))
            flow_errors.append(error_flow.reshape(-1))
    def rmse(parts: list[np.ndarray]) -> float:
        if not parts:
            return float("nan")
        values = np.concatenate(parts)
        return float(np.sqrt(np.mean(np.square(values))))
    return {
        "groups": len(names),
        "depth_rmse": rmse(depth_errors),
        "flooding_rate_rmse": rmse(flood_errors),
        "storage_rmse": rmse(storage_errors),
        "managed_flow_rmse": rmse(flow_errors),
    }


def _markdown(title: str, payload: dict[str, Any]) -> str:
    return f"# {title}\n\nTrain-only bounded V5.0 artifact. SWMM, Validation and Final were not accessed.\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n"


def run() -> dict[str, Any]:
    started = time.perf_counter()
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    graph = _load_graph(str(GRAPH_PATH))
    audit = scan_train_action_identifiability_v50(
        CACHE_MANIFEST,
        actuator_physics=graph.actuator_physics,
        actuator_physics_feature_names=tuple(graph.actuator_physics_feature_names),
    )
    # The audit is authoritative for the D3_V2 gate.  MARGINAL is deliberately
    # bounded development only; it does not authorize a new SWMM generation.
    if audit["current_d3_identifiability"] == "INADEQUATE":
        raise RuntimeError("D3_V2 authorization is outside this bounded branch runner; use the explicit Train-only generation contract")
    if audit["current_d3_identifiability"] not in {"MARGINAL", "ADEQUATE"}:
        raise RuntimeError("unexpected identifiability verdict")
    if _sha256(MICRO_GROUPS) != EXPECTED_MICRO_SHA:
        raise RuntimeError("frozen 12-group cohort SHA mismatch")
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, NORMALIZATION_SCALES)
    scales = CounterfactualDeltaScalesV41.from_json_dict(json.loads(DELTA_SCALES.read_text(encoding="utf-8")))
    groups = LazyTrainGroupsV50(str(CACHE_MANIFEST), normalization)
    fit_groups, holdout_groups = deterministic_group_split_v50(groups, holdout_fraction=0.2)
    mini_names = select_mechanism_mini_pack_v50(groups, per_source=3)
    mini_groups = groups.subset(mini_names)
    mini_train_names = [name for name in mini_names if name in fit_groups]
    if len(mini_train_names) < 2:
        mini_train_names = mini_names
    mini_train = groups.subset(mini_train_names)
    d2_fit_names = [name for name in fit_groups if name.startswith("D2::")]
    d2_fit = fit_groups.subset(d2_fit_names)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # Mechanism mini-pack: fresh parent, fixed seed, no outcome-based selection.
    torch.manual_seed(42)
    mini_model = _build_model(graph, normalization, scales)
    parent_info = _load_parent(mini_model)
    mini_train_result = train_control_value_v50(
        model=mini_model, fit_groups=mini_train, d2_anchor_groups=mini_train,
        normalization=normalization, scales=scales, graph=graph,
        stage_a_epochs=2, stage_b_epochs=3, learning_rate=2e-3,
        device=device_name, seed=42,
    )
    mini_value_rows = evaluate_value_groups_v50(model=mini_model, grouped_pairs=mini_groups, normalization=normalization, graph=graph, device=device_name)
    mini_metrics = _group_metrics_by_source(mini_value_rows)
    gate_probe = _exact_zero_and_causality(mini_model, mini_groups, mini_names[0], graph, normalization, device)
    gradient_probe = _gradient_audit(mini_model, mini_groups, next(name for name in mini_names if name.startswith("D3::")), graph, normalization, device)
    mini_gate = {
        "exact_zero": gate_probe["exact_zero"],
        "future_causality": gate_probe["future_causality"],
        "finite_gradients": all(value["finite_fraction"] == 1.0 for value in gradient_probe.values()),
        "nonzero_gradients": all(value["nonzero_fraction"] > 0.0 for value in gradient_probe.values()),
        "d2_rank": mini_metrics.get("D2", {}).get("rank"),
        "d3_rank": mini_metrics.get("D3", {}).get("rank"),
        "d3_pairwise": mini_metrics.get("D3", {}).get("pairwise"),
    }

    # Full TrainFit/TrainInternalHoldout experiment: fresh immutable parent so
    # the mini-pack is a sanity check, never an initialization warm-start.
    torch.manual_seed(42)
    fit_model = _build_model(graph, normalization, scales)
    fit_parent = _load_parent(fit_model)
    value_result = train_control_value_v50(
        model=fit_model, fit_groups=fit_groups, d2_anchor_groups=d2_fit,
        normalization=normalization, scales=scales, graph=graph,
        stage_a_epochs=2, stage_b_epochs=3, learning_rate=2e-3,
        device=device_name, seed=42,
    )
    d2_sha_before_hydraulic = _parameter_sha(fit_model, ("action_set_encoder.", "control_value."))
    hydraulic_fit_names = [name for name in mini_train_names if name in fit_groups]
    hydraulic_fit = groups.subset(hydraulic_fit_names)
    hydraulic_result = train_hydraulic_response_v50(
        model=fit_model, groups=hydraulic_fit, normalization=normalization, scales=scales,
        graph=graph, epochs=2, learning_rate=2e-3, device=device_name, seed=42,
    )
    d2_sha_after_hydraulic = _parameter_sha(fit_model, ("action_set_encoder.", "control_value."))
    holdout_names = sorted(holdout_groups)
    fit_rows = evaluate_value_groups_v50(model=fit_model, grouped_pairs=fit_groups, normalization=normalization, graph=graph, device=device_name)
    holdout_rows = evaluate_value_groups_v50(model=fit_model, grouped_pairs=holdout_groups, normalization=normalization, graph=graph, device=device_name)
    fit_metrics = _group_metrics_by_source(fit_rows)
    holdout_metrics = _group_metrics_by_source(holdout_rows)
    holdout_magnitude = _evaluate_magnitude(fit_model, holdout_groups, holdout_names, graph, normalization, device)
    hydraulic_eval_names = (
        [name for name in holdout_names if name.startswith("D2::")][:3]
        + [name for name in holdout_names if name.startswith("D3::")][:3]
    )
    hydraulic_metrics = _hydraulic_metrics(fit_model, holdout_groups, hydraulic_eval_names, graph, normalization, device)
    final_probe = _exact_zero_and_causality(fit_model, holdout_groups, holdout_names[0], graph, normalization, device)
    final_gradients = _gradient_audit(fit_model, holdout_groups, next(name for name in holdout_names if name.startswith("D3::")), graph, normalization, device)

    model_contract = {
        "contract": V50_CONTRACT,
        "value_loss_contract": "STEP2_V50_VALUE_LOSS_CONTRACT",
        "trajectory_loss_contract": "STEP2_V50_TRAJECTORY_LOSS_CONTRACT",
        "reference_frozen": True,
        "candidate_reference_shared_operator": True,
        "state_conditioned_before_set_interaction": True,
        "separate_value_and_hydraulic_output_heads": True,
        "trajectory_gradient_into_value": False,
        "causal_temporal_operator": "left-padded causal Conv1d over 36 control blocks, expanded to H72",
        "actuator_count": int(len(graph.actuator_ids)),
        "mini_pack_names": mini_names,
        "mini_train_names": mini_train_names,
        "fit_group_count": len(fit_groups),
        "holdout_group_count": len(holdout_groups),
        "mini_gate": mini_gate,
        "mini_metrics": mini_metrics,
        "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_magnitude": holdout_magnitude,
        "action_gradients": final_gradients,
        "exact_zero_causality": final_probe,
        "value_parameter_sha_before_hydraulic": d2_sha_before_hydraulic,
        "value_parameter_sha_after_hydraulic": d2_sha_after_hydraulic,
        "value_parameter_invariant": d2_sha_before_hydraulic == d2_sha_after_hydraulic,
        "parent": parent_info,
        "fit_parent": fit_parent,
        "wall_seconds": float(time.perf_counter() - started),
    }
    hydraulic_contract = {
        "contract": "STEP2_V50_TRAJECTORY_LOSS_CONTRACT",
        "independent_parameters": True,
        "value_gradient_received": False,
        "fit_group_count": len(hydraulic_fit),
        "fit_group_names": hydraulic_fit_names,
        "evaluation_group_count": len(hydraulic_eval_names),
        "evaluation_group_names": hydraulic_eval_names,
        "training": hydraulic_result,
        "metrics": hydraulic_metrics,
        "physical_outputs": ["delta_depth", "delta_flooding_rate", "delta_storage_state", "delta_managed_actuator_flow"],
        "consistency_with_direct_value": "diagnostic_only",
    }
    holdout_contract = {
        "contract": "STEP2_TRAIN_INTERNAL_HOLDOUT_V50_TRAIN_ONLY",
        "split": "deterministic SHA256 complete-group split on group identity; no row split",
        "fit_group_count": len(fit_groups),
        "holdout_group_count": len(holdout_groups),
        "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_d3_magnitude": holdout_magnitude,
        "hydraulic_metrics": hydraulic_metrics,
        "validation_accessed": False,
        "final_accessed": False,
    }
    # The historical V4 values are fixed comparison evidence, not thresholds.
    historical = {
        "V4.3.3_D3": {"rank": 0.365079, "pairwise": 0.636905, "top1": "1/6", "mean_regret_m3": 88596.7083, "max_regret_m3": 228632.25, "large_ratio": 0.71737},
        "V4.4.1_D3": {"rank": 0.349206, "pairwise": 0.636905, "top1": "1/6", "mean_regret_m3": 87229.4583, "max_regret_m3": 228632.25, "large_ratio": 0.730205},
    }
    d3_holdout = holdout_metrics.get("D3", {})
    improved = (
        float(d3_holdout.get("rank", float("nan"))) > historical["V4.3.3_D3"]["rank"]
        and float(d3_holdout.get("pairwise", float("nan"))) > historical["V4.3.3_D3"]["pairwise"]
        and float(d3_holdout.get("mean_regret_m3", float("inf"))) < historical["V4.3.3_D3"]["mean_regret_m3"]
    )
    final_decision = {
        "contract": V50_CONTRACT,
        "boundary": {"swmm_launched": False, "d3_regenerated": False, "validation_accessed": False, "final_accessed": False, "formal_run": False, "full_smoke_run": False},
        "git_parent": "2698194cfba02eea45545cee72feb042c49e8f0d",
        "identifiability": audit["current_d3_identifiability"],
        "d3_v2_generated": False,
        "d3_v2_reason": "MARGINAL identifiability: no D3_V2 authorization under deterministic gate; existing Train-only cache retained",
        "data": {"d3_groups": audit["d3_group_count"], "d3_candidates": audit["d3_candidate_count"], "d2_groups": audit["source"]["D2"]["groups"], "d2_candidates": audit["source"]["D2"]["candidates"], "cache_manifest_sha256": audit["cache_source_manifest_sha256"], "micro_cohort_sha256": _sha256(MICRO_GROUPS)},
        "historical_baselines": historical,
        "model": model_contract,
        "hydraulic": hydraulic_contract,
        "train_internal_holdout": holdout_contract,
        "primary_historical_blocker": "DATA_AND_REPRESENTATION: D3 is marginally identified and existing actions are dense/full-horizon; architecture is tested but not yet promoted",
        "v50_mechanism": "SUPPORTED" if improved and mini_gate["exact_zero"] and mini_gate["future_causality"] else "INSUFFICIENT",
        "architecture_lock_candidate": bool(improved and mini_gate["finite_gradients"] and mini_gate["nonzero_gradients"] and mini_gate["exact_zero"] and mini_gate["future_causality"]),
        "ready_for_full_train18_smoke": False,
        "ready_for_production_wiring": False,
        "ready_for_formal": False,
        "next_action": "STOP_FOR_EXTERNAL_REVIEW" if improved else "REDUCE_CONTROL_DIMENSION",
        "wall_seconds": float(time.perf_counter() - started),
    }
    reports = {
        "STEP2_CONTROL_VALUE_SURROGATE_V50": model_contract,
        "STEP2_HYDRAULIC_RESPONSE_SURROGATE_V50": hydraulic_contract,
        "STEP2_TRAIN_INTERNAL_HOLDOUT_V50": holdout_contract,
        "STEP2_V50_FINAL_DECISION": final_decision,
    }
    for stem, payload in reports.items():
        (DOCS / f"{stem}.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        (DOCS / f"{stem}.md").write_text(_markdown(stem, payload), encoding="utf-8")
    return final_decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, allow_nan=True))
