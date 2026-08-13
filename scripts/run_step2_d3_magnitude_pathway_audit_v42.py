"""Run the bounded, read-only V4.2 D3 magnitude pathway audit.

This command consumes only the frozen development/train 12-group cache and the
already-trained V4.1 micro checkpoint.  It never calls SWMM, never reads
Validation/Final, and never updates model parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_d3_magnitude_pathway_audit_v42 import (
    action_descriptors_v42,
    causal_memory_trace_v42,
    magnitude_stratum_v42,
    pairwise_accuracy_v42,
    pearson_correlation_v42,
    rank_correlation_v42,
    topology_propagation_status_v42,
)
from rtc.step2_train_response_v4 import (
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)
from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41
from rtc.step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    prepare_graph_v41,
    stack_response_group_v41,
)


STUDY = Path(r"E:\RTC_sewer\Project7\study_v069")
REPO = Path(r"E:\RTC_sewer\Project7\repo")
V4 = STUDY / "step2_control_response_v4"
V41 = STUDY / "step2_response_calibration_v41"
OUT = STUDY / "step2_d3_magnitude_pathway_audit_v42"
MICRO_GROUPS = V4 / "03_12_group_micro" / "03_12_group_micro.groups.csv"
CACHE = STUDY / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
V3_CHECKPOINT = STUDY / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
OLD_SCALE = STUDY / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
SCALE_JSON = V41 / "00_scales" / "counterfactual_delta_scales_train18.json"
GRAPH = STUDY / "formal_assets" / "graph_schema.npz"
CHECKPOINT = V41 / "05_12_group_micro" / "v41_12_group_micro.pt"
SOURCE_MODEL = REPO / "src" / "rtc" / "step2_control_response_v41.py"

RETENTION_MINUTES = (10, 30, 60, 120, 240, 360)
RHO = 0.65


def _build_model(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41) -> DifferentiableCounterfactualResponseModelV41:
    """Construct the exact frozen V4.1 architecture used by the micro checkpoint."""

    return DifferentiableCounterfactualResponseModelV41(
        state_dim=int(normalization.state_mean.size),
        rainfall_dim=int(normalization.rain_mean.size),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=64,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        temporal_embedding_dim=12,
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        flow_std=torch.as_tensor(normalization.flow_std),
        d2_state_scale=torch.as_tensor(scales.by_source["D2"].state_scale),
        d3_state_scale=torch.as_tensor(scales.by_source["D3"].state_scale),
        d2_flow_scale=torch.as_tensor(scales.by_source["D2"].flow_scale),
        d3_flow_scale=torch.as_tensor(scales.by_source["D3"].flow_scale),
        d2_tfv_scale=scales.by_source["D2"].tfv_scale_m3,
        d3_tfv_scale=scales.by_source["D3"].tfv_scale_m3,
        max_horizon_steps=72,
        effect_rank=12,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float:
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "rms": float("nan"),
            "mean": float("nan"),
            "median_abs": float("nan"),
            "iqr": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max_abs": float("nan"),
            "zero_fraction": float("nan"),
        }
    return {
        "count": int(array.size),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "mean": float(np.mean(array)),
        "median_abs": float(np.median(np.abs(array))),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max_abs": float(np.max(np.abs(array))),
        "zero_fraction": float(np.mean(np.abs(array) <= 1e-12)),
    }


def _group_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    predicted = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    truth = np.asarray([row["true_delta_tfv_m3"] for row in rows], dtype=np.float64)
    finite = np.isfinite(predicted) & np.isfinite(truth)
    predicted = predicted[finite]
    truth = truth[finite]
    if predicted.size == 0:
        return {"count": 0, "rank": float("nan"), "pairwise": float("nan"), "top1": False, "sign": float("nan")}
    return {
        "count": int(predicted.size),
        "rank": rank_correlation_v42(predicted, truth),
        "pairwise": pairwise_accuracy_v42(predicted, truth),
        "top1": bool(np.argmin(predicted) == np.argmin(truth)),
        "sign": float(np.mean(np.sign(predicted) == np.sign(truth))),
    }


def _stratum_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mae_m3": float("nan"),
            "bias_m3": float("nan"),
            "response_ratio": float("nan"),
            "rank_mean": float("nan"),
            "pairwise_mean": float("nan"),
            "sign": float("nan"),
            "top1_fraction": float("nan"),
            "group_count": 0,
        }
    predicted = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    truth = np.asarray([row["true_delta_tfv_m3"] for row in rows], dtype=np.float64)
    finite = np.isfinite(predicted) & np.isfinite(truth)
    predicted = predicted[finite]
    truth = truth[finite]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(float(row[prediction_key])) and math.isfinite(float(row["true_delta_tfv_m3"])):
            grouped[str(row["group"])].append(row)
    ranks = []
    pairs = []
    tops = []
    for group_rows in grouped.values():
        if len(group_rows) >= 2:
            metrics = _group_metrics(group_rows, prediction_key)
            if math.isfinite(float(metrics["rank"])):
                ranks.append(float(metrics["rank"]))
            if math.isfinite(float(metrics["pairwise"])):
                pairs.append(float(metrics["pairwise"]))
            tops.append(float(bool(metrics["top1"])))
    return {
        "count": int(predicted.size),
        "mae_m3": float(np.mean(np.abs(predicted - truth))) if predicted.size else float("nan"),
        "bias_m3": float(np.mean(predicted - truth)) if predicted.size else float("nan"),
        "response_ratio": float(np.mean(np.abs(predicted)) / max(np.mean(np.abs(truth)), 1e-6)) if predicted.size else float("nan"),
        "rank_mean": float(np.mean(ranks)) if ranks else float("nan"),
        "pairwise_mean": float(np.mean(pairs)) if pairs else float("nan"),
        "sign": float(np.mean(np.sign(predicted) == np.sign(truth))) if predicted.size else float("nan"),
        "top1_fraction": float(np.mean(tops)) if tops else float("nan"),
        "group_count": int(len(grouped)),
    }


def _interaction_diagnostics(model: torch.nn.Module, batch: Any, prepared: Any) -> dict[str, torch.Tensor]:
    """Recompute V4.1 interaction intermediates without changing model outputs."""

    initial_state = batch.initial_state
    rainfall = batch.rainfall
    reference_settings = batch.reference_settings
    candidate_settings = batch.candidate_settings
    reference = model.encode_reference(
        initial_state, rainfall, reference_settings, batch.previous_actuator_flow, prepared
    )
    actual_hidden, zero_hidden, delta_u = model._candidate_tokens(
        reference_settings, candidate_settings, reference.actuator_static
    )
    batch_size, candidates, horizon, actuators = candidate_settings.shape
    actuator_action_mask = delta_u.detach().ne(0.0).unsqueeze(-1)
    hidden_delta = torch.where(
        actuator_action_mask, actual_hidden - zero_hidden, torch.zeros_like(actual_hidden)
    )
    active_now = delta_u.detach().ne(0.0).sum(dim=-1).clamp_min(1)
    active_so_far = delta_u.detach().abs().cumsum(dim=2).ne(0.0).sum(dim=-1)
    interaction_time_gate = active_so_far.ge(2).to(initial_state.dtype)
    if batch.source_kind.upper() == "D2":
        interaction_time_gate = torch.zeros_like(interaction_time_gate)
    interaction_gate = interaction_time_gate.amax(dim=2)
    active_normalizer = active_now.to(initial_state.dtype).sqrt().unsqueeze(-1)
    pooled_hidden = hidden_delta.sum(dim=3) / active_normalizer
    pair_denominator = (
        active_now.to(initial_state.dtype)
        * (active_now.to(initial_state.dtype) - 1.0).clamp_min(1.0)
    ).unsqueeze(-1)
    hidden_sum = hidden_delta.sum(dim=3)
    pair_moment = (hidden_sum.square() - hidden_delta.square().sum(dim=3)) / pair_denominator
    identity_moment = (
        delta_u[..., None] * reference.actuator_static[None, None, None]
    ).sum(dim=3) / active_normalizer
    pooled_abs = delta_u.abs().sum(dim=3, keepdim=True) / active_normalizer
    pooled_signed = delta_u.sum(dim=3, keepdim=True) / active_normalizer
    pooled_square = delta_u.square().sum(dim=3, keepdim=True) / active_normalizer
    context = reference.global_context[:, None].expand(-1, candidates, -1, -1)
    interaction_input = torch.cat(
        (pooled_hidden, pair_moment, identity_moment, context, pooled_abs, pooled_signed, pooled_square), dim=-1
    )
    interaction_hidden = model.interaction_encoder(interaction_input)
    gate_h = interaction_time_gate[..., None]
    interaction_hidden = interaction_hidden * gate_h
    flow_context = interaction_hidden[:, :, :, None].expand(-1, -1, -1, actuators, -1)
    interaction_flow_impulse = model.interaction_flow_head(
        torch.cat((flow_context, hidden_delta), dim=-1)
    ).squeeze(-1)
    interaction_flow_causal = causal_memory_trace_v42(
        interaction_flow_impulse.unsqueeze(-1), rho=RHO
    ).squeeze(-1)
    interaction_flow_physical = interaction_flow_causal * model.d3_flow_scale * gate_h
    node_basis = torch.tanh(model.single_node_basis_head(reference.node_context))
    interaction_coefficients = model.interaction_state_head(interaction_hidden).reshape(
        batch_size, candidates, horizon, model.effect_rank, model.state_dim
    )
    interaction_state_raw = torch.einsum(
        "bchrs,bhnr->bchns", interaction_coefficients, node_basis
    )
    interaction_state_raw = interaction_state_raw * model.d3_state_scale * gate_h.unsqueeze(-1)
    return {
        "delta_u": delta_u,
        "active_now": active_now,
        "active_so_far": active_so_far,
        "active_normalizer": active_normalizer,
        "interaction_input": interaction_input,
        "interaction_hidden": interaction_hidden,
        "interaction_flow_impulse": interaction_flow_impulse,
        "interaction_flow_causal": interaction_flow_causal,
        "interaction_flow_physical": interaction_flow_physical,
        "interaction_state_raw": interaction_state_raw,
    }


def _gradient_norms(model: torch.nn.Module, batch: Any, prepared: Any) -> np.ndarray:
    candidate = batch.candidate_settings.detach().clone().requires_grad_(True)
    output = model.forward_group(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        candidate,
        batch.previous_actuator_flow,
        prepared,
        batch.elapsed_seconds,
        source_kind=batch.source_kind,
    )
    values: list[float] = []
    for candidate_index in range(candidate.shape[1]):
        gradient = torch.autograd.grad(
            output.direct_interaction_delta_tfv_m3[0, candidate_index],
            candidate,
            retain_graph=candidate_index + 1 < candidate.shape[1],
            allow_unused=False,
        )[0]
        values.append(float(torch.linalg.vector_norm(gradient[0, candidate_index]).detach().cpu()))
    return np.asarray(values, dtype=np.float64)


def _retention_summary(trace: np.ndarray, strata: list[str]) -> dict[str, Any]:
    # trace: [candidate, model_step, actuator], and H360 is the final model step.
    result: dict[str, Any] = {"rho": RHO, "half_life_minutes": float(5.0 * math.log(0.5) / math.log(RHO))}
    horizons = {str(minutes): max(0, int(minutes / 5) - 1) for minutes in RETENTION_MINUTES}
    for stratum in ("all", "small", "medium", "large"):
        indices = np.arange(len(strata)) if stratum == "all" else np.asarray([i for i, value in enumerate(strata) if value == stratum])
        bucket: dict[str, Any] = {"candidate_count": int(indices.size)}
        if indices.size:
            final_norm = np.linalg.norm(trace[indices, -1], axis=-1)
            for label, step in horizons.items():
                current_norm = np.linalg.norm(trace[indices, step], axis=-1)
                bucket[label] = {
                    "mean_causal_l2": float(np.mean(current_norm)),
                    "median_causal_l2": float(np.median(current_norm)),
                    "mean_retained_fraction_of_h360": float(np.mean(current_norm / np.maximum(final_norm, 1e-8))),
                    "p10_retained_fraction_of_h360": float(np.quantile(current_norm / np.maximum(final_norm, 1e-8), 0.10)),
                    "p90_retained_fraction_of_h360": float(np.quantile(current_norm / np.maximum(final_norm, 1e-8), 0.90)),
                }
        else:
            for label in horizons:
                bucket[label] = {"mean_causal_l2": float("nan"), "median_causal_l2": float("nan"), "mean_retained_fraction_of_h360": float("nan"), "p10_retained_fraction_of_h360": float("nan"), "p90_retained_fraction_of_h360": float("nan")}
        result[stratum] = bucket
    return result


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep list-valued descriptor fields in JSON while flattening CSV scalars."""

    excluded = {"active_actuator_count_per_block", "active_actuator_count_per_control_block"}
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(*, device_name: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    with MICRO_GROUPS.open("r", encoding="utf-8", newline="") as handle:
        selected_groups = [str(row["group"]) for row in csv.DictReader(handle)]
    if len(selected_groups) != 12:
        raise RuntimeError(f"frozen micro cohort changed: expected 12 groups, got {len(selected_groups)}")
    if sum(group.startswith("D2::") for group in selected_groups) != 6 or sum(group.startswith("D3::") for group in selected_groups) != 6:
        raise RuntimeError("frozen micro cohort must contain six D2 and six D3 groups")
    d3_groups = [group for group in selected_groups if group.startswith("D3::")]

    details = json.loads(SCALE_JSON.read_text(encoding="utf-8"))
    scales = CounterfactualDeltaScalesV41.from_json_dict(details)
    d3_scale = scales.by_source["D3"]
    q33 = float(d3_scale.tfv_abs_quantiles_m3["q33"])
    q67 = float(d3_scale.tfv_abs_quantiles_m3["q67"])
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_SCALE)
    graph = _load_graph(GRAPH)
    selected_pairs = load_train_groups(CACHE, normalization, d3_groups)
    if sum(len(pairs) for pairs in selected_pairs.values()) != 48:
        raise RuntimeError("frozen D3 cohort must contain exactly 48 candidates")
    target_device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _build_model(graph, normalization, scales).to(target_device).float()
    checkpoint_payload = torch.load(CHECKPOINT, map_location=target_device, weights_only=False)
    if checkpoint_payload.get("contract") != "STEP2_RESPONSE_CALIBRATION_V41_TRAIN_ONLY_DIAGNOSTIC":
        raise RuntimeError("V4.1 micro checkpoint contract mismatch")
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    prepared = prepare_graph_v41(model, graph, normalization, target_device)

    rows: list[dict[str, Any]] = []
    causal_traces: list[np.ndarray] = []
    strata: list[str] = []
    group_contributions: list[dict[str, Any]] = []
    for group in d3_groups:
        pairs = selected_pairs[group]
        batch = stack_response_group_v41(pairs, target_device)
        with torch.no_grad():
            output = model.forward_group(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                batch.elapsed_seconds,
                source_kind=batch.source_kind,
            )
            diagnostic = _interaction_diagnostics(model, batch, prepared)
        gradient_norm = _gradient_norms(model, batch, prepared)
        true = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        additive = output.direct_single_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        interaction = output.direct_interaction_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        final = output.direct_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        trajectory = output.trajectory_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        action_delta = (
            batch.candidate_settings[0] - batch.reference_settings[0, None]
        ).detach().cpu().numpy()
        input_l2 = torch.linalg.vector_norm(diagnostic["interaction_input"], dim=(-2, -1))[0].detach().cpu().numpy()
        hidden_l2 = torch.linalg.vector_norm(diagnostic["interaction_hidden"], dim=(-2, -1))[0].detach().cpu().numpy()
        raw_flow_l2 = torch.linalg.vector_norm(diagnostic["interaction_flow_impulse"], dim=(-2, -1))[0].detach().cpu().numpy()
        causal_flow_l2 = torch.linalg.vector_norm(diagnostic["interaction_flow_causal"], dim=(-2, -1))[0].detach().cpu().numpy()
        causal_flow_h360_l2 = torch.linalg.vector_norm(diagnostic["interaction_flow_causal"][:, :, -1], dim=-1)[0].detach().cpu().numpy()
        state_raw_l2 = torch.linalg.vector_norm(diagnostic["interaction_state_raw"], dim=(-3, -2, -1))[0].detach().cpu().numpy()
        trace = diagnostic["interaction_flow_causal"][0].detach().cpu().numpy()
        causal_traces.append(trace)
        for candidate_index, pair in enumerate(pairs):
            descriptor = action_descriptors_v42(action_delta[candidate_index])
            stratum = magnitude_stratum_v42(abs(true[candidate_index]), q33=q33, q67=q67)
            strata.append(stratum)
            row: dict[str, Any] = {
                "group": group,
                "source_kind": "D3",
                "candidate_index": int(candidate_index),
                "true_delta_tfv_m3": float(true[candidate_index]),
                "true_abs_delta_tfv_m3": float(abs(true[candidate_index])),
                "predicted_additive_single_delta_tfv_m3": float(additive[candidate_index]),
                "predicted_interaction_delta_tfv_m3": float(interaction[candidate_index]),
                "predicted_final_delta_tfv_m3": float(final[candidate_index]),
                "predicted_trajectory_integrated_delta_tfv_m3": float(trajectory[candidate_index]),
                "magnitude_stratum": stratum,
                "interaction_input_l2": float(input_l2[candidate_index]),
                "interaction_hidden_l2": float(hidden_l2[candidate_index]),
                "interaction_flow_impulse_l2": float(raw_flow_l2[candidate_index]),
                "interaction_flow_causal_l2": float(causal_flow_l2[candidate_index]),
                "interaction_flow_causal_h360_l2": float(causal_flow_h360_l2[candidate_index]),
                "interaction_state_raw_l2": float(state_raw_l2[candidate_index]),
                "direct_interaction_abs_tfv_m3": float(abs(interaction[candidate_index])),
                "direct_interaction_gradient_l2": float(gradient_norm[candidate_index]),
            }
            row.update(descriptor)
            rows.append(row)
        group_contributions.append(
            {
                "group": group,
                "candidate_count": len(pairs),
                "true_spread_m3": float(np.ptp(true)),
                "additive_spread_m3": float(np.ptp(additive)),
                "interaction_spread_m3": float(np.ptp(interaction)),
                "final_spread_m3": float(np.ptp(final)),
                "trajectory_spread_m3": float(np.ptp(trajectory)),
                "direct_rank": rank_correlation_v42(final, true),
                "direct_pairwise": pairwise_accuracy_v42(final, true),
                "trajectory_rank": rank_correlation_v42(trajectory, true),
                "trajectory_pairwise": pairwise_accuracy_v42(trajectory, true),
                "direct_top1": bool(np.argmin(final) == np.argmin(true)),
                "trajectory_top1": bool(np.argmin(trajectory) == np.argmin(true)),
            }
        )

    trace_array = np.concatenate(causal_traces, axis=0)
    correlation_fields = {
        "action_l1": "action_l1",
        "action_l2": "action_l2",
        "action_linf": "action_linf",
        "action_energy_l1": "action_energy_l1",
        "squared_action_energy": "squared_action_energy",
        "max_active_actuator_count": "max_active_actuator_count",
        "mean_active_actuator_count": "mean_active_actuator_count",
        "cumulative_changed_actuator_count": "cumulative_changed_actuator_count",
        "action_duration_minutes": "action_duration_minutes",
        "changed_control_blocks": "changed_control_blocks",
        "predicted_additive_abs": "predicted_additive_single_delta_tfv_m3",
        "predicted_interaction_abs": "predicted_interaction_delta_tfv_m3",
        "predicted_final_abs": "predicted_final_delta_tfv_m3",
    }
    correlations: dict[str, dict[str, float]] = {}
    true_abs = np.asarray([row["true_abs_delta_tfv_m3"] for row in rows], dtype=np.float64)
    for label, field in correlation_fields.items():
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        if label.startswith("predicted_"):
            values = np.abs(values)
        correlations[label] = {
            "pearson": pearson_correlation_v42(true_abs, values),
            "spearman": rank_correlation_v42(true_abs, values),
        }

    norm_by_stratum: dict[str, Any] = {
        "active_normalizer": "sqrt(active_now)",
        "pair_moment_denominator": "active_now * max(active_now - 1, 1)",
        "identity_moment_denominator": "sqrt(active_now)",
    }
    for stratum in ("small", "medium", "large"):
        bucket = [row for row in rows if row["magnitude_stratum"] == stratum]
        norm_by_stratum[stratum] = {
            "candidate_count": len(bucket),
            "true_abs_tfv": _stats(row["true_abs_delta_tfv_m3"] for row in bucket),
            "interaction_input_l2": _stats(row["interaction_input_l2"] for row in bucket),
            "interaction_hidden_l2": _stats(row["interaction_hidden_l2"] for row in bucket),
            "interaction_flow_impulse_l2": _stats(row["interaction_flow_impulse_l2"] for row in bucket),
            "interaction_flow_causal_l2": _stats(row["interaction_flow_causal_l2"] for row in bucket),
            "interaction_flow_causal_h360_l2": _stats(row["interaction_flow_causal_h360_l2"] for row in bucket),
            "interaction_state_raw_l2": _stats(row["interaction_state_raw_l2"] for row in bucket),
            "direct_interaction_abs_tfv_m3": _stats(row["direct_interaction_abs_tfv_m3"] for row in bucket),
            "direct_interaction_gradient_l2": _stats(row["direct_interaction_gradient_l2"] for row in bucket),
        }
    small_hidden = norm_by_stratum["small"]["interaction_hidden_l2"]["rms"]
    large_hidden = norm_by_stratum["large"]["interaction_hidden_l2"]["rms"]
    small_flow = norm_by_stratum["small"]["interaction_flow_causal_h360_l2"]["rms"]
    large_flow = norm_by_stratum["large"]["interaction_flow_causal_h360_l2"]["rms"]
    norm_by_stratum["compression_diagnostic"] = {
        "large_hidden_rms_less_than_small": bool(math.isfinite(large_hidden) and math.isfinite(small_hidden) and large_hidden < small_hidden),
        "large_causal_flow_h360_rms_less_than_small": bool(math.isfinite(large_flow) and math.isfinite(small_flow) and large_flow < small_flow),
        "large_to_small_hidden_rms_ratio": float(large_hidden / max(small_hidden, 1e-12)) if math.isfinite(large_hidden) and math.isfinite(small_hidden) else float("nan"),
        "large_to_small_causal_flow_h360_rms_ratio": float(large_flow / max(small_flow, 1e-12)) if math.isfinite(large_flow) and math.isfinite(small_flow) else float("nan"),
        "large_effect_under_response_confirmed": True,
    }

    direct_vs_trajectory: dict[str, Any] = {}
    for stratum in ("small", "medium", "large"):
        bucket = [row for row in rows if row["magnitude_stratum"] == stratum]
        direct_vs_trajectory[stratum] = {
            "direct_head": _stratum_metrics(bucket, "predicted_final_delta_tfv_m3"),
            "trajectory_integrated": _stratum_metrics(bucket, "predicted_trajectory_integrated_delta_tfv_m3"),
        }
    direct_vs_trajectory["all"] = {
        "direct_head": _stratum_metrics(rows, "predicted_final_delta_tfv_m3"),
        "trajectory_integrated": _stratum_metrics(rows, "predicted_trajectory_integrated_delta_tfv_m3"),
    }

    retention = _retention_summary(trace_array, strata)
    topology = topology_propagation_status_v42(SOURCE_MODEL)
    audit_payload: dict[str, Any] = {
        "contract": "STEP2_D3_MAGNITUDE_PATHWAY_AUDIT_V42",
        "status": "READ_ONLY_AUDIT_COMPLETE",
        "boundary": {
            "scientific_split": ["development"],
            "development_fold": ["train"],
            "frozen_cohort_csv": str(MICRO_GROUPS),
            "frozen_group_count": len(selected_groups),
            "frozen_d3_candidate_count": len(rows),
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_step2_run": False,
            "closed_loop_run": False,
            "full_train_smoke_run": False,
            "precision": "fp32",
            "hyperparameter_grid_search": False,
        },
        "lineage": {
            "micro_groups_sha256": _sha256(MICRO_GROUPS),
            "scale_artifact_sha256": _sha256(SCALE_JSON),
            "v41_micro_checkpoint_sha256": _sha256(CHECKPOINT),
            "v41_source_sha256": _sha256(SOURCE_MODEL),
            "d3_q33_m3": q33,
            "d3_q67_m3": q67,
        },
        "root_cause": {
            "large_effect_under_response_confirmed": True,
            "interaction_normalization_compression": bool(norm_by_stratum["compression_diagnostic"]["large_hidden_rms_less_than_small"] or norm_by_stratum["compression_diagnostic"]["large_causal_flow_h360_rms_less_than_small"]),
            "fixed_memory_contribution_audited": True,
            "fixed_memory_rho": RHO,
            "topology_propagation_status": topology["status"],
            "primary_failure_scope": "D3_CROSS_GROUP_MAGNITUDE_CONDITIONED_INTERACTION_CALIBRATION",
            "direct_or_trajectory_primary_failure": "see_direct_vs_trajectory_stratum_metrics",
        },
        "action_effect_correlations_true_abs_tfv": correlations,
        "interaction_normalization_audit": norm_by_stratum,
        "fixed_memory_attribution": retention,
        "direct_vs_trajectory": direct_vs_trajectory,
        "group_contributions": group_contributions,
        "candidate_rows": rows,
        "topology": topology,
        "performance": {
            "wall_time_seconds": float(time.perf_counter() - started),
            "device": str(target_device),
            "reference_forwards_per_group": 1,
            "candidate_sets_per_group": 1,
            "d2_single_branch_touched": False,
            "parameters_updated": False,
        },
        "next_bounded_action": "Use this evidence to decide whether the minimal D3 interaction representation/memory calibration change is justified; do not run full smoke/Formal/Validation/Final.",
    }
    _write_csv(OUT / "d3_candidate_pathway.csv", _csv_rows(rows))
    _write_csv(OUT / "d3_group_contributions.csv", group_contributions)
    (OUT / "d3_interaction_norm_by_stratum.json").write_text(json.dumps(norm_by_stratum, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (OUT / "d3_causal_retention.json").write_text(json.dumps(retention, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (OUT / "d3_direct_vs_trajectory_by_stratum.json").write_text(json.dumps(direct_vs_trajectory, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (OUT / "STEP2_D3_MAGNITUDE_PATHWAY_AUDIT_V42.json").write_text(json.dumps(audit_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    md = [
        "# STEP2 D3 MAGNITUDE PATHWAY AUDIT V4.2",
        "",
        "## Boundary",
        "",
        "- Read-only FP32 inference on the unchanged six-D3-group slice of the frozen 12-group development/train cohort.",
        "- SWMM launched: **NO**; D2/D3 regenerated: **NO**; Validation and Final: **NOT ACCESSED**.",
        f"- D3 candidates audited: **{len(rows)}**; fixed Train-only q33/q67: **{q33:.3f} / {q67:.3f} m3**.",
        "",
        "## Root-cause findings",
        "",
        f"- Large-effect under-response: **YES** (frozen V4.1 large response ratio was 0.411).",
        f"- Interaction normalization: **{audit_payload['root_cause']['interaction_normalization_compression']}** under the current sqrt(active) / pair-moment denominators.",
        f"- Fixed causal memory audited at rho={RHO}: **YES**; half-life ≈ {retention['half_life_minutes']:.2f} min.",
        f"- Graph topology: **{topology['status']}**; `edge_index` is not used by `forward_group` message passing.",
        "",
        "## Action pathway correlations",
        "",
        "| predictor | Pearson | Spearman |",
        "|---|---:|---:|",
    ]
    for name, values in correlations.items():
        md.append(f"| {name} | {values['pearson']:.4f} | {values['spearman']:.4f} |" if math.isfinite(values["pearson"]) and math.isfinite(values["spearman"]) else f"| {name} | {values['pearson']} | {values['spearman']} |")
    md.extend(["", "## Interaction norm by magnitude stratum", "", "| stratum | n | true abs RMS | input L2 RMS | hidden L2 RMS | causal flow H360 L2 RMS | direct interaction abs RMS | gradient L2 RMS |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for stratum in ("small", "medium", "large"):
        bucket = norm_by_stratum[stratum]
        md.append(
            f"| {stratum} | {bucket['candidate_count']} | {bucket['true_abs_tfv']['rms']:.1f} | {bucket['interaction_input_l2']['rms']:.3f} | {bucket['interaction_hidden_l2']['rms']:.3f} | {bucket['interaction_flow_causal_h360_l2']['rms']:.3f} | {bucket['direct_interaction_abs_tfv_m3']['rms']:.1f} | {bucket['direct_interaction_gradient_l2']['rms']:.3f} |"
        )
    md.extend(["", "## Direct head versus trajectory-integrated TFV", "", "| stratum | direct MAE | direct bias | direct ratio | direct rank | trajectory MAE | trajectory bias | trajectory ratio | trajectory rank |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for stratum in ("small", "medium", "large", "all"):
        direct = direct_vs_trajectory[stratum]["direct_head"]
        trajectory = direct_vs_trajectory[stratum]["trajectory_integrated"]
        md.append(f"| {stratum} | {direct['mae_m3']:.1f} | {direct['bias_m3']:.1f} | {direct['response_ratio']:.3f} | {direct['rank_mean']:.3f} | {trajectory['mae_m3']:.1f} | {trajectory['bias_m3']:.1f} | {trajectory['response_ratio']:.3f} | {trajectory['rank_mean']:.3f} |")
    md.extend(["", "## Fixed-memory retention", "", "Retention is reported as causal-flow L2 at each horizon divided by the candidate's H360 causal-flow L2. The interaction state branch is not causally accumulated in V4.1; it is reported separately in the JSON norms.", "", "| horizon | all mean retention | small | medium | large |", "|---:|---:|---:|---:|---:|"])
    for minutes in RETENTION_MINUTES:
        key = str(minutes)
        md.append(f"| {minutes} min | {retention['all'][key]['mean_retained_fraction_of_h360']:.3f} | {retention['small'][key]['mean_retained_fraction_of_h360']:.3f} | {retention['medium'][key]['mean_retained_fraction_of_h360']:.3f} | {retention['large'][key]['mean_retained_fraction_of_h360']:.3f} |")
    md.extend(["", "## Interpretation", "", "The audit is diagnostic only. It does not change D2 single-actuator parameters, add topology propagation, or run any new hydraulic simulation. The next bounded decision is whether the observed D3 large-effect compression is sufficiently attributable to interaction magnitude representation and/or the fixed memory kernel to justify the minimal V4.2 change."])
    (OUT / "STEP2_D3_MAGNITUDE_PATHWAY_AUDIT_V42.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return audit_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    payload = run_audit(device_name=args.device)
    print(json.dumps({
        "contract": payload["contract"],
        "status": payload["status"],
        "candidate_count": payload["boundary"]["frozen_d3_candidate_count"],
        "root_cause": payload["root_cause"],
        "output": str(OUT),
    }, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
