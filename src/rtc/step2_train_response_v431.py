"""V4.3.1 training-semantics utilities.

This module deliberately keeps the V4.3 architecture unchanged.  It only makes
the three optimisation phases semantically explicit: reference hydraulic fit,
D2 single-actuator fit, and D3 interaction-residual fit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v43 import (
    DifferentiableCounterfactualResponseModelV43,
    interaction_parameter_names,
    parameter_sha256,
    reference_parameter_names,
    set_trainable_phase,
    single_parameter_names,
)
from .step2_train_response_v4 import ResponseNormalizationV4, ResponsePairV4
from .step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    CounterfactualGroupResponseV41,
    ResponseGroupBatchV41,
    ResponseLossWeightsV41,
    evaluate_response_groups_v41,
    prepare_graph_v41,
    stack_response_group_v41,
    tfv_loss_components_v41,
)


TRAINING_CONTRACT_V431 = "STEP2_TRAINING_SEMANTICS_V431_REFERENCE_D2_D3_PHASED_TRAIN_ONLY"


def _tensor_scale(value: np.ndarray | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype).clamp_min(1e-6)


def _reference_terms(
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    normalization: ResponseNormalizationV4,
) -> dict[str, torch.Tensor]:
    state_scale = _tensor_scale(normalization.state_std, output.reference_states_physical)
    flow_scale = _tensor_scale(normalization.flow_std, output.reference_flows_physical)
    return {
        "reference_state": F.smooth_l1_loss(
            (output.reference_states_physical - batch.true_reference_states_physical) / state_scale,
            torch.zeros_like(output.reference_states_physical),
        ),
        "reference_flow": F.smooth_l1_loss(
            (output.reference_flows_physical - batch.true_reference_flows_physical) / flow_scale,
            torch.zeros_like(output.reference_flows_physical),
        ),
    }


def _effect_terms(
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    source_scales: Any,
    *,
    magnitude_calibration: bool,
    include_interaction_energy: bool,
) -> dict[str, torch.Tensor]:
    state_scale = _tensor_scale(source_scales.state_scale, output.delta_states_physical)
    flow_scale = _tensor_scale(source_scales.flow_scale, output.delta_flows_physical)
    state_error = (output.delta_states_physical - batch.true_delta_states_physical) / state_scale
    flow_error = (output.delta_flows_physical - batch.true_delta_flows_physical) / flow_scale
    terms: dict[str, torch.Tensor] = {
        "delta_state": F.smooth_l1_loss(state_error, torch.zeros_like(state_error)),
        "delta_flow": F.smooth_l1_loss(flow_error, torch.zeros_like(flow_error)),
    }
    tfv = tfv_loss_components_v41(
        output.direct_delta_tfv_m3,
        output.trajectory_delta_tfv_m3,
        batch.true_delta_tfv_m3,
        source_scale=torch.as_tensor(
            source_scales.tfv_scale_m3,
            device=output.direct_delta_tfv_m3.device,
            dtype=output.direct_delta_tfv_m3.dtype,
        ),
        magnitude_calibration=bool(magnitude_calibration),
        magnitude_q33=source_scales.tfv_abs_quantiles_m3.get("q33"),
        magnitude_q67=source_scales.tfv_abs_quantiles_m3.get("q67"),
    )
    terms.update(
        {
            "direct_TFV": tfv["absolute_direct"],
            "centered_TFV": tfv["group_centered_direct"],
            "trajectory_TFV": tfv["authoritative_trajectory"],
            "consistency": tfv["direct_trajectory_consistency"],
            "ranking": tfv["ranking"],
            "magnitude_calibration": tfv["log_magnitude_calibration"],
        }
    )
    if include_interaction_energy:
        terms["interaction_energy"] = (
            output.interaction_delta_states_physical.div(state_scale).square().mean()
            + output.interaction_delta_flows_physical.div(flow_scale).square().mean()
        )
    return terms


def _weighted_total(
    terms: dict[str, torch.Tensor], weights: ResponseLossWeightsV41, *, include_reference: bool
) -> torch.Tensor:
    total = terms["delta_state"].new_zeros(())
    if include_reference:
        total = total + weights.reference_state * terms["reference_state"]
        total = total + weights.reference_flow * terms["reference_flow"]
    total = total + weights.delta_state * terms["delta_state"]
    total = total + weights.delta_flow * terms["delta_flow"]
    total = total + weights.direct_tfv * terms["direct_TFV"]
    total = total + weights.centered_tfv * terms["centered_TFV"]
    total = total + weights.trajectory_tfv * terms["trajectory_TFV"]
    total = total + weights.consistency * terms["consistency"]
    total = total + weights.ranking * terms["ranking"]
    if "interaction_energy" in terms:
        total = total + weights.interaction_energy * terms["interaction_energy"]
    if "magnitude_calibration" in terms:
        total = total + weights.magnitude_calibration * terms["magnitude_calibration"]
    return total


def reference_phase_loss_v431(
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    normalization: ResponseNormalizationV4,
    *,
    weights: ResponseLossWeightsV41 | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reference-only hydraulic fit; no counterfactual target is consulted."""

    w = weights or ResponseLossWeightsV41()
    terms = _reference_terms(output, batch, normalization)
    total = w.reference_state * terms["reference_state"] + w.reference_flow * terms["reference_flow"]
    return total, terms | {"loss": total}


def d2_single_phase_loss_v431(
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    source_scales: Any,
    normalization: ResponseNormalizationV4,
    *,
    weights: ResponseLossWeightsV41 | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """D2 single-actuator response fit with reference and interaction losses absent."""

    del normalization
    w = weights or ResponseLossWeightsV41()
    terms = _effect_terms(
        output,
        batch,
        source_scales,
        magnitude_calibration=False,
        include_interaction_energy=False,
    )
    total = _weighted_total(terms, w, include_reference=False)
    return total, terms | {"loss": total}


def d3_interaction_phase_loss_v431(
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    source_scales: Any,
    normalization: ResponseNormalizationV4,
    *,
    weights: ResponseLossWeightsV41 | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """D3 total-effect fit: frozen additive D2 effect plus trainable residual."""

    del normalization
    w = weights or ResponseLossWeightsV41()
    terms = _effect_terms(
        output,
        batch,
        source_scales,
        magnitude_calibration=batch.source_kind.upper() == "D3",
        include_interaction_energy=True,
    )
    total = _weighted_total(terms, w, include_reference=False)
    return total, terms | {"loss": total}


@dataclass(frozen=True)
class PhaseResultV431:
    phase: str
    selected_groups: tuple[str, ...]
    raw_reference_group_count: int
    unique_reference_group_count: int
    epochs_completed: int
    history: list[dict[str, Any]]
    parameter_sha256: str
    profile_seconds: dict[str, float]


def validate_fresh_parent_lineage_v431(
    immutable_parent: str | Path,
    tiny_parent: str | Path,
    micro_parent: str | Path,
) -> bool:
    """Return true only when both experiments point at the same immutable parent."""

    parent = Path(immutable_parent).resolve()
    tiny = Path(tiny_parent).resolve()
    micro = Path(micro_parent).resolve()
    return tiny == parent and micro == parent and tiny != Path("tiny.pt").resolve()


def _phase_loss(
    phase: str,
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    key = phase.lower()
    if key == "reference":
        return reference_phase_loss_v431(output, batch, normalization)
    if key == "d2":
        return d2_single_phase_loss_v431(output, batch, scales.by_source[batch.source_kind], normalization)
    if key == "d3":
        return d3_interaction_phase_loss_v431(output, batch, scales.by_source[batch.source_kind], normalization)
    raise ValueError("phase must be reference, d2, or d3")


def train_phase_v431(
    *,
    model: torch.nn.Module,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    phase: str,
    out_path: str | Path,
    epochs: int,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 42,
) -> PhaseResultV431:
    """Run exactly one explicit phase and save its checkpoint/history."""

    if not grouped_pairs:
        raise ValueError("V4.3.1 phase requires at least one group")
    key = phase.lower()
    selected = set_trainable_phase(model, key)
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target_device).float()
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    batches = {group: stack_response_group_v41(pairs, target_device) for group, pairs in sorted(grouped_pairs.items())}
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"phase {key} selected no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    profile = {"forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        order = sorted(batches)
        np.random.default_rng(seed + epoch).shuffle(order)
        model.train()
        losses: list[float] = []
        gradients: list[float] = []
        component_sums: dict[str, float] = {}
        for group in order:
            batch = batches[group]
            tick = time.perf_counter()
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=batch.source_kind,
            )
            loss, terms = _phase_loss(key, output, batch, normalization, scales)
            profile["forward_seconds"] += time.perf_counter() - tick
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V4.3.1 {key} loss at epoch {epoch}, group {group}")
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(trainable, 5.0))
            profile["backward_seconds"] += time.perf_counter() - tick
            if not np.isfinite(gradient):
                raise FloatingPointError(f"non-finite V4.3.1 {key} gradient at epoch {epoch}, group {group}")
            tick = time.perf_counter()
            optimizer.step()
            profile["optimizer_seconds"] += time.perf_counter() - tick
            losses.append(float(loss.detach()))
            gradients.append(gradient)
            for name, value in terms.items():
                if name != "loss":
                    component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
        metrics, _ = evaluate_response_groups_v41(
            model=model, grouped_pairs=grouped_pairs, prepared=prepared,
            device=target_device, batches=batches,
        )
        d2 = [m for m in metrics if m["source_kind"] == "D2"]
        d3 = [m for m in metrics if m["source_kind"] == "D3"]
        row: dict[str, Any] = {
            "epoch": epoch,
            "phase": key,
            "loss": float(np.mean(losses)),
            "gradient_norm": float(np.mean(gradients)),
            "rank": float(np.nanmean([m["rank"] for m in metrics])),
            "pairwise": float(np.nanmean([m["pairwise"] for m in metrics])),
            "d2_rank": float(np.nanmean([m["rank"] for m in d2])) if d2 else float("nan"),
            "d3_rank": float(np.nanmean([m["rank"] for m in d3])) if d3 else float("nan"),
            "d3_pairwise": float(np.nanmean([m["pairwise"] for m in d3])) if d3 else float("nan"),
        }
        row.update({name: value / len(order) for name, value in component_sums.items()})
        history.append(row)
    profile["wall_time_seconds"] = time.perf_counter() - started
    model.eval()
    metrics, contributions = evaluate_response_groups_v41(
        model=model, grouped_pairs=grouped_pairs, prepared=prepared,
        device=target_device, batches=batches,
    )
    output_path = Path(out_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": TRAINING_CONTRACT_V431,
            "phase": key,
            "model_state_dict": model.state_dict(),
            "seed": int(seed),
            "precision": "fp32",
            "selected_parameter_names": sorted(selected),
            "group_metrics": metrics,
            "candidate_contributions": contributions,
        },
        output_path,
    )
    output_path.with_suffix(output_path.suffix + ".history.json").write_text(
        json.dumps({"contract": TRAINING_CONTRACT_V431, "phase": key, "history": history, "profile_seconds": profile}, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return PhaseResultV431(
        phase=key,
        selected_groups=tuple(sorted(grouped_pairs)),
        raw_reference_group_count=len(grouped_pairs),
        unique_reference_group_count=len(grouped_pairs),
        epochs_completed=len(history),
        history=history,
        parameter_sha256=parameter_sha256(model, selected),
        profile_seconds=profile,
    )


def current_state_diagnostics_v431(
    pair: Any,
    normalization: Any,
    graph: Any | None = None,
) -> dict[str, float | str]:
    """Compute causal diagnostics from the current ``initial_state`` only."""

    initial = np.asarray(pair.reference["initial_state"], dtype=np.float64)
    mean = np.asarray(normalization.state_mean, dtype=np.float64)
    std = np.maximum(np.asarray(normalization.state_std, dtype=np.float64), 1e-6)
    physical = initial * std + mean
    depth = physical[..., 0]
    flood = physical[..., 2] if physical.shape[-1] > 2 else np.zeros_like(depth)
    node_volume = physical[..., 3] if physical.shape[-1] > 3 else np.zeros_like(depth)
    if graph is not None and "storage_capacity_m3" in tuple(graph.static_node_feature_names):
        feature_names = tuple(graph.static_node_feature_names)
        storage_mask = np.asarray(graph.static_node_features)[:, feature_names.index("is_storage")] > 0.5
        capacity = np.asarray(graph.static_node_features)[:, feature_names.index("storage_capacity_m3")]
        valid = storage_mask & (capacity > 1e-6)
        storage_utilization = float(np.mean(node_volume[..., valid] / capacity[valid])) if valid.any() else float("nan")
        storage_source = "initial_state.node_volume/storage_capacity_m3"
    else:
        storage_utilization = float("nan")
        storage_source = "unavailable_without_graph_storage_capacity"
    return {
        "source": "initial_state",
        "mean_depth": float(np.mean(depth)),
        "high_depth_fraction": float(np.mean(depth > 1.0)),
        "flooding_active_fraction": float(np.mean(flood > 1e-6)),
        "storage_utilization": storage_utilization,
        "storage_source": storage_source,
        "mean_total_inflow": float(np.mean(physical[..., 4])) if physical.shape[-1] > 4 else float("nan"),
        "mean_total_outflow": float(np.mean(physical[..., 5])) if physical.shape[-1] > 5 else float("nan"),
    }


def d3_loss_gradient_conflict_audit_v431(
    *,
    model: torch.nn.Module,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Measure each D3 loss component on the interaction/topology parameters.

    This is a read-only attribution pass.  It never changes parameters or loss
    weights and reports cosine agreement with the direct-TFV and ranking terms.
    """

    target_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target_device).float()
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    set_trainable_phase(model, "d3")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    accum: dict[str, list[torch.Tensor]] = {}
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, target_device)
        output = model.forward_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            batch.candidate_settings, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind="D3",
        )
        terms = _effect_terms(
            output, batch, scales.by_source["D3"],
            magnitude_calibration=True, include_interaction_energy=True,
        )
        for key in (
            "delta_state", "delta_flow", "direct_TFV", "centered_TFV",
            "trajectory_TFV", "ranking", "magnitude_calibration", "interaction_energy",
        ):
            gradients = torch.autograd.grad(
                terms[key], params, retain_graph=True, allow_unused=True
            )
            pieces = [
                gradient.reshape(-1) if gradient is not None else parameter.detach().reshape(-1) * 0.0
                for gradient, parameter in zip(gradients, params, strict=True)
            ]
            vector = torch.cat(pieces) if pieces else torch.zeros(1, device=target_device)
            accum.setdefault(key, []).append(vector.detach())
    reference_vector = None
    ranking_vector = None
    result: dict[str, Any] = {}
    for key, vectors in accum.items():
        vector = torch.stack(vectors).mean(dim=0)
        if key == "direct_TFV":
            reference_vector = vector
        if key == "ranking":
            ranking_vector = vector
        result[key] = {
            "gradient_l2": float(torch.linalg.vector_norm(vector).item()),
            "gradient_linf": float(vector.abs().max().item()),
            "finite_fraction": float(torch.isfinite(vector).float().mean().item()),
        }
    for key, stats in result.items():
        vector = torch.stack(accum[key]).mean(dim=0)
        if reference_vector is not None:
            stats["cosine_vs_direct_TFV"] = float(
                F.cosine_similarity(vector.reshape(1, -1), reference_vector.reshape(1, -1), dim=1).item()
            )
        if ranking_vector is not None:
            stats["cosine_vs_ranking"] = float(
                F.cosine_similarity(vector.reshape(1, -1), ranking_vector.reshape(1, -1), dim=1).item()
            )
    return {
        "parameter_count": len(params),
        "parameter_names": names,
        "components": result,
        "interaction_energy_weight": ResponseLossWeightsV41().interaction_energy,
    }


__all__ = [
    "TRAINING_CONTRACT_V431",
    "PhaseResultV431",
    "current_state_diagnostics_v431",
    "d3_loss_gradient_conflict_audit_v431",
    "d2_single_phase_loss_v431",
    "d3_interaction_phase_loss_v431",
    "reference_phase_loss_v431",
    "train_phase_v431",
    "validate_fresh_parent_lineage_v431",
]
