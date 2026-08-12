"""V4.3.2 bounded audits and primary-TFV D3 training utilities."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v43 import (
    interaction_parameter_names,
    parameter_sha256,
    set_trainable_phase,
)
from .step2_control_response_v432 import project_auxiliary_gradient_v432
from .step2_train_response_v431 import _effect_terms, _weighted_total
from .step2_train_response_v4 import ResponseNormalizationV4, ResponsePairV4
from .step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    ResponseLossWeightsV41,
    evaluate_response_groups_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)


TRAINING_CONTRACT_V432 = "PROJECT7_STEP2_PRIMARY_TFV_CONSISTENCY_V432"


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def d2_prediction_snapshot_v432(output: Any) -> dict[str, torch.Tensor]:
    """Copy every D2 output relevant to the preservation contract."""

    return {
        "delta_states": output.delta_states_physical.detach().clone(),
        "delta_flows": output.delta_flows_physical.detach().clone(),
        "direct_tfv": output.direct_delta_tfv_m3.detach().clone(),
        "trajectory_tfv": output.trajectory_delta_tfv_m3.detach().clone(),
    }


def compare_d2_prediction_snapshots_v432(
    before: dict[str, dict[str, torch.Tensor]],
    after: dict[str, dict[str, torch.Tensor]],
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    invariant = True
    for group in sorted(before):
        if group not in after:
            invariant = False
            rows[group] = {"missing_after": True}
            continue
        differences: dict[str, float] = {}
        for key, value in before[group].items():
            if key not in after[group]:
                invariant = False
                differences[key] = float("inf")
                continue
            delta = (value - after[group][key]).abs()
            differences[key] = float(delta.max().item())
            if not torch.allclose(value, after[group][key], atol=atol, rtol=rtol):
                invariant = False
        rows[group] = differences
    rows["prediction_invariant"] = invariant
    return rows


def classify_first_degradation_v432(
    stage_metrics: dict[str, dict[str, float]],
    *,
    reference_metric: str = "rank",
    tolerance: float = 1e-9,
) -> str:
    """Classify the first phase whose D2 metric falls below the parent."""

    required = ("A0", "A1", "A2", "A3")
    if any(stage not in stage_metrics for stage in required):
        raise ValueError("phase transition audit requires A0, A1, A2, and A3")
    baseline = float(stage_metrics["A0"][reference_metric])
    for stage, label in (("A1", "REFERENCE"), ("A2", "D2"), ("A3", "D3")):
        value = float(stage_metrics[stage][reference_metric])
        if value < baseline - tolerance:
            return label
    return "NONE"


def resolve_best_d2_checkpoint_v432(stage_result_path: str | Path) -> dict[str, Any]:
    """Resolve the selected V4.2 checkpoint from its immutable stage artifact."""

    path = Path(stage_result_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training", {})
    checkpoint_value = training.get("checkpoint") or payload.get("checkpoint")
    if not checkpoint_value:
        raise RuntimeError(f"stage result has no checkpoint: {path}")
    checkpoint = Path(checkpoint_value).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "stage_result": str(path),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "best_epoch": training.get("best_epoch", payload.get("best_epoch")),
        "selection_policy": training.get("selection_policy", payload.get("selection_policy")),
        "group_metric_count": len(payload.get("group_metrics", [])),
    }


def _flatten_gradients(
    gradients: Iterable[torch.Tensor | None],
    parameters: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        pieces.append(
            gradient.reshape(-1)
            if gradient is not None
            else parameter.detach().reshape(-1) * 0.0
        )
    return torch.cat(pieces) if pieces else torch.zeros(1)


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return float("nan")
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1), dim=1).item())


def d3_primary_aux_gradient_audit_v432(
    *,
    model: torch.nn.Module,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Return group-wise gradients and primary/auxiliary cosine distributions."""

    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).float()
    prepared = prepare_graph_v41(model, graph, normalization, target)
    set_trainable_phase(model, "d3")
    parameters = [p for p in model.parameters() if p.requires_grad]
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("D3 gradient audit selected no interaction parameters")
    weights = ResponseLossWeightsV41()
    keys = (
        "delta_state", "delta_flow", "direct_TFV", "centered_TFV",
        "trajectory_TFV", "ranking", "magnitude_calibration",
        "consistency", "interaction_energy",
    )
    primary_keys = ("direct_TFV", "ranking")
    rows: list[dict[str, Any]] = []
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, target)
        output = model.forward_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            batch.candidate_settings, batch.previous_actuator_flow, prepared,
            batch.elapsed_seconds, source_kind="D3",
        )
        terms = _effect_terms(
            output, batch, scales.by_source["D3"],
            magnitude_calibration=True, include_interaction_energy=True,
        )
        gradients: dict[str, torch.Tensor] = {}
        for key in keys:
            term = terms[key]
            if not term.requires_grad:
                gradients[key] = torch.zeros(sum(p.numel() for p in parameters), device=target)
                continue
            gradients[key] = _flatten_gradients(
                torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True),
                parameters,
            ).detach()
        primary = weights.direct_tfv * gradients["direct_TFV"] + weights.ranking * gradients["ranking"]
        row: dict[str, Any] = {
            "group": group,
            "primary_gradient_l2": float(torch.linalg.vector_norm(primary).item()),
            "primary_gradient_linf": float(primary.abs().max().item()),
            "components": {},
        }
        for key in keys:
            vector = gradients[key]
            row["components"][key] = {
                "gradient_l2": float(torch.linalg.vector_norm(vector).item()),
                "gradient_linf": float(vector.abs().max().item()),
                "finite_fraction": float(torch.isfinite(vector).float().mean().item()),
                "cosine_vs_direct_TFV": _safe_cosine(vector, gradients["direct_TFV"]),
                "cosine_vs_ranking": _safe_cosine(vector, gradients["ranking"]),
                "cosine_vs_primary": _safe_cosine(vector, primary),
            }
        row["auxiliary_conflicts_vs_primary"] = {
            key: bool(row["components"][key]["cosine_vs_primary"] < 0.0)
            for key in keys if key not in primary_keys
        }
        rows.append(row)
    summary: dict[str, Any] = {"groups": len(rows), "components": {}}
    for key in keys:
        values = [row["components"][key] for row in rows]
        summary["components"][key] = {
            "gradient_l2_mean": float(np.nanmean([v["gradient_l2"] for v in values])) if values else float("nan"),
            "gradient_linf_mean": float(np.nanmean([v["gradient_linf"] for v in values])) if values else float("nan"),
            "finite_fraction_min": float(np.nanmin([v["finite_fraction"] for v in values])) if values else float("nan"),
            "cosine_vs_direct_TFV_mean": float(np.nanmean([v["cosine_vs_direct_TFV"] for v in values])) if values else float("nan"),
            "cosine_vs_ranking_mean": float(np.nanmean([v["cosine_vs_ranking"] for v in values])) if values else float("nan"),
            "cosine_vs_primary_mean": float(np.nanmean([v["cosine_vs_primary"] for v in values])) if values else float("nan"),
            "fraction_cosine_vs_primary_negative": float(np.mean([v["cosine_vs_primary"] < 0.0 for v in values])) if values else float("nan"),
        }
    return {
        "parameter_names": names,
        "primary_components": list(primary_keys),
        "auxiliary_components": [key for key in keys if key not in primary_keys],
        "fixed_loss_weights": weights.__dict__,
        "summary": summary,
        "groups": rows,
    }


def _component_gradient(
    term: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    total_numel: int,
) -> torch.Tensor:
    if not term.requires_grad:
        return parameters[0].detach().new_zeros(total_numel)
    return _flatten_gradients(
        torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True),
        parameters,
    )


def train_d3_variant_v432(
    *,
    model: torch.nn.Module,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    epochs: int,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 42,
    use_gradient_surgery: bool = False,
) -> dict[str, Any]:
    """Train only D3 interaction parameters with optional primary surgery."""

    if not grouped_pairs:
        raise ValueError("D3 variant requires at least one group")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).float()
    prepared = prepare_graph_v41(model, graph, normalization, target)
    batches = {
        group: stack_response_group_v41(pairs, target)
        for group, pairs in sorted(grouped_pairs.items())
    }
    set_trainable_phase(model, "d3")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("D3 variant selected no trainable interaction parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    weights = ResponseLossWeightsV41()
    history: list[dict[str, Any]] = []
    profile = {"forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
    started = time.perf_counter()
    auxiliaries = (
        ("delta_state", weights.delta_state),
        ("delta_flow", weights.delta_flow),
        ("trajectory_TFV", weights.trajectory_tfv),
        ("centered_TFV", weights.centered_tfv),
        ("magnitude_calibration", weights.magnitude_calibration),
        ("consistency", weights.consistency),
        ("interaction_energy", weights.interaction_energy),
    )
    for epoch in range(1, int(epochs) + 1):
        order = sorted(batches)
        np.random.default_rng(seed + epoch).shuffle(order)
        losses: list[float] = []
        gradients: list[float] = []
        conflicts: list[int] = []
        pre_cosines: list[float] = []
        post_cosines: list[float] = []
        for group in order:
            batch = batches[group]
            tick = time.perf_counter()
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            terms = _effect_terms(
                output, batch, scales.by_source["D3"],
                magnitude_calibration=True, include_interaction_energy=True,
            )
            profile["forward_seconds"] += time.perf_counter() - tick
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            if not use_gradient_surgery:
                loss = _weighted_total(terms, weights, include_reference=False)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
                losses.append(float(loss.detach()))
                conflicts.append(0)
                pre_cosines.append(float("nan"))
                post_cosines.append(float("nan"))
            else:
                total_numel = sum(p.numel() for p in parameters)
                direct_gradient = _component_gradient(terms["direct_TFV"], parameters, total_numel)
                ranking_gradient = _component_gradient(terms["ranking"], parameters, total_numel)
                primary = weights.direct_tfv * direct_gradient + weights.ranking * ranking_gradient
                if not torch.isfinite(primary).all() or torch.linalg.vector_norm(primary) <= 1e-12:
                    raise RuntimeError("zero primary gradient; D3 surgery fail-closed")
                final = primary.clone()
                group_conflicts = 0
                group_pre: list[float] = []
                group_post: list[float] = []
                for key, weight in auxiliaries:
                    raw = _component_gradient(terms[key], parameters, total_numel) * float(weight)
                    pre = _safe_cosine(raw, primary)
                    projected = project_auxiliary_gradient_v432(primary, raw)
                    post = _safe_cosine(projected, primary)
                    if np.isfinite(pre):
                        group_pre.append(pre)
                    if np.isfinite(post):
                        group_post.append(post)
                    if np.isfinite(pre) and pre < 0.0:
                        group_conflicts += 1
                    final = final + projected
                if not torch.isfinite(final).all():
                    raise FloatingPointError("non-finite post-surgery D3 gradient")
                offset = 0
                for parameter in parameters:
                    count = parameter.numel()
                    parameter.grad = final[offset : offset + count].reshape_as(parameter).clone()
                    offset += count
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
                loss = _weighted_total(terms, weights, include_reference=False)
                losses.append(float(loss.detach()))
                conflicts.append(group_conflicts)
                pre_cosines.extend(group_pre)
                post_cosines.extend(group_post)
            profile["backward_seconds"] += time.perf_counter() - tick
            if not np.isfinite(grad_norm):
                raise FloatingPointError("non-finite D3 gradient")
            gradients.append(grad_norm)
            tick = time.perf_counter()
            optimizer.step()
            profile["optimizer_seconds"] += time.perf_counter() - tick
        metrics, _ = evaluate_response_groups_v41(
            model=model, grouped_pairs=grouped_pairs, prepared=prepared,
            device=target, batches=batches,
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(gradients)),
                "conflicting_auxiliaries": int(np.sum(conflicts)),
                "pre_projection_cosine_mean": float(np.nanmean(pre_cosines)) if any(np.isfinite(pre_cosines)) else float("nan"),
                "post_projection_cosine_mean": float(np.nanmean(post_cosines)) if any(np.isfinite(post_cosines)) else float("nan"),
                "metrics": metrics,
            }
        )
    profile["wall_time_seconds"] = time.perf_counter() - started
    model.eval()
    metrics, contributions = evaluate_response_groups_v41(
        model=model, grouped_pairs=grouped_pairs, prepared=prepared,
        device=target, batches=batches,
    )
    return {
        "use_gradient_surgery": bool(use_gradient_surgery),
        "epochs_completed": len(history),
        "history": history,
        "group_metrics": metrics,
        "candidate_contributions": contributions,
        "profile_seconds": profile,
        "interaction_parameter_sha256": parameter_sha256(model, interaction_parameter_names(model)),
    }


__all__ = [
    "TRAINING_CONTRACT_V432",
    "classify_first_degradation_v432",
    "compare_d2_prediction_snapshots_v432",
    "d2_prediction_snapshot_v432",
    "d3_primary_aux_gradient_audit_v432",
    "resolve_best_d2_checkpoint_v432",
    "train_d3_variant_v432",
]
