"""Train-only phased utilities for the isolated V4.3 prototype."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    ResponseGroupBatchV41,
    evaluate_response_groups_v41,
    prepare_graph_v41,
    response_group_loss_v41,
    stack_response_group_v41,
)


TRAINING_CONTRACT_V43 = "STEP2_STATE_TOPOLOGY_INTERACTION_V43_TRAIN_ONLY"


@dataclass(frozen=True)
class PhaseResultV43:
    phase: str
    selected_groups: tuple[str, ...]
    reference_group_count: int
    epochs_completed: int
    history: list[dict[str, Any]]
    parameter_sha256: str
    profile_seconds: dict[str, float]


def _reference_signature(pairs: list[ResponsePairV4]) -> str:
    if not pairs:
        raise ValueError("cannot hash an empty response group")
    reference = pairs[0].reference
    digest = hashlib.sha256()
    for name in ("initial_state", "rainfall", "settings", "previous_actuator_flow"):
        array = np.asarray(reference[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def deduplicate_reference_groups_v43(
    grouped_pairs: dict[str, list[ResponsePairV4]],
) -> dict[str, list[ResponsePairV4]]:
    """Keep one complete group per exact same-prefix reference signature."""

    selected: dict[str, list[ResponsePairV4]] = {}
    seen: set[str] = set()
    for group, pairs in sorted(grouped_pairs.items()):
        signature = _reference_signature(pairs)
        if signature in seen:
            continue
        seen.add(signature)
        selected[group] = pairs
    return selected


def _phase_prefixes(phase: str, model: torch.nn.Module) -> tuple[str, ...]:
    key = phase.lower()
    if key == "reference":
        return reference_parameter_names(model)
    if key == "d2":
        return single_parameter_names(model)
    if key == "d3":
        return interaction_parameter_names(model)
    raise ValueError("phase must be reference, d2, or d3")


def _phase_metric_summary(
    model: DifferentiableCounterfactualResponseModelV43,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    prepared: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return evaluate_response_groups_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        prepared=prepared,
        device=device,
    )


def train_phase_v43(
    *,
    model: DifferentiableCounterfactualResponseModelV43,
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
    magnitude_calibration: bool = True,
) -> PhaseResultV43:
    """Fit exactly one phase with an explicit disjoint parameter set."""

    if not grouped_pairs:
        raise ValueError("V4.3 phase requires at least one complete group")
    phase_key = phase.lower()
    selected_names = set_trainable_phase(model, phase_key)
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(
        device if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model.to(target_device).float()
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    batches = {
        group: stack_response_group_v41(pairs, target_device)
        for group, pairs in sorted(grouped_pairs.items())
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"phase {phase_key} selected no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    profile = {"data_load_seconds": 0.0, "forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        order = sorted(batches)
        np.random.default_rng(seed + epoch).shuffle(order)
        model.train()
        loss_sum = 0.0
        gradient_norms: list[float] = []
        components_sum: dict[str, float] = {}
        for group in order:
            batch = batches[group]
            stamp = time.perf_counter()
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
            loss, components = response_group_loss_v41(
                output,
                batch,
                scales.by_source[batch.source_kind],
                normalization,
                magnitude_calibration=(magnitude_calibration and phase_key == "d3"),
            )
            profile["forward_seconds"] += time.perf_counter() - stamp
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V4.3 {phase_key} loss at epoch {epoch}, group {group}")
            optimizer.zero_grad(set_to_none=True)
            stamp = time.perf_counter()
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 5.0))
            profile["backward_seconds"] += time.perf_counter() - stamp
            if not np.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite V4.3 {phase_key} gradient at epoch {epoch}, group {group}")
            stamp = time.perf_counter()
            optimizer.step()
            profile["optimizer_seconds"] += time.perf_counter() - stamp
            loss_sum += float(loss.detach())
            gradient_norms.append(gradient_norm)
            for name, value in components.items():
                components_sum[name] = components_sum.get(name, 0.0) + float(value)
        metrics, contributions = _phase_metric_summary(model, grouped_pairs, prepared, target_device)
        d2 = [row for row in metrics if row["source_kind"] == "D2"]
        d3 = [row for row in metrics if row["source_kind"] == "D3"]
        row = {
            "epoch": epoch,
            "phase": phase_key,
            "loss": loss_sum / len(order),
            "gradient_norm": float(np.mean(gradient_norms)),
            "rank": float(np.nanmean([item["rank"] for item in metrics])),
            "pairwise": float(np.nanmean([item["pairwise"] for item in metrics])),
            "sign": float(np.nanmean([item["sign"] for item in metrics])),
            "top1_fraction": float(np.mean([item["top1"] for item in metrics])),
            "d2_rank": float(np.nanmean([item["rank"] for item in d2])) if d2 else float("nan"),
            "d3_rank": float(np.nanmean([item["rank"] for item in d3])) if d3 else float("nan"),
            "d3_pairwise": float(np.nanmean([item["pairwise"] for item in d3])) if d3 else float("nan"),
            "d3_top1_fraction": float(np.mean([item["top1"] for item in d3])) if d3 else float("nan"),
        }
        row.update({name: value / len(order) for name, value in components_sum.items()})
        history.append(row)
    profile["wall_time_seconds"] = time.perf_counter() - started
    model.eval()
    metrics, contributions = _phase_metric_summary(model, grouped_pairs, prepared, target_device)
    output = Path(out_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": TRAINING_CONTRACT_V43,
            "phase": phase_key,
            "model_state_dict": model.state_dict(),
            "seed": int(seed),
            "precision": "fp32",
            "selected_parameter_names": sorted(selected_names),
            "group_metrics": metrics,
            "candidate_contributions": contributions,
        },
        output,
    )
    output.with_suffix(output.suffix + ".history.json").write_text(
        json.dumps(
            {
                "contract": TRAINING_CONTRACT_V43,
                "phase": phase_key,
                "history": history,
                "profile_seconds": profile,
                "parameter_sha256": parameter_sha256(model, selected_names),
            },
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PhaseResultV43(
        phase=phase_key,
        selected_groups=tuple(sorted(grouped_pairs)),
        reference_group_count=len(deduplicate_reference_groups_v43(grouped_pairs)),
        epochs_completed=len(history),
        history=history,
        parameter_sha256=parameter_sha256(model, selected_names),
        profile_seconds=profile,
    )


def d2_prediction_snapshot_v43(
    model: DifferentiableCounterfactualResponseModelV43,
    batch: ResponseGroupBatchV41,
    prepared: Any,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        output = model.forward_group(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            batch.previous_actuator_flow,
            prepared,
            batch.elapsed_seconds,
            source_kind="D2",
        )
    return {
        "delta_states": output.delta_states_physical.detach().clone(),
        "delta_flows": output.delta_flows_physical.detach().clone(),
        "direct_tfv": output.direct_delta_tfv_m3.detach().clone(),
        "trajectory_tfv": output.trajectory_delta_tfv_m3.detach().clone(),
    }


def compare_d2_prediction_snapshots_v43(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, float | bool]:
    keys = tuple(before)
    differences = {
        f"{key}_max_abs": float((before[key] - after[key]).abs().max().item())
        for key in keys
    }
    differences["prediction_invariant"] = bool(
        all(value <= 1e-7 for name, value in differences.items() if name.endswith("max_abs"))
    )
    return differences


__all__ = [
    "TRAINING_CONTRACT_V43",
    "PhaseResultV43",
    "compare_d2_prediction_snapshots_v43",
    "deduplicate_reference_groups_v43",
    "d2_prediction_snapshot_v43",
    "train_phase_v43",
]
