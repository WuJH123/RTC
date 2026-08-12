"""Bounded source-balanced training for the V4.5 direct response operator."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import numpy as np
import torch

from .step2_control_response_v45 import (
    DifferentiableCounterfactualResponseModelV45,
    direct_effect_parameter_names_v45,
    set_trainable_direct_effect_v45,
)
from .step2_control_response_v43 import parameter_sha256
from .step2_train_response_v431 import _effect_terms, _weighted_total
from .step2_train_response_v4 import ResponseNormalizationV4, ResponsePairV4
from .step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    ResponseLossWeightsV41,
    evaluate_response_groups_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)


TRAINING_CONTRACT_V45 = "PROJECT7_STEP2_DIRECT_MULTI_ACTUATOR_V45"


def _sample_gpu_telemetry() -> tuple[float, float] | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
        utilization, memory_used = (float(value.strip()) for value in raw.split(",", 1))
        return utilization, memory_used
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def train_direct_multi_actuator_v45(
    *,
    model: DifferentiableCounterfactualResponseModelV45,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    epochs: int,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, Any]:
    """Fit only the direct effect branch, giving every group one optimizer step."""

    if not grouped_pairs:
        raise ValueError("V4.5 requires at least one complete group")
    if not all(pairs and pairs[0].source_kind.upper() in {"D2", "D3"} for pairs in grouped_pairs.values()):
        raise ValueError("V4.5 groups must be D2 or D3")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).float()
    prepared = prepare_graph_v41(model, graph, normalization, target)
    batches = {group: stack_response_group_v41(pairs, target) for group, pairs in sorted(grouped_pairs.items())}
    names = set_trainable_direct_effect_v45(model, enabled=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("V4.5 selected no trainable direct parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    weights = ResponseLossWeightsV41()
    history: list[dict[str, Any]] = []
    profile = {"data_load_seconds": 0.0, "forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
    gpu_samples: list[tuple[float, float]] = []
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        order = sorted(batches)
        np.random.default_rng(seed + epoch).shuffle(order)
        losses: list[float] = []
        gradients: list[float] = []
        source_components: dict[str, dict[str, list[float]]] = {"D2": {}, "D3": {}}
        # Accumulate one source-balanced group objective before stepping.  A
        # sequential D2-step then D3-step lets the latter overwrite the former
        # even though both sources are explicitly balanced; one joint update
        # makes the documented D2+D3 objective literal.
        optimizer.zero_grad(set_to_none=True)
        for group in order:
            batch = batches[group]
            source = batch.source_kind.upper()
            tick = time.perf_counter()
            model.train()
            output = model.forward_group(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                batch.elapsed_seconds,
                source_kind=source,
            )
            terms = _effect_terms(
                output,
                batch,
                scales.by_source[source],
                magnitude_calibration=source == "D3",
                include_interaction_energy=False,
            )
            profile["forward_seconds"] += time.perf_counter() - tick
            loss = _weighted_total(terms, weights, include_reference=False)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V4.5 loss at epoch {epoch}, group {group}")
            tick = time.perf_counter()
            (loss / float(len(order))).backward()
            profile["backward_seconds"] += time.perf_counter() - tick
            losses.append(float(loss.detach()))
            terms_for_log = source_components[source]
            for name, value in terms.items():
                terms_for_log.setdefault(name, []).append(float(value.detach()))
        tick = time.perf_counter()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
        profile["optimizer_seconds"] += time.perf_counter() - tick
        if not np.isfinite(gradient):
            raise FloatingPointError(f"non-finite V4.5 gradient at epoch {epoch}")
        gradients.append(gradient)
        tick = time.perf_counter()
        optimizer.step()
        profile["optimizer_seconds"] += time.perf_counter() - tick
        metrics, _ = evaluate_response_groups_v41(
            model=model,
            grouped_pairs=grouped_pairs,
            prepared=prepared,
            device=target,
            batches=batches,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "gradient_norm": float(np.mean(gradients)),
            "groups": len(order),
        }
        for source in ("D2", "D3"):
            source_rows = [item for item in metrics if item["source_kind"] == source]
            if source_rows:
                row[f"{source.lower()}_rank"] = float(np.nanmean([item["rank"] for item in source_rows]))
                row[f"{source.lower()}_pairwise"] = float(np.nanmean([item["pairwise"] for item in source_rows]))
                row[f"{source.lower()}_sign"] = float(np.nanmean([item["sign"] for item in source_rows]))
                row[f"{source.lower()}_top1"] = int(sum(bool(item["top1"]) for item in source_rows))
                row[f"{source.lower()}_mean_regret_m3"] = float(np.mean([item["regret_m3"] for item in source_rows]))
                row[f"{source.lower()}_max_regret_m3"] = float(np.max([item["regret_m3"] for item in source_rows]))
            for name, values in source_components[source].items():
                row[f"{source.lower()}_{name}"] = float(np.mean(values))
        history.append(row)
        sample = _sample_gpu_telemetry() if target.type == "cuda" else None
        if sample is not None:
            gpu_samples.append(sample)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        sample = _sample_gpu_telemetry()
        if sample is not None:
            gpu_samples.append(sample)
    profile["wall_time_seconds"] = time.perf_counter() - started
    if target.type == "cuda":
        profile.update(
            {
                "gpu_name": torch.cuda.get_device_name(target),
                "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(target)),
                "gpu_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(target)),
            }
        )
        if gpu_samples:
            util = np.asarray([sample[0] for sample in gpu_samples], dtype=np.float64)
            memory = np.asarray([sample[1] for sample in gpu_samples], dtype=np.float64)
            profile.update(
                {
                    "gpu_utilization_mean_percent": float(util.mean()),
                    "gpu_utilization_p90_percent": float(np.percentile(util, 90)),
                    "gpu_utilization_max_percent": float(util.max()),
                    "gpu_memory_used_mean_mib": float(memory.mean()),
                    "gpu_memory_used_p90_mib": float(np.percentile(memory, 90)),
                    "gpu_memory_used_max_mib": float(memory.max()),
                    "gpu_telemetry_samples": len(gpu_samples),
                }
            )
    model.eval()
    metrics, contributions = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        prepared=prepared,
        device=target,
        batches=batches,
    )
    return {
        "contract": TRAINING_CONTRACT_V45,
        "epochs_completed": len(history),
        "trainable_parameter_names": sorted(names),
        "history": history,
        "group_metrics": metrics,
        "candidate_contributions": contributions,
        "profile_seconds": profile,
        "direct_effect_parameter_sha256": parameter_sha256(model, names),
    }


__all__ = ["TRAINING_CONTRACT_V45", "train_direct_multi_actuator_v45"]
