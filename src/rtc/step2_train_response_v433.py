"""Bounded V4.3.3 residual-only development training utilities."""

from __future__ import annotations

import time
import subprocess
from typing import Any

import numpy as np
import torch

from .step2_control_response_v433 import set_trainable_nodewise_residual_v433
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


TRAINING_CONTRACT_V433 = "PROJECT7_STEP2_NODEWISE_TFV_CORRECTNESS_V433"


def _sample_gpu_telemetry() -> tuple[float, float] | None:
    """Best-effort utilization and used-memory sample for the active GPU."""

    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
        utilization, memory_used = (float(value.strip()) for value in raw.split(",", 1))
        return utilization, memory_used
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def train_nodewise_residual_v433(
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
) -> dict[str, Any]:
    """Train only the zero-initialized nodewise residual parameters."""

    if not grouped_pairs:
        raise ValueError("V4.3.3 residual training requires at least one group")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).float()
    prepared = prepare_graph_v41(model, graph, normalization, target)
    batches = {
        group: stack_response_group_v41(pairs, target)
        for group, pairs in sorted(grouped_pairs.items())
    }
    names = set_trainable_nodewise_residual_v433(model, enabled=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("V4.3.3 residual phase selected no parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    weights = ResponseLossWeightsV41()
    history: list[dict[str, Any]] = []
    profile = {"forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
    gpu_samples: list[tuple[float, float]] = []
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        order = sorted(batches)
        np.random.default_rng(seed + epoch).shuffle(order)
        losses: list[float] = []
        gradients: list[float] = []
        for group in order:
            batch = batches[group]
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
                source_kind="D3",
            )
            terms = _effect_terms(
                output,
                batch,
                scales.by_source["D3"],
                magnitude_calibration=True,
                include_interaction_energy=True,
            )
            profile["forward_seconds"] += time.perf_counter() - tick
            loss = _weighted_total(terms, weights, include_reference=False)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V4.3.3 loss at epoch {epoch}, group {group}")
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
            profile["backward_seconds"] += time.perf_counter() - tick
            if not np.isfinite(gradient):
                raise FloatingPointError(f"non-finite V4.3.3 gradient at epoch {epoch}, group {group}")
            tick = time.perf_counter()
            optimizer.step()
            profile["optimizer_seconds"] += time.perf_counter() - tick
            losses.append(float(loss.detach()))
            gradients.append(gradient)
        metrics, _ = evaluate_response_groups_v41(
            model=model,
            grouped_pairs=grouped_pairs,
            prepared=prepared,
            device=target,
            batches=batches,
        )
        d3 = [row for row in metrics if row["source_kind"] == "D3"]
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(gradients)),
                "d3_rank": float(np.nanmean([row["rank"] for row in d3])),
                "d3_pairwise": float(np.nanmean([row["pairwise"] for row in d3])),
                "d3_sign": float(np.nanmean([row["sign"] for row in d3])),
                "d3_top1": int(sum(bool(row["top1"]) for row in d3)),
                "d3_max_regret_m3": float(max(row["regret_m3"] for row in d3)),
            }
        )
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
            utilization = np.asarray([sample[0] for sample in gpu_samples], dtype=np.float64)
            memory_used = np.asarray([sample[1] for sample in gpu_samples], dtype=np.float64)
            profile.update(
                {
                    "gpu_utilization_mean_percent": float(utilization.mean()),
                    "gpu_utilization_p90_percent": float(np.percentile(utilization, 90)),
                    "gpu_utilization_max_percent": float(utilization.max()),
                    "gpu_memory_used_mean_mib": float(memory_used.mean()),
                    "gpu_memory_used_p90_mib": float(np.percentile(memory_used, 90)),
                    "gpu_memory_used_max_mib": float(memory_used.max()),
                    "gpu_telemetry_samples": len(gpu_samples),
                }
            )
        else:
            profile["gpu_telemetry_samples"] = 0
    model.eval()
    metrics, contributions = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        prepared=prepared,
        device=target,
        batches=batches,
    )
    return {
        "contract": TRAINING_CONTRACT_V433,
        "epochs_completed": len(history),
        "trainable_parameter_names": sorted(names),
        "history": history,
        "group_metrics": metrics,
        "candidate_contributions": contributions,
        "profile_seconds": profile,
        "interaction_parameter_sha256": parameter_sha256(model, names),
    }


__all__ = ["TRAINING_CONTRACT_V433", "train_nodewise_residual_v433"]
