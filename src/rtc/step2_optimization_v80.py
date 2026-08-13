"""Staged optimizer for Project7 Step2 V8.0 direct hydraulic effects."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v80 import DirectHydraulicEffectSurrogateV80, prepare_static_v80
from .step2_hydraulic_objective_v80 import hydraulic_effect_loss_v80
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v80_contract import DirectHydraulicEffectLossContractV80


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        entry = cache.entry(name)
        grouped[f"{entry.rainfall_group}::{entry.event_id}"].append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def _emit_progress(payload: dict[str, Any]) -> None:
    fields = " ".join(
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in payload.items()
    )
    print(f"[V8_HYDRAULIC_EFFECT] {fields}", flush=True)


def _summarize(stage: str, epoch: int, records: list[dict[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": stage, "epoch": int(epoch)}
    for key in sorted({name for record in records for name in record}):
        values = [float(record[key]) for record in records if np.isfinite(float(record[key]))]
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


def train_hydraulic_effect_v80(
    model: DirectHydraulicEffectSurrogateV80,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    onset_positive_weight: float,
    graph: Any,
    device: str = "cuda",
    seed: int = 42,
    contract: DirectHydraulicEffectLossContractV80 = DirectHydraulicEffectLossContractV80(),
) -> list[dict[str, Any]]:
    """D2 single-actuator pretraining followed by D3-primary multi-actuator learning."""
    contract.validate()
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V8 effect training requires D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + 31)
    np.random.seed(seed + 31)
    model.to(target).train()
    prepared = prepare_static_v80(graph, target)
    trainable = list(model.trainable_parameters())
    if not trainable:
        raise RuntimeError("V8 direct effect model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )
    rng = np.random.default_rng(seed + 31)
    d2_events = _events(cache, fit_d2_names)
    d3_events = _events(cache, fit_d3_names)
    history: list[dict[str, Any]] = []

    def backward_group(name: str, weight: float) -> dict[str, float]:
        batch = cache.batch(name, normalization, target)
        output = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            batch.previous_actuator_flow,
            prepared,
        )
        loss, metrics = hydraulic_effect_loss_v80(
            output,
            batch,
            normalization,
            scales,
            onset_positive_weight=onset_positive_weight,
            contract=contract,
        )
        (weight * loss).backward()
        return metrics

    # Stage 1: use clean single-actuator counterfactuals to teach spatial response before
    # asking the model to represent multi-actuator interaction.
    for epoch in range(1, contract.d2_pretrain_epochs + 1):
        keys = list(d2_events)
        rng.shuffle(keys)
        records: list[dict[str, float]] = []
        gradient_norms: list[float] = []
        for key in keys:
            optimizer.zero_grad(set_to_none=True)
            groups = d2_events[key]
            records.extend(backward_group(name, 1.0 / len(groups)) for name in groups)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            gradient_norms.append(float(gradient_norm.detach()))
            optimizer.step()
        row = _summarize("D2_direct_spatial_effect", epoch, records)
        row["gradient_norm"] = float(np.mean(gradient_norms)) if gradient_norms else float("nan")
        history.append(row)
        _emit_progress(row)

    # Stage 2: targeted D3 is primary; D2 remains an anchor against losing single-actuator effects.
    d2_keys = list(d2_events)
    for epoch in range(1, contract.d3_joint_epochs + 1):
        d3_keys = list(d3_events)
        rng.shuffle(d3_keys)
        rng.shuffle(d2_keys)
        records = []
        gradient_norms = []
        for index, key in enumerate(d3_keys):
            optimizer.zero_grad(set_to_none=True)
            d3_groups = d3_events[key]
            d2_groups = d2_events[d2_keys[index % len(d2_keys)]]
            records.extend(
                backward_group(
                    name, contract.d3_weight / len(d3_groups)
                )
                for name in d3_groups
            )
            records.extend(
                backward_group(
                    name, contract.d2_anchor_weight / len(d2_groups)
                )
                for name in d2_groups
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            gradient_norms.append(float(gradient_norm.detach()))
            optimizer.step()
        row = _summarize("D3_direct_effect_with_D2_anchor", epoch, records)
        row["gradient_norm"] = float(np.mean(gradient_norms)) if gradient_norms else float("nan")
        history.append(row)
        _emit_progress(row)
    return history


__all__ = ["train_hydraulic_effect_v80"]
